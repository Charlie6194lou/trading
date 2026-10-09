"""Chargement, normalisation, sauvegarde et téléchargement de bougies OHLCV.

Le format canonique d'un DataFrame de bougies (voir ``docs/ARCHITECTURE.md``) :

- index ``DatetimeIndex`` **tz-aware UTC**, trié croissant, sans doublon, nommé
  ``timestamp`` ;
- colonnes ``open, high, low, close, volume`` en ``float64``, en minuscules.

La fonction :func:`ensure_ohlcv` convertit à peu près n'importe quel CSV ou DataFrame
« à la mode Yahoo / Binance / TradingView » vers ce format, et c'est elle qui est
appelée à l'entrée de chaque composant du robot.

Le téléchargement (:func:`fetch_ohlcv`) repose sur ``ccxt`` : l'objet ``exchange`` est
injectable, ce qui permet de tester toute la pagination sans réseau.
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "OHLCV_COLUMNS",
    "cache_path",
    "ensure_ohlcv",
    "fetch_ohlcv",
    "infer_periods_per_year",
    "load_csv",
    "load_or_fetch",
    "resample_ohlcv",
    "save_csv",
    "timeframe_to_timedelta",
]

#: colonnes canoniques, dans l'ordre
OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")
#: colonnes obligatoires (``volume`` est optionnelle : 0.0 si absente)
REQUIRED_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")
#: nom de l'index dans le format canonique
INDEX_NAME = "timestamp"

# alias (en minuscules, espaces -> "_") acceptés pour chaque colonne de prix / volume
_COLUMN_ALIASES: dict[str, str] = {
    "open": "open",
    "o": "open",
    "high": "high",
    "h": "high",
    "low": "low",
    "l": "low",
    "close": "close",
    "c": "close",
    "volume": "volume",
    "vol": "volume",
    "v": "volume",
}
# alias de la colonne temporelle, par ordre de priorité
_TIME_ALIASES: tuple[str, ...] = (
    "timestamp",
    "datetime",
    "date",
    "time",
    "open_time",
    "ts",
)
# colonnes connues que l'on ignore volontairement
_IGNORED_ALIASES: frozenset[str] = frozenset({"adj_close", "adjclose", "adj._close"})

_TIMEFRAME_RE = re.compile(r"^\s*(\d+)\s*([mhdwHDW])\s*$")
# unité de timeframe -> alias d'offset pandas pour `resample` (semaines calées sur le lundi)
_RESAMPLE_ALIASES: dict[str, str] = {
    "m": "min",
    "h": "h",
    "H": "h",
    "d": "D",
    "D": "D",
    "w": "W-MON",
    "W": "W-MON",
}
_TIMEFRAME_UNITS: dict[str, str] = {
    "m": "minutes",
    "h": "hours",
    "H": "hours",
    "d": "days",
    "D": "days",
    "w": "weeks",
    "W": "weeks",
}

_EPOCH_MS = pd.Timestamp(0, tz="UTC")
_MS = pd.Timedelta(milliseconds=1)
#: résolution par défaut des DatetimeIndex de la version de pandas installée
#: ("ns" avec pandas 2.x, "us" avec pandas 3) : les horodatages analysés l'adoptent, pour
#: être identiques à ceux de ``pd.date_range`` ; un DatetimeIndex fourni garde sa résolution.
_DEFAULT_UNIT: str = pd.DatetimeIndex(["2000-01-01"]).unit


# --- normalisation ------------------------------------------------------------------


def _normalise_name(name: Any) -> str:
    """``" Adj Close "`` -> ``"adj_close"`` : clé de recherche des alias."""
    return re.sub(r"\s+", "_", str(name).strip().lower())


def _epoch_unit(values: np.ndarray) -> str:
    """Devine l'unité d'un tableau d'entiers « epoch » d'après son ordre de grandeur."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "ms"
    magnitude = float(np.max(np.abs(finite)))
    if magnitude >= 1e17:
        return "ns"
    if magnitude >= 1e14:
        return "us"
    if magnitude >= 1e11:
        return "ms"
    return "s"


def _parse_timestamps(values: Any, *, source: str) -> pd.DatetimeIndex:
    """Convertit une colonne / un index quelconque en ``DatetimeIndex`` UTC.

    Accepte : datetimes (naïfs = UTC, tz-aware convertis), entiers epoch en secondes,
    millisecondes, microsecondes ou nanosecondes (unité devinée), chaînes ISO 8601 ou
    chaînes de chiffres (epoch). Lève ``ValueError`` si rien ne convient. Un
    ``DatetimeIndex`` fourni garde sa résolution ; les valeurs analysées prennent la
    résolution par défaut de pandas (``_DEFAULT_UNIT``).
    """
    if isinstance(values, pd.Index):
        arr = values
    else:
        arr = pd.Index(values)

    try:
        if isinstance(arr, pd.DatetimeIndex):
            parsed = arr
        elif pd.api.types.is_bool_dtype(arr):
            raise ValueError("valeurs booléennes")
        elif pd.api.types.is_numeric_dtype(arr):
            numbers = np.asarray(arr, dtype="float64")
            parsed = pd.to_datetime(numbers, unit=_epoch_unit(numbers), utc=True)
        else:
            # chaînes : d'abord des chiffres (epoch), sinon ISO 8601, sinon libre.
            as_num = pd.to_numeric(arr, errors="coerce")
            non_null = arr.notna()
            if non_null.any() and bool(as_num[non_null].notna().all()):
                numbers = np.asarray(as_num, dtype="float64")
                parsed = pd.to_datetime(numbers, unit=_epoch_unit(numbers), utc=True)
            else:
                try:
                    parsed = pd.to_datetime(arr, utc=True, format="ISO8601")
                except (ValueError, TypeError):
                    parsed = pd.to_datetime(arr, utc=True, format="mixed")
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(
            f"Impossible d'interpréter {source} comme des dates/heures : {exc}"
        ) from exc

    keep_unit = isinstance(arr, pd.DatetimeIndex)
    parsed = pd.DatetimeIndex(parsed)
    if parsed.tz is None:
        parsed = parsed.tz_localize("UTC")
    else:
        parsed = parsed.tz_convert("UTC")
    return parsed if keep_unit else parsed.as_unit(_DEFAULT_UNIT)


def _try_parse_timestamps(values: Any, *, source: str) -> pd.DatetimeIndex | None:
    """Variante tolérante de :func:`_parse_timestamps` : ``None`` si ça ne ressemble pas
    à des dates (utilisée pour deviner la première colonne)."""
    arr = values if isinstance(values, pd.Index) else pd.Index(values)
    if len(arr) == 0 or pd.api.types.is_bool_dtype(arr):
        return None
    if pd.api.types.is_numeric_dtype(arr):
        numbers = np.asarray(arr, dtype="float64")
        finite = numbers[np.isfinite(numbers)]
        # un epoch plausible : > 1e8 s (~1973) ; sinon ce sont des prix, pas des dates
        if finite.size == 0 or float(np.min(finite)) < 1e8:
            return None
    try:
        parsed = _parse_timestamps(arr, source=source)
    except ValueError:
        return None
    if parsed.isna().all():
        return None
    return parsed


def _resolve_time_index(df: pd.DataFrame, col_map: dict[Any, str]) -> tuple[pd.DatetimeIndex, Any]:
    """Trouve l'axe temporel : index datetime, colonne alias, index nommé, 1re colonne.

    Renvoie ``(index_utc, colonne_utilisée_ou_None)``.
    """
    # 1. index déjà temporel
    if isinstance(df.index, pd.DatetimeIndex):
        return _parse_timestamps(df.index, source="l'index"), None

    # 2. colonne portant un alias connu (priorité : timestamp > datetime > date > time)
    by_alias: dict[str, Any] = {}
    for col in df.columns:
        key = _normalise_name(col)
        if key in _TIME_ALIASES and key not in by_alias:
            by_alias[key] = col
    for alias in _TIME_ALIASES:
        if alias in by_alias:
            col = by_alias[alias]
            return _parse_timestamps(df[col], source=f"la colonne {col!r}"), col

    # 3. index nommé comme une colonne temporelle
    if df.index.name is not None and _normalise_name(df.index.name) in _TIME_ALIASES:
        return _parse_timestamps(df.index, source=f"l'index {df.index.name!r}"), None

    # 4. première colonne non-OHLCV qui ressemble à des dates
    if len(df.columns) > 0:
        first = df.columns[0]
        if first not in col_map:
            parsed = _try_parse_timestamps(df[first], source=f"la colonne {first!r}")
            if parsed is not None:
                return parsed, first

    # 5. index non temporel mais interprétable (ex. CSV lu avec index_col=0)
    if not isinstance(df.index, pd.RangeIndex):
        parsed = _try_parse_timestamps(df.index, source="l'index")
        if parsed is not None:
            return parsed, None

    raise ValueError(
        "Impossible de trouver la colonne de dates : attendu un index DatetimeIndex ou une "
        "colonne nommée 'timestamp', 'date', 'datetime' ou 'time' (ISO 8601 ou epoch)."
    )


def ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise et valide un DataFrame de bougies vers le format canonique.

    - colonnes reconnues quelle que soit la casse (``Open``, ``HIGH``...), alias ``Vol``,
      colonne ``Adj Close`` ignorée, ``volume`` mise à ``0.0`` si absente ;
    - axe temporel : index ``DatetimeIndex``, ou colonne ``timestamp`` / ``date`` /
      ``datetime`` / ``time`` (ISO 8601, epoch secondes ou millisecondes), ou à défaut la
      première colonne si elle ressemble à des dates ;
    - horodatages naïfs considérés UTC, tz-aware convertis en UTC ;
    - index trié, doublons supprimés (la dernière occurrence est gardée), lignes avec
      ``NaN`` sur open/high/low/close supprimées, valeurs en ``float64``.

    Renvoie toujours un **nouveau** DataFrame ; l'entrée n'est pas modifiée.

    Lève ``ValueError`` si une colonne obligatoire manque ou si le résultat est vide.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"`df` doit être un pd.DataFrame, reçu {type(df).__name__}")
    if len(df) == 0:
        raise ValueError("Le DataFrame de bougies est vide : aucune ligne à traiter.")

    # correspondance colonne source -> colonne canonique
    col_map: dict[Any, str] = {}
    seen: dict[str, Any] = {}
    for col in df.columns:
        key = _normalise_name(col)
        if key in _IGNORED_ALIASES:
            continue
        target = _COLUMN_ALIASES.get(key)
        if target is None:
            continue
        if target in seen:
            raise ValueError(
                f"Colonne {target!r} ambiguë : {seen[target]!r} et {col!r} désignent la même chose."
            )
        seen[target] = col
        col_map[col] = target

    missing = [c for c in REQUIRED_COLUMNS if c not in seen]
    if missing:
        raise ValueError(
            f"Colonnes OHLCV manquantes : {missing} (colonnes reçues : {list(df.columns)})."
        )

    index, _time_col = _resolve_time_index(df, col_map)

    data: dict[str, np.ndarray] = {}
    for target in OHLCV_COLUMNS:
        if target in seen:
            series = pd.to_numeric(df[seen[target]], errors="coerce")
            data[target] = np.asarray(series, dtype="float64")
        else:  # volume absente
            data[target] = np.zeros(len(df), dtype="float64")

    out = pd.DataFrame(data, index=index)
    out.index.name = INDEX_NAME

    # nettoyage : index NaT, NaN sur les prix, volume NaN -> 0
    valid = out.index.notna()
    for col in REQUIRED_COLUMNS:
        valid &= np.isfinite(out[col].to_numpy())
    dropped = int((~valid).sum())
    if dropped:
        logger.warning("ensure_ohlcv : %d ligne(s) ignorée(s) (date ou prix invalide).", dropped)
        out = out.loc[valid]
    if out["volume"].isna().any():
        out = out.assign(volume=out["volume"].fillna(0.0))

    out = out.sort_index(kind="stable")
    duplicated = out.index.duplicated(keep="last")
    if duplicated.any():
        logger.warning(
            "ensure_ohlcv : %d horodatage(s) en double supprimé(s).", int(duplicated.sum())
        )
        out = out.loc[~duplicated]

    if len(out) == 0:
        raise ValueError(
            "Le DataFrame de bougies est vide après nettoyage (dates ou prix invalides)."
        )

    incoherent = int(((out["high"] < out["low"]).to_numpy()).sum())
    if incoherent:
        logger.warning("ensure_ohlcv : %d bougie(s) avec high < low.", incoherent)

    out = out.loc[:, list(OHLCV_COLUMNS)]
    out.index = _with_inferred_freq(out.index)
    out.index.name = INDEX_NAME
    return out


def _with_inferred_freq(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Renseigne ``index.freq`` quand l'index est parfaitement régulier (ex. ``"h"``), pour
    qu'un aller-retour CSV redonne un index identique à celui de ``pd.date_range``."""
    if index.freq is not None or len(index) < 3:
        return index
    try:
        inferred = pd.infer_freq(index)
    except (ValueError, TypeError):
        return index
    if inferred is None:
        return index
    new_index = index.copy()
    try:
        new_index.freq = inferred
    except (ValueError, TypeError):
        return index
    return new_index


def _empty_ohlcv() -> pd.DataFrame:
    """DataFrame OHLCV vide mais au bon format (utile quand un téléchargement ne renvoie rien)."""
    index = pd.DatetimeIndex([], tz="UTC", name=INDEX_NAME).as_unit(_DEFAULT_UNIT)
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in OHLCV_COLUMNS}, index=index)


# --- CSV ----------------------------------------------------------------------------


def _sniff_separator(path: Path) -> str:
    """Devine le séparateur du CSV d'après sa première ligne (``,`` ou ``;`` ou tabulation)."""
    with path.open("r", encoding="utf-8-sig", errors="replace") as fh:
        header = fh.readline()
    for sep in (",", ";", "\t"):
        if sep in header:
            return sep
    return ","


def load_csv(path: str | Path) -> pd.DataFrame:
    """Charge un CSV de bougies et le normalise via :func:`ensure_ohlcv`.

    Le CSV peut avoir la date en première colonne (nommée ou non) ou dans une colonne
    ``timestamp`` / ``date`` / ``datetime`` / ``time``, au format ISO 8601 ou epoch
    (secondes ou millisecondes). Séparateur ``,`` ``;`` ou tabulation.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Fichier CSV introuvable : {path}")
    sep = _sniff_separator(path)
    raw = pd.read_csv(path, sep=sep, encoding="utf-8-sig")
    if len(raw) == 0:
        raise ValueError(f"Le fichier CSV {path} ne contient aucune bougie.")
    df = ensure_ohlcv(raw)
    logger.info(
        "%d bougies chargées depuis %s (%s -> %s)", len(df), path, df.index[0], df.index[-1]
    )
    return df


def _iso_strings(index: pd.DatetimeIndex) -> pd.Index:
    """Horodatages ISO 8601 (``2024-01-01T00:00:00+00:00``), exacts même avec fractions."""
    idx = index.tz_convert("UTC")
    if bool((idx == idx.floor("s")).all()):
        return idx.strftime("%Y-%m-%dT%H:%M:%S") + "+00:00"
    return pd.Index([ts.isoformat() for ts in idx])


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Sauvegarde des bougies en CSV (horodatages ISO 8601 UTC, colonnes canoniques).

    ``load_csv(path)`` redonne exactement le même DataFrame. Les répertoires parents sont
    créés si besoin.
    """
    path = Path(path)
    if len(df) == 0:
        out = _empty_ohlcv()
    else:
        out = ensure_ohlcv(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    to_write = out.copy()
    to_write.index = _iso_strings(out.index)
    to_write.index.name = INDEX_NAME
    to_write.to_csv(path, index=True, lineterminator="\n")
    logger.info("%d bougies sauvegardées dans %s", len(out), path)


# --- timeframes ---------------------------------------------------------------------


def timeframe_to_timedelta(tf: str) -> pd.Timedelta:
    """Convertit un timeframe ccxt (``"1m"``, ``"4h"``, ``"1d"``, ``"1w"``) en ``pd.Timedelta``.

    Unités acceptées : ``m`` (minute), ``h`` (heure), ``d`` (jour), ``w`` (semaine).
    Lève ``ValueError`` pour toute autre forme (``"1M"`` mois, ``"h"``, ``"0m"``...).
    """
    if not isinstance(tf, str):
        raise TypeError(f"Le timeframe doit être une chaîne (ex. '1h'), reçu {type(tf).__name__}")
    match = _TIMEFRAME_RE.match(tf)
    if match is None:
        raise ValueError(
            f"Timeframe invalide : {tf!r}. Attendu <entier><unité> avec unité m, h, d ou w "
            "(ex. 1m, 15m, 1h, 4h, 1d, 1w)."
        )
    count = int(match.group(1))
    if count <= 0:
        raise ValueError(f"Timeframe invalide : {tf!r} (le nombre doit être >= 1).")
    unit = _TIMEFRAME_UNITS[match.group(2)]
    return pd.Timedelta(**{unit: count})


def infer_periods_per_year(index: pd.Index | pd.Series | pd.DataFrame) -> float:
    """Nombre de bougies par an déduit de l'écart médian de l'index (marché 24h/24, 7j/7).

    ``1h`` -> 8760, ``4h`` -> 2190, ``1d`` -> 365, ``1m`` -> 525600. Avec moins de deux
    horodatages valides, renvoie ``365.0`` par défaut.
    """
    if isinstance(index, pd.DataFrame | pd.Series):
        index = index.index
    if not isinstance(index, pd.DatetimeIndex):
        try:
            index = pd.DatetimeIndex(index)
        except (ValueError, TypeError) as exc:
            raise TypeError(
                "`index` doit être un DatetimeIndex (ou un objet qui en porte un)."
            ) from exc
    if len(index) < 2:
        logger.debug("infer_periods_per_year : index trop court, valeur par défaut 365.")
        return 365.0
    diffs = pd.Series(index).diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        return 365.0
    median = diffs.median()
    return float(pd.Timedelta(days=365) / median)


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Agrège des bougies vers un timeframe plus large (ex. ``"4h"``, ``"1d"``, ``"1w"``).

    open = première, high = max, low = min, close = dernière, volume = somme ; les bougies
    vides (aucune donnée dans l'intervalle) sont supprimées. Les intervalles sont alignés
    sur minuit UTC (semaines : lundi 00:00 UTC), comme sur les exchanges crypto.
    ``rule`` accepte un timeframe ccxt ou un alias d'offset pandas (``"4h"``, ``"D"``).
    """
    data = ensure_ohlcv(df)
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    kwargs: dict[str, Any] = {"closed": "left", "label": "left"}
    match = _TIMEFRAME_RE.match(rule) if isinstance(rule, str) else None
    if match is not None:
        count = int(match.group(1))
        if count <= 0:
            raise ValueError(f"Timeframe invalide : {rule!r} (le nombre doit être >= 1).")
        alias = _RESAMPLE_ALIASES[match.group(2)]
        # alias pandas plutôt que Timedelta : même `freq` que celle déduite par pd.infer_freq.
        # `origin="epoch"` cale les intervalles intra-journaliers sur minuit UTC ; les
        # jours / semaines y sont déjà alignés (et pandas refuse `origin` pour eux).
        if alias in ("min", "h"):
            kwargs["origin"] = "epoch"
        resampler = data.resample(f"{count}{alias}", **kwargs)
    else:
        resampler = data.resample(rule, **kwargs)
    out = resampler.agg(agg)
    out = out.loc[out["open"].notna()]
    out = out.astype("float64")
    index = pd.DatetimeIndex(out.index).tz_convert("UTC")
    index.freq = (
        None  # on ne garde que la fréquence réellement observée après suppression des trous
    )
    out.index = _with_inferred_freq(index)
    out.index.name = INDEX_NAME
    return out.loc[:, list(OHLCV_COLUMNS)]


# --- téléchargement ccxt ------------------------------------------------------------


def _to_ms(value: Any, exchange: Any = None) -> int | None:
    """Convertit ``None`` / int (ms) / float / datetime / Timestamp / chaîne ISO en epoch ms."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("Une date ne peut pas être un booléen.")
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        if not math.isfinite(value):
            raise ValueError(f"Date invalide : {value!r}")
        return int(value)
    if isinstance(value, str):
        parse8601 = getattr(exchange, "parse8601", None)
        if callable(parse8601):
            try:
                parsed = parse8601(value)
            except Exception:  # noqa: BLE001 - ccxt peut lever n'importe quoi
                parsed = None
            if isinstance(parsed, int | float) and not isinstance(parsed, bool):
                return int(parsed)
        try:
            ts = pd.Timestamp(value)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Date invalide : {value!r} (attendu ISO 8601, ex. 2024-01-01)."
            ) from exc
    elif isinstance(value, pd.Timestamp):
        ts = value
    elif isinstance(value, datetime):
        ts = pd.Timestamp(value)
    else:
        raise TypeError(f"Type de date non supporté : {type(value).__name__}")
    if pd.isna(ts):
        raise ValueError(f"Date invalide : {value!r}")
    ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
    return int((ts - _EPOCH_MS) // _MS)


def _build_exchange(exchange_id: str) -> Any:
    """Instancie ``ccxt.<exchange_id>({'enableRateLimit': True})`` (import paresseux)."""
    try:
        import ccxt
    except ImportError as exc:  # pragma: no cover - dépendance déclarée
        raise ImportError(
            "La bibliothèque `ccxt` est requise pour télécharger des données."
        ) from exc
    cls = getattr(ccxt, exchange_id, None)
    if cls is None or not isinstance(cls, type):
        raise ValueError(
            f"Exchange ccxt inconnu : {exchange_id!r}. Exemples valides : binance, kraken, "
            "coinbase, bybit, okx."
        )
    return cls({"enableRateLimit": True})


def fetch_ohlcv(
    exchange_id: str,
    symbol: str,
    timeframe: str,
    since: Any = None,
    until: Any = None,
    limit: int | None = None,
    exchange: Any = None,
    *,
    drop_last_incomplete: bool = True,
    page_size: int = 1000,
    sleep: Callable[[float], Any] = time.sleep,
    now_ms: int | None = None,
) -> pd.DataFrame:
    """Télécharge des bougies via ccxt avec pagination, et renvoie le format canonique.

    - ``since`` / ``until`` : ISO 8601 (``"2024-01-01"``), ``datetime`` ou epoch ms ;
      ``until`` est inclusif (heure d'ouverture de la bougie). Sans ``since``, l'exchange
      renvoie les bougies les plus récentes.
    - ``limit`` : nombre maximal de bougies renvoyées (``None`` = pas de plafond).
    - ``exchange`` : objet ccxt déjà construit (ou un faux pour les tests) exposant
      ``fetch_ohlcv(symbol, timeframe, since, limit)``, ``parse8601``, ``milliseconds``,
      ``rateLimit`` et ``timeframes`` ; s'il est ``None``, ``ccxt.<exchange_id>`` est créé.
    - ``drop_last_incomplete`` : retire la dernière bougie si elle n'est pas encore clôturée
      (``ouverture + timeframe > maintenant``) ; ``now_ms`` permet de fixer « maintenant ».
    - ``page_size`` : bougies demandées par appel (1000 sur Binance ; à réduire pour les
      exchanges plus restrictifs). La pagination s'arrête quand un appel renvoie moins de
      ``page_size`` bougies, quand ``until`` ou ``limit`` est atteint, ou faute de progrès.
    - ``sleep`` : fonction d'attente appelée avec ``rateLimit / 1000`` secondes entre deux
      appels (injectable dans les tests).
    """
    tf_delta = timeframe_to_timedelta(timeframe)
    tf_ms = int(tf_delta // _MS)
    if limit is not None and limit <= 0:
        raise ValueError(f"`limit` doit être >= 1 ou None, reçu {limit}.")
    if page_size <= 0:
        raise ValueError(f"`page_size` doit être >= 1, reçu {page_size}.")

    if exchange is None:
        exchange = _build_exchange(exchange_id)

    supported = getattr(exchange, "timeframes", None)
    if isinstance(supported, dict) and supported and timeframe not in supported:
        raise ValueError(
            f"Timeframe {timeframe!r} non supporté par {exchange_id} "
            f"(disponibles : {', '.join(sorted(supported))})."
        )

    since_ms = _to_ms(since, exchange)
    until_ms = _to_ms(until, exchange)
    if since_ms is not None and until_ms is not None and until_ms < since_ms:
        raise ValueError("`until` doit être postérieur à `since`.")

    rate_limit_ms = getattr(exchange, "rateLimit", 0) or 0
    pause = max(float(rate_limit_ms) / 1000.0, 0.0)

    rows: list[Sequence[Any]] = []
    cursor = since_ms
    n_calls = 0
    while True:
        batch_limit = page_size if limit is None else min(page_size, limit - len(rows))
        if batch_limit <= 0:
            break
        if n_calls > 0 and pause > 0:
            sleep(pause)
        batch = exchange.fetch_ohlcv(symbol, timeframe, cursor, batch_limit)
        n_calls += 1
        batch = list(batch or [])
        if not batch:
            break
        for row in batch:
            ts = int(row[0])
            if cursor is not None and ts < cursor:
                continue
            if until_ms is not None and ts > until_ms:
                continue
            rows.append(row)
        last_ts = int(batch[-1][0])
        if len(batch) < batch_limit:
            break
        if until_ms is not None and last_ts >= until_ms:
            break
        if limit is not None and len(rows) >= limit:
            break
        next_cursor = last_ts + 1
        if cursor is not None and next_cursor <= cursor:
            logger.warning("fetch_ohlcv : l'exchange ne progresse plus (since=%s), arrêt.", cursor)
            break
        cursor = next_cursor

    logger.info(
        "fetch_ohlcv : %d bougies brutes %s %s sur %s en %d appel(s).",
        len(rows),
        symbol,
        timeframe,
        exchange_id,
        n_calls,
    )
    if not rows:
        logger.warning("fetch_ohlcv : aucune bougie reçue pour %s %s.", symbol, timeframe)
        return _empty_ohlcv()

    raw = pd.DataFrame(
        [list(r[:6]) + [0.0] * (6 - len(r[:6])) for r in rows],
        columns=["timestamp", *OHLCV_COLUMNS],
    )
    df = ensure_ohlcv(raw)
    if limit is not None and len(df) > limit:
        df = df.iloc[:limit]

    if drop_last_incomplete and len(df) > 0:
        if now_ms is None:
            milliseconds = getattr(exchange, "milliseconds", None)
            now_ms = int(milliseconds()) if callable(milliseconds) else int(time.time() * 1000)
        last_open_ms = int((df.index[-1] - _EPOCH_MS) // _MS)
        if last_open_ms + tf_ms > now_ms:
            logger.debug("fetch_ohlcv : dernière bougie %s non clôturée, ignorée.", df.index[-1])
            df = df.iloc[:-1]
    return df


# --- cache --------------------------------------------------------------------------


def _sanitise(text: str) -> str:
    """``"BTC/USDT:USDT"`` -> ``"BTC-USDT-USDT"`` : nom de fichier sûr."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(text).strip())
    return cleaned.strip("-") or "unknown"


def cache_path(cache_dir: str | Path, exchange_id: str, symbol: str, timeframe: str) -> Path:
    """Chemin du CSV de cache : ``<cache_dir>/<exchange>_<SYMBOLE>_<timeframe>.csv``."""
    name = f"{_sanitise(exchange_id)}_{_sanitise(symbol)}_{_sanitise(timeframe)}.csv"
    return Path(cache_dir) / name


def load_or_fetch(
    exchange_id: str,
    symbol: str,
    timeframe: str,
    since: Any = None,
    cache_dir: str | Path = "data/cache",
    refresh: bool = False,
    exchange: Any = None,
    **fetch_kwargs: Any,
) -> pd.DataFrame:
    """Renvoie les bougies du cache CSV si présent (et ``refresh`` faux), sinon télécharge
    via :func:`fetch_ohlcv` puis sauvegarde dans le cache.

    Les arguments nommés supplémentaires (``until``, ``limit``, ``sleep``, ``now_ms``...)
    sont transmis à :func:`fetch_ohlcv`.
    """
    path = cache_path(cache_dir, exchange_id, symbol, timeframe)
    if path.is_file() and not refresh:
        logger.info("Cache trouvé : %s", path)
        return load_csv(path)
    df = fetch_ohlcv(exchange_id, symbol, timeframe, since=since, exchange=exchange, **fetch_kwargs)
    if len(df) > 0:
        save_csv(df, path)
    else:
        logger.warning("Rien à mettre en cache pour %s %s (%s).", symbol, timeframe, exchange_id)
    return df
