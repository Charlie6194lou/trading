"""Chargement, normalisation, sauvegarde et téléchargement de bougies OHLCV.

Le format canonique d'un DataFrame de bougies (voir ``docs/ARCHITECTURE.md``) :

- index ``DatetimeIndex`` **tz-aware UTC**, trié croissant, sans doublon, nommé
  ``timestamp`` ;
- colonnes ``open, high, low, close, volume`` en ``float64``, en minuscules.

La fonction :func:`ensure_ohlcv` convertit à peu près n'importe quel CSV ou DataFrame
« à la mode Yahoo / Binance / TradingView / Excel » vers ce format, et c'est elle qui est
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
    "AmbiguousDateError",
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
#: jours de bourse par an pour les marchés fermés le week-end (actions, forex)
TRADING_DAYS_PER_YEAR = 252.0
#: jours par an pour les marchés ouverts 24h/24, 7j/7 (crypto)
CALENDAR_DAYS_PER_YEAR = 365.0

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
# colonnes connues que l'on ignore volontairement (``Adj Close`` : voir ensure_ohlcv)
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
_DAY_MS = 86_400_000
#: résolution par défaut des DatetimeIndex de la version de pandas installée
#: ("ns" avec pandas 2.x, "us" avec pandas 3) : les horodatages analysés l'adoptent, pour
#: être identiques à ceux de ``pd.date_range`` ; un DatetimeIndex fourni garde sa résolution.
_DEFAULT_UNIT: str = pd.DatetimeIndex(["2000-01-01"]).unit

# --- formats de dates reconnus (chaînes) ---
# JJ/MM/AAAA ou MM/JJ/AAAA (séparateur / . ou -), heure facultative
_DMY_RE = (
    r"^\s*(\d{1,2})([/.\-])(\d{1,2})\2(\d{4})"
    r"(?:[ T](\d{1,2}):(\d{2})(?::(\d{2})(\.\d+)?)?)?\s*$"
)
# AAAA/MM/JJ ou AAAA.MM.JJ (TradingView, certains exports), heure facultative
_YMD_RE = (
    r"^\s*(\d{4})([/.])(\d{1,2})\2(\d{1,2})"
    r"(?:[ T](\d{1,2}):(\d{2})(?::(\d{2})(\.\d+)?)?)?\s*$"
)
# date ISO 8601 seule (``2024-01-31``)
_DATE_ONLY_RE = re.compile(r"^\s*\d{4}-\d{2}-\d{2}\s*$")
# date ISO 8601 avec heure facultative (``2024-01-31``, ``2024-01-31 12:00``, ``...T12:00:00Z``)
_ISO_DATETIME_RE = re.compile(
    r"^\s*\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?\s*(?:Z|[+-]\d{2}:?\d{2})?\s*$"
)
# nombre « epoch » (secondes, millisecondes, microsecondes ou nanosecondes)
_DIGITS_RE = re.compile(r"^\s*\d+(?:\.\d+)?\s*$")
# heure seule (``10:00``, ``10:00:00``, ``10:00:00.250``) : colonne Time à combiner avec Date
_TIME_OF_DAY_RE = re.compile(r"^\s*\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?\s*$")
_CONTAINS_TIME_RE = re.compile(r"\d{1,2}:\d{2}")
# nombre décimal à virgule (``1234,56``) dans un CSV ``;`` ou tabulation
_DECIMAL_COMMA_RE = re.compile(r"^\s*-?\d+,\d+\s*$")


class AmbiguousDateError(ValueError):
    """Dates du type ``01/02/2024`` dont on ne peut pas dire si c'est JJ/MM ou MM/JJ."""


# --- normalisation ------------------------------------------------------------------


def _normalise_name(name: Any) -> str:
    """``" Adj Close "`` -> ``"adj_close"`` : clé de recherche des alias."""
    return re.sub(r"\s+", "_", str(name).strip().lower())


def _epoch_unit(values: np.ndarray) -> str:
    """Devine l'unité d'un tableau d'entiers « epoch » d'après son ordre de grandeur.

    ``< 1e11`` secondes (jusqu'en 5138), ``< 1e14`` millisecondes, ``< 1e17`` microsecondes,
    au-delà nanosecondes. Même heuristique pour les colonnes CSV et pour ``since`` / ``until``.
    """
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


def _parse_day_month_year(arr: pd.Index, *, source: str, dayfirst: bool | None) -> pd.DatetimeIndex:
    """Analyse des dates ``JJ/MM/AAAA`` / ``MM/JJ/AAAA`` / ``AAAA/MM/JJ`` (heure facultative)
    avec un ordre jour/mois **explicite**, jamais deviné ligne par ligne.

    ``dayfirst=None`` : détection automatique sur toute la colonne (jour en premier dès qu'un
    premier champ dépasse 12, mois en premier dès qu'un second champ dépasse 12) ; si les deux
    lectures sont possibles pour toutes les lignes, lève :class:`AmbiguousDateError`.
    """
    text = pd.Series(arr, dtype="object")
    mask = text.notna().to_numpy()
    strings = text[mask].astype(str).str.strip()

    dmy = strings.str.extract(_DMY_RE)
    if len(strings) and dmy[0].notna().all():
        first = dmy[0].astype(int)
        second = dmy[2].astype(int)
        year = dmy[3]
        if dayfirst is None:
            if (first > 12).any():
                dayfirst = True
            elif (second > 12).any():
                dayfirst = False
            else:
                raise AmbiguousDateError(
                    f"Format de date ambigu dans {source} (JJ/MM/AAAA ou MM/JJ/AAAA ? ex. "
                    f"{strings.iloc[0]!r}) : utilisez le format AAAA-MM-JJ, ou précisez "
                    "dayfirst=True (jour en premier) / dayfirst=False (mois en premier)."
                )
        day, month = (dmy[0], dmy[2]) if dayfirst else (dmy[2], dmy[0])
        hour, minute, second_, fraction = dmy[4], dmy[5], dmy[6], dmy[7]
    else:
        ymd = strings.str.extract(_YMD_RE)
        if not len(strings) or not ymd[0].notna().all():
            raise ValueError(
                "format de date non reconnu : utilisez ISO 8601 (AAAA-MM-JJ ou "
                "AAAA-MM-JJ HH:MM:SS), JJ/MM/AAAA, ou un epoch en secondes / millisecondes"
            )
        year, month, day = ymd[0], ymd[2], ymd[3]
        hour, minute, second_, fraction = ymd[4], ymd[5], ymd[6], ymd[7]

    iso = (
        year
        + "-"
        + month.str.zfill(2)
        + "-"
        + day.str.zfill(2)
        + "T"
        + hour.fillna("0").str.zfill(2)
        + ":"
        + minute.fillna("00")
        + ":"
        + second_.fillna("00")
        + fraction.fillna("")
    )
    full = np.full(len(text), None, dtype=object)
    full[mask] = iso.to_numpy()
    return pd.DatetimeIndex(pd.to_datetime(pd.Index(full), utc=True, format="ISO8601"))


def _parse_timestamps(
    values: Any, *, source: str, dayfirst: bool | None = None
) -> pd.DatetimeIndex:
    """Convertit une colonne / un index quelconque en ``DatetimeIndex`` UTC.

    Accepte : datetimes (naïfs = UTC, tz-aware convertis), entiers epoch en secondes,
    millisecondes, microsecondes ou nanosecondes (unité devinée), chaînes de chiffres
    (epoch), chaînes ISO 8601, et dates ``JJ/MM/AAAA`` / ``MM/JJ/AAAA`` / ``AAAA/MM/JJ``
    (ordre jour/mois détecté sur toute la colonne, voir :func:`_parse_day_month_year` ;
    ``dayfirst`` force l'ordre). Lève ``ValueError`` si rien ne convient — jamais de
    devinette ligne par ligne (``format="mixed"``), qui mélangerait JJ/MM et MM/JJ.
    Un ``DatetimeIndex`` fourni garde sa résolution ; les valeurs analysées prennent la
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
            # chaînes : d'abord des chiffres (epoch), sinon ISO 8601, sinon JJ/MM/AAAA.
            as_num = pd.to_numeric(arr, errors="coerce")
            non_null = arr.notna()
            if non_null.any() and bool(as_num[non_null].notna().all()):
                numbers = np.asarray(as_num, dtype="float64")
                parsed = pd.to_datetime(numbers, unit=_epoch_unit(numbers), utc=True)
            else:
                try:
                    parsed = pd.to_datetime(arr, utc=True, format="ISO8601")
                except (ValueError, TypeError):
                    parsed = _parse_day_month_year(arr, source=source, dayfirst=dayfirst)
    except AmbiguousDateError:
        raise
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


def _try_parse_timestamps(
    values: Any, *, source: str, dayfirst: bool | None = None
) -> pd.DatetimeIndex | None:
    """Variante tolérante de :func:`_parse_timestamps` : ``None`` si ça ne ressemble pas
    à des dates (utilisée pour deviner la première colonne). Une ambiguïté JJ/MM est
    propagée : la colonne est bien une colonne de dates, c'est le format qui pose problème."""
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
        parsed = _parse_timestamps(arr, source=source, dayfirst=dayfirst)
    except AmbiguousDateError:
        raise
    except ValueError:
        return None
    if parsed.isna().all():
        return None
    return parsed


def _looks_like_time_of_day(series: pd.Series) -> bool:
    """``True`` si toutes les valeurs non nulles sont des heures seules (``10:00``, ``10:00:00``)."""
    values = series.dropna()
    if len(values) == 0 or pd.api.types.is_numeric_dtype(values):
        return False
    return bool(values.astype(str).str.match(_TIME_OF_DAY_RE).all())


def _combine_date_and_time(date: pd.Series, time_: pd.Series) -> pd.Index | None:
    """``Date`` (sans heure) + ``Time`` (heure seule) -> chaînes ``"<date> <heure>"``.

    ``None`` si la colonne ``Date`` contient déjà une heure, est numérique (epoch), ou si
    ``Time`` n'est pas une heure seule : on retombe alors sur la colonne prioritaire.
    """
    if pd.api.types.is_numeric_dtype(date) or isinstance(date.dtype, pd.DatetimeTZDtype):
        return None
    if pd.api.types.is_datetime64_any_dtype(date):
        return None
    if not _looks_like_time_of_day(time_):
        return None
    date_text = date.astype("object").where(date.notna(), None)
    non_null = date.notna().to_numpy()
    if non_null.any():
        sample = pd.Series(date_text[non_null]).astype(str)
        if sample.str.contains(_CONTAINS_TIME_RE).any():
            return None
    both = non_null & time_.notna().to_numpy()
    combined = np.full(len(date), None, dtype=object)
    combined[both] = (
        pd.Series(date_text[both]).astype(str).str.strip()
        + " "
        + pd.Series(time_.to_numpy()[both]).astype(str).str.strip()
    ).to_numpy()
    return pd.Index(combined)


def _resolve_time_index(
    df: pd.DataFrame, col_map: dict[Any, str], *, dayfirst: bool | None = None
) -> tuple[pd.DatetimeIndex, Any]:
    """Trouve l'axe temporel : index datetime, colonne alias, index nommé, 1re colonne.

    Des colonnes ``Date`` (sans heure) et ``Time`` (heure seule) séparées — exports
    MetaTrader / Excel — sont combinées. Renvoie ``(index_utc, colonne_utilisée_ou_None)``.
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
    if (
        "date" in by_alias
        and "time" in by_alias
        and "timestamp" not in by_alias
        and "datetime" not in by_alias
    ):
        date_col, time_col = by_alias["date"], by_alias["time"]
        combined = _combine_date_and_time(df[date_col], df[time_col])
        if combined is not None:
            source = f"les colonnes {date_col!r} + {time_col!r}"
            return _parse_timestamps(combined, source=source, dayfirst=dayfirst), date_col
    for alias in _TIME_ALIASES:
        if alias in by_alias:
            col = by_alias[alias]
            parsed = _parse_timestamps(df[col], source=f"la colonne {col!r}", dayfirst=dayfirst)
            return parsed, col

    # 3. index nommé comme une colonne temporelle
    if df.index.name is not None and _normalise_name(df.index.name) in _TIME_ALIASES:
        source = f"l'index {df.index.name!r}"
        return _parse_timestamps(df.index, source=source, dayfirst=dayfirst), None

    # 4. première colonne non-OHLCV qui ressemble à des dates
    if len(df.columns) > 0:
        first = df.columns[0]
        if first not in col_map:
            parsed = _try_parse_timestamps(
                df[first], source=f"la colonne {first!r}", dayfirst=dayfirst
            )
            if parsed is not None:
                return parsed, first

    # 5. index non temporel mais interprétable (ex. CSV lu avec index_col=0)
    if not isinstance(df.index, pd.RangeIndex):
        parsed = _try_parse_timestamps(df.index, source="l'index", dayfirst=dayfirst)
        if parsed is not None:
            return parsed, None

    raise ValueError(
        "Impossible de trouver la colonne de dates : attendu un index DatetimeIndex ou une "
        "colonne nommée 'timestamp', 'date', 'datetime' ou 'time' (ISO 8601 ou epoch)."
    )


def ensure_ohlcv(df: pd.DataFrame, *, dayfirst: bool | None = None) -> pd.DataFrame:
    """Normalise et valide un DataFrame de bougies vers le format canonique.

    - colonnes reconnues quelle que soit la casse (``Open``, ``HIGH``...), alias ``Vol``,
      ``volume`` mise à ``0.0`` si absente ; la colonne ``Adj Close`` (Yahoo) est
      **ignorée** : les prix bruts, non ajustés des dividendes/splits, sont conservés ;
    - axe temporel : index ``DatetimeIndex``, ou colonne ``timestamp`` / ``date`` /
      ``datetime`` / ``time`` (ISO 8601, epoch secondes ou millisecondes, ``JJ/MM/AAAA``),
      ou colonnes ``Date`` + ``Time`` séparées (combinées), ou à défaut la première colonne
      si elle ressemble à des dates ;
    - ``dayfirst`` : ordre jour/mois des dates ``01/02/2024`` ; ``None`` (défaut) = détecté
      sur toute la colonne, :class:`AmbiguousDateError` si indécidable ;
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

    index, _time_col = _resolve_time_index(df, col_map, dayfirst=dayfirst)

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
        n_dup = int(duplicated.sum())
        if len(out) >= 10 and n_dup > 0.1 * len(out):
            logger.warning(
                "ensure_ohlcv : %d horodatage(s) en double supprimé(s) sur %d lignes : la "
                "colonne de dates ne contient probablement pas l'heure (colonnes Date et Time "
                "séparées ? format sans heure ?). Vérifiez le fichier.",
                n_dup,
                len(out),
            )
        else:
            logger.warning("ensure_ohlcv : %d horodatage(s) en double supprimé(s).", n_dup)
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


def _sniff_decimal(path: Path, sep: str) -> str:
    """``","`` si un CSV ``;`` / tabulation contient des nombres à virgule (``1234,56``, export
    Excel français), sinon ``"."``. Avec le séparateur ``,`` la virgule décimale est impossible."""
    if sep == ",":
        return "."
    with path.open("r", encoding="utf-8-sig", errors="replace") as fh:
        for _ in range(20):
            line = fh.readline()
            if not line:
                break
            if any(_DECIMAL_COMMA_RE.match(field) for field in line.rstrip("\r\n").split(sep)):
                return ","
    return "."


def _is_headerless(columns: pd.Index, decimal: str) -> bool:
    """``True`` si tous les « noms de colonnes » sont des nombres : la première ligne du
    fichier est une bougie, pas un en-tête (exports Binance Vision, Kraken OHLCVT)."""
    names = [str(c).replace(",", ".") if decimal == "," else str(c) for c in columns]
    return len(names) > 0 and bool(pd.to_numeric(pd.Index(names), errors="coerce").notna().all())


def _positional_names(raw: pd.DataFrame) -> list[str]:
    """Noms des colonnes d'un CSV sans en-tête : ``timestamp, open, high, low, close, volume``.

    Fichier Kraken OHLCVT (``timestamp, open, high, low, close, vwap, volume, count``) :
    si la 6e colonne est un prix (toujours entre low et high) et pas la 7e, la 6e est le
    ``vwap`` et la 7e le ``volume``.
    """
    n = raw.shape[1]
    names = ["timestamp", "open", "high", "low", "close"]
    names += [f"col{i}" for i in range(5, n)]
    if n >= 6:
        names[5] = "volume"
    if n >= 7:
        high = pd.to_numeric(raw.iloc[:, 2], errors="coerce").to_numpy(dtype="float64")
        low = pd.to_numeric(raw.iloc[:, 3], errors="coerce").to_numpy(dtype="float64")
        sixth = pd.to_numeric(raw.iloc[:, 5], errors="coerce").to_numpy(dtype="float64")
        seventh = pd.to_numeric(raw.iloc[:, 6], errors="coerce").to_numpy(dtype="float64")
        sixth_is_price = bool(np.all((low <= sixth) & (sixth <= high)))
        seventh_is_price = bool(np.all((low <= seventh) & (seventh <= high)))
        if sixth_is_price and not seventh_is_price:
            names[5], names[6] = "vwap", "volume"
    return names


def load_csv(path: str | Path, *, dayfirst: bool | None = None) -> pd.DataFrame:
    """Charge un CSV de bougies et le normalise via :func:`ensure_ohlcv`.

    Formats acceptés :

    - date en première colonne (nommée ou non) ou dans une colonne ``timestamp`` / ``date``
      / ``datetime`` / ``time`` (ISO 8601, epoch secondes ou millisecondes, ``JJ/MM/AAAA`` —
      ``dayfirst`` force l'ordre jour/mois, voir :func:`ensure_ohlcv`), ou colonnes ``Date``
      + ``Time`` séparées ;
    - séparateur ``,`` ``;`` ou tabulation ; virgule décimale (``1234,56``) détectée dans
      les fichiers ``;`` / tabulation (exports Excel français) ;
    - fichier **sans en-tête** dont la première ligne est numérique (Binance Vision,
      Kraken OHLCVT) : colonnes supposées ``timestamp, open, high, low, close, volume``
      (Kraken : ``vwap`` puis ``volume``), avec un avertissement journalisé.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Fichier CSV introuvable : {path}")
    sep = _sniff_separator(path)
    decimal = _sniff_decimal(path, sep)
    if decimal == ",":
        logger.info("%s : virgule décimale détectée (export Excel français).", path)
    read_kwargs: dict[str, Any] = {"sep": sep, "decimal": decimal, "encoding": "utf-8-sig"}
    raw = pd.read_csv(path, **read_kwargs)
    if raw.shape[1] >= 5 and _is_headerless(raw.columns, decimal):
        raw = pd.read_csv(path, header=None, **read_kwargs)
        raw.columns = _positional_names(raw)
        logger.warning(
            "%s : fichier sans en-tête, colonnes supposées %s.",
            path,
            ", ".join(c for c in raw.columns if not c.startswith("col")),
        )
    if len(raw) == 0:
        raise ValueError(f"Le fichier CSV {path} ne contient aucune bougie.")
    df = ensure_ohlcv(raw, dayfirst=dayfirst)
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


#: durée minimale pour décider qu'un marché est fermé le week-end (au moins un week-end
#: complet encadré par des jours de semaine)
_WEEKDAY_DETECTION_SPAN = pd.Timedelta(days=11)
#: part maximale de bougies tombant un samedi/dimanche pour un marché « jours ouvrés »
#: (le forex ouvre le dimanche soir : 2 bougies horaires sur 120 ≈ 1,7 %)
_WEEKEND_SHARE_MAX = 0.05


def infer_periods_per_year(index: pd.Index | pd.Series | pd.DataFrame) -> float:
    """Nombre de bougies par an déduit de l'index, pour annualiser rendements et volatilité.

    - Marché ouvert 24h/24, 7j/7 (crypto) : ``365 jours / écart médian`` entre bougies :
      ``1h`` -> 8760, ``4h`` -> 2190, ``1d`` -> 365, ``1m`` -> 525600, ``1w`` -> 52.14.
    - Marché fermé le week-end (actions, forex : moins de 5 % des bougies un samedi ou un
      dimanche, sur au moins 11 jours) et bougies ≤ 1 jour : ``252 jours de bourse ×
      bougies par jour`` (médiane du nombre de bougies par date) : ``1d`` -> 252,
      ``1h`` forex (24 h/jour) -> 6048, ``1h`` actions (7 bougies/séance) -> 1764.
    - Avec moins de deux horodatages valides, renvoie ``365.0`` par défaut.
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
        return CALENDAR_DAYS_PER_YEAR
    diffs = pd.Series(index).diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        return CALENDAR_DAYS_PER_YEAR
    median = diffs.median()
    one_day = pd.Timedelta(days=1)
    valid = index.dropna()
    span = valid.max() - valid.min()
    if median <= one_day and span >= _WEEKDAY_DETECTION_SPAN:
        weekend_share = float(np.mean(valid.dayofweek.to_numpy() >= 5))
        if weekend_share < _WEEKEND_SHARE_MAX:
            bars_per_day = float(
                pd.Series(1, index=valid).groupby(valid.normalize()).size().median()
            )
            logger.debug(
                "infer_periods_per_year : marché fermé le week-end, %.0f jours × %.1f bougies/jour",
                TRADING_DAYS_PER_YEAR,
                bars_per_day,
            )
            return TRADING_DAYS_PER_YEAR * bars_per_day
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


def _epoch_to_ms(number: float) -> int:
    """Nombre « epoch » -> millisecondes, unité devinée comme pour les colonnes CSV
    (:func:`_epoch_unit`) : ``1704067200`` (s) et ``1704067200000`` (ms) donnent le même résultat."""
    unit = _epoch_unit(np.asarray([number], dtype="float64"))
    if unit == "s":
        return int(round(number * 1000))
    if unit == "ms":
        return int(number)
    if unit == "us":
        return int(number) // 1_000
    return int(number) // 1_000_000


def _to_ms(value: Any, exchange: Any = None) -> int | None:
    """Convertit une date en epoch millisecondes UTC (``None`` reste ``None``).

    Entrées acceptées :

    - ``int`` / ``float`` / chaîne de chiffres : epoch en secondes, millisecondes,
      microsecondes ou nanosecondes (unité devinée d'après l'ordre de grandeur, comme pour
      les colonnes CSV) ;
    - chaîne ISO 8601 : ``"2024-01-01"``, ``"2024-01-01 12:00"``, ``"2024-01-01T12:00:00Z"``,
      ``"2024-01-01T12:00:00+02:00"`` (naïf = UTC) ; les formats ``JJ/MM/AAAA`` sont
      **refusés** (ambigus) avec un message demandant AAAA-MM-JJ ;
    - ``datetime`` / ``pd.Timestamp`` (naïf = UTC, tz-aware convertis).
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("Une date ne peut pas être un booléen.")
    if isinstance(value, int | np.integer):
        return _epoch_to_ms(int(value))
    if isinstance(value, float | np.floating):
        if not math.isfinite(value):
            raise ValueError(f"Date invalide : {value!r}")
        return _epoch_to_ms(float(value))
    if isinstance(value, str):
        text = value.strip()
        if _DIGITS_RE.match(text):
            return _epoch_to_ms(float(text))
        if not _ISO_DATETIME_RE.match(text):
            raise ValueError(
                f"Date invalide : {value!r} (attendu ISO 8601 : AAAA-MM-JJ ou "
                "AAAA-MM-JJTHH:MM:SSZ, ex. 2024-01-01 ; les formats JJ/MM/AAAA sont refusés "
                "car ambigus ; un epoch en secondes ou millisecondes est aussi accepté)."
            )
        parse8601 = getattr(exchange, "parse8601", None)
        if callable(parse8601):
            try:
                parsed = parse8601(text)
            except Exception:  # noqa: BLE001 - ccxt peut lever n'importe quoi
                parsed = None
            if isinstance(parsed, int | float) and not isinstance(parsed, bool):
                return int(parsed)
        try:
            ts = pd.Timestamp(text)
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


def _bounds_ms(since: Any, until: Any, exchange: Any = None) -> tuple[int | None, int | None]:
    """``(since_ms, until_ms)`` d'une demande ; une chaîne ``until`` sans heure
    (``"2024-12-31"``) couvre **toute la journée** (23:59:59.999)."""
    since_ms = _to_ms(since, exchange)
    until_ms = _to_ms(until, exchange)
    if until_ms is not None and isinstance(until, str) and _DATE_ONLY_RE.match(until):
        until_ms += _DAY_MS - 1
    if since_ms is not None and until_ms is not None and until_ms < since_ms:
        raise ValueError("`until` doit être postérieur à `since`.")
    return since_ms, until_ms


def _ms_to_timestamp(ms: int) -> pd.Timestamp:
    return pd.Timestamp(ms, unit="ms", tz="UTC")


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


def _now_ms(exchange: Any, now_ms: int | None) -> int:
    """« Maintenant » en ms : ``now_ms`` injecté, sinon ``exchange.milliseconds()``, sinon l'horloge."""
    if now_ms is not None:
        return int(now_ms)
    milliseconds = getattr(exchange, "milliseconds", None)
    if callable(milliseconds):
        return int(milliseconds())
    return int(time.time() * 1000)


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

    - ``since`` / ``until`` : ISO 8601 (``"2024-01-01"``, ``"2024-01-01T12:00:00Z"``),
      ``datetime`` / ``Timestamp``, ou epoch (secondes ou millisecondes, unité devinée) ;
      voir :func:`_to_ms`. ``until`` est inclusif et comparé à l'heure d'**ouverture** de la
      bougie ; une chaîne ``until`` sans heure (``"2024-12-31"``) inclut toute la journée.
      Sans ``since``, l'exchange renvoie les bougies les plus récentes.
    - ``limit`` : nombre maximal de bougies brutes renvoyées (``None`` = pas de plafond).
      Avec ``since`` : les ``limit`` premières à partir de ``since`` ; sans ``since`` : les
      ``limit`` plus récentes, y compris la bougie en cours (retirée ensuite si
      ``drop_last_incomplete``) — la pagination remonte aussi loin que nécessaire, même si
      ``limit`` dépasse ``page_size`` ou le plafond par appel de l'exchange.
    - ``exchange`` : objet ccxt déjà construit (ou un faux pour les tests) exposant
      ``fetch_ohlcv(symbol, timeframe, since, limit)``, ``parse8601``, ``milliseconds``,
      ``rateLimit`` et ``timeframes`` ; s'il est ``None``, ``ccxt.<exchange_id>`` est créé.
    - ``drop_last_incomplete`` : retire la dernière bougie si elle n'est pas encore clôturée
      (``ouverture + timeframe > maintenant``) ; ``now_ms`` permet de fixer « maintenant »
      (sinon ``exchange.milliseconds()``, sinon l'horloge locale).
    - ``page_size`` : bougies demandées par appel (1000 sur Binance). Une page plus courte
      n'est **pas** un signe de fin : certains exchanges plafonnent à 300 (coinbase, okx) ou
      200 bougies par appel. La pagination s'arrête sur une page vide, quand ``until`` ou
      ``limit`` est atteint, quand la dernière bougie reçue est la bougie en cours, ou
      faute de progrès.
    - ``sleep`` : fonction d'attente appelée avec ``rateLimit / 1000`` secondes entre deux
      appels (injectable dans les tests).

    Un avertissement est journalisé si l'exchange ne renvoie des bougies qu'à partir d'une
    date nettement postérieure au ``since`` demandé (historique limité, ex. Kraken : 720
    dernières bougies seulement).
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

    since_ms, until_ms = _bounds_ms(since, until, exchange)
    now = _now_ms(exchange, now_ms)

    rate_limit_ms = getattr(exchange, "rateLimit", 0) or 0
    pause = max(float(rate_limit_ms) / 1000.0, 0.0)

    # Sans `since` mais avec `limit` : on veut les `limit` bougies les plus récentes. Si elles
    # ne tiennent pas dans une page, on paginera depuis une fenêtre calculée à partir de
    # « maintenant » (marge de deux bougies) et on gardera les `limit` dernières.
    keep_latest = since_ms is None and limit is not None
    window_ms = now - (limit + 1) * tf_ms if keep_latest else None
    window_mode = False
    if keep_latest and limit > page_size:
        since_ms = window_ms
        window_mode = True

    rows: list[Sequence[Any]] = []
    cursor = since_ms
    n_calls = 0
    while True:
        if window_mode:
            batch_limit = min(page_size, limit + 2)
        elif limit is None:
            batch_limit = page_size
        else:
            batch_limit = min(page_size, limit - len(rows))
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
        if until_ms is not None and last_ts >= until_ms:
            break
        if not window_mode and limit is not None and len(rows) >= limit:
            break
        if cursor is None:
            # sans `since`, la page reçue est la plus récente : rien de plus récent à chercher
            if keep_latest and len(rows) < limit:
                # ... mais plus courte que demandé (plafond par appel de l'exchange, ou
                # historique court) : on repart d'une fenêtre explicite depuis « maintenant ».
                logger.debug(
                    "fetch_ohlcv : %d bougie(s) reçue(s) pour %d demandées sans `since`, "
                    "pagination depuis %s.",
                    len(rows),
                    limit,
                    _ms_to_timestamp(window_ms),
                )
                rows = []
                cursor = window_ms
                window_mode = True
                continue
            break
        if last_ts + tf_ms > now:
            break  # la bougie en cours est arrivée : rien de plus récent à aller chercher
        next_cursor = last_ts + 1
        if next_cursor <= cursor:
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
        df = df.iloc[-limit:] if keep_latest else df.iloc[:limit]

    if since_ms is not None and not window_mode and len(df) > 0:
        first_ms = int((df.index[0] - _EPOCH_MS) // _MS)
        if first_ms - since_ms > 2 * tf_ms:
            logger.warning(
                "fetch_ohlcv : %s n'a renvoyé des bougies %s %s qu'à partir de %s (demandé : "
                "%s) : historique limité sur cet exchange. Essayez un autre exchange (ex. "
                "binance) ou un fichier CSV.",
                exchange_id,
                symbol,
                timeframe,
                df.index[0],
                _ms_to_timestamp(since_ms),
            )

    if drop_last_incomplete and len(df) > 0:
        last_open_ms = int((df.index[-1] - _EPOCH_MS) // _MS)
        if last_open_ms + tf_ms > now:
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


def _slice_period(
    df: pd.DataFrame,
    since_ms: int | None,
    until_ms: int | None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Restreint des bougies à ``[since, until]`` (bornes en ms, inclusives, sur l'ouverture),
    puis à ``limit`` bougies : les premières depuis ``since``, sinon les plus récentes
    (mêmes conventions que :func:`fetch_ohlcv`)."""
    if len(df) == 0:
        return df
    mask = np.ones(len(df), dtype=bool)
    if since_ms is not None:
        mask &= np.asarray(df.index >= _ms_to_timestamp(since_ms), dtype=bool)
    if until_ms is not None:
        mask &= np.asarray(df.index <= _ms_to_timestamp(until_ms), dtype=bool)
    out = df if mask.all() else df.loc[mask]
    if limit is not None and len(out) > limit:
        out = out.iloc[:limit] if since_ms is not None else out.iloc[-limit:]
    return out


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
    """Bougies depuis le cache CSV (``<cache_dir>/<exchange>_<SYMBOLE>_<timeframe>.csv``),
    complété ou téléchargé via :func:`fetch_ohlcv` si nécessaire.

    - Pas de cache, ou ``refresh`` vrai : téléchargement complet puis sauvegarde.
    - Cache présent : il est utilisé **s'il couvre la période demandée**. Si ``since`` est
      antérieur à sa première bougie, ou ``until`` postérieur à sa dernière, seul le
      complément manquant (début et/ou fin) est téléchargé, fusionné et sauvegardé ; un
      ``limit`` supérieur au nombre de bougies en cache déclenche un téléchargement
      complet. Sans ``until``, le cache n'est pas rafraîchi automatiquement (``refresh``
      ou la commande ``download`` pour le mettre à jour). Chaque décision est journalisée.
    - Le résultat est restreint à ``[since, until]`` quand ces bornes sont fournies.

    Les arguments nommés supplémentaires (``until``, ``limit``, ``sleep``, ``now_ms``...)
    sont transmis à :func:`fetch_ohlcv`.
    """
    path = cache_path(cache_dir, exchange_id, symbol, timeframe)
    until = fetch_kwargs.get("until")
    limit = fetch_kwargs.get("limit")
    since_ms, until_ms = _bounds_ms(since, until, exchange)

    if not path.is_file() or refresh:
        df = fetch_ohlcv(
            exchange_id, symbol, timeframe, since=since, exchange=exchange, **fetch_kwargs
        )
        if len(df) > 0:
            save_csv(df, path)
        else:
            logger.warning(
                "Rien à mettre en cache pour %s %s (%s).", symbol, timeframe, exchange_id
            )
        return df

    cached = load_csv(path)
    tf_ms = int(timeframe_to_timedelta(timeframe) // _MS)
    first_ms = int((cached.index[0] - _EPOCH_MS) // _MS)
    last_ms = int((cached.index[-1] - _EPOCH_MS) // _MS)
    needs_head = since_ms is not None and first_ms - since_ms > tf_ms
    needs_tail = until_ms is not None and last_ms + tf_ms <= until_ms
    period = f"{cached.index[0]} → {cached.index[-1]}"

    if limit is not None and len(_slice_period(cached, since_ms, until_ms)) < limit:
        logger.info(
            "Cache %s : %d bougies (%s), moins que les %d demandées : nouveau téléchargement.",
            path,
            len(cached),
            period,
            limit,
        )
        df = fetch_ohlcv(
            exchange_id, symbol, timeframe, since=since, exchange=exchange, **fetch_kwargs
        )
        if len(df) > 0:
            save_csv(df, path)
        return df

    if not needs_head and not needs_tail:
        logger.info(
            "Cache utilisé : %s (%d bougies, %s) ; --refresh pour retélécharger.",
            path,
            len(cached),
            period,
        )
        return _slice_period(cached, since_ms, until_ms, limit)

    wanted = (
        f"{_ms_to_timestamp(since_ms) if since_ms is not None else 'début du cache'} → "
        f"{_ms_to_timestamp(until_ms) if until_ms is not None else 'fin du cache'}"
    )
    logger.info(
        "Cache %s incomplet (en cache : %s ; demandé : %s) : téléchargement du complément.",
        path,
        period,
        wanted,
    )
    extra_kwargs = {k: v for k, v in fetch_kwargs.items() if k not in ("until", "limit")}
    parts: list[pd.DataFrame] = []
    if needs_head:
        head = fetch_ohlcv(
            exchange_id,
            symbol,
            timeframe,
            since=since_ms,
            until=first_ms - 1,
            exchange=exchange,
            **extra_kwargs,
        )
        parts.append(head)
    parts.append(cached)
    if needs_tail:
        tail = fetch_ohlcv(
            exchange_id,
            symbol,
            timeframe,
            since=last_ms + tf_ms,
            until=until_ms,
            exchange=exchange,
            **extra_kwargs,
        )
        parts.append(tail)
    merged = ensure_ohlcv(pd.concat([p for p in parts if len(p) > 0]))
    if len(merged) != len(cached):
        save_csv(merged, path)
        logger.info("Cache %s mis à jour : %d bougies.", path, len(merged))
    else:
        logger.warning(
            "Cache %s inchangé : l'exchange n'a fourni aucune bougie supplémentaire pour la "
            "période demandée (%s).",
            path,
            wanted,
        )
    return _slice_period(merged, since_ms, until_ms, limit)
