"""Génération de bougies OHLCV synthétiques, sans réseau.

Utilisé par la commande ``demo`` et par les tests : une marche aléatoire géométrique
découpée en « régimes » (haussier, baissier, latéral) enchaînés, avec des bougies
cohérentes (``low <= min(open, close)``, ``high >= max(open, close)``, ``volume > 0``)
et reproductibles grâce à ``seed``.

Ces données ne ressemblent à aucun marché réel : elles servent à vérifier que la
mécanique du robot fonctionne, pas à juger de la rentabilité d'une stratégie.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["default_regimes", "generate_ohlcv"]

#: (n_bars, dérive par bougie, volatilité par bougie)
Regime = tuple[int, float, float]

_TIMEFRAME_RE = re.compile(r"^\s*(\d+)\s*([mhdw])\s*$")
# unité ccxt -> (alias d'offset pandas, multiplicateur) ; les alias sont ceux que
# ``pd.infer_freq`` redonne, pour qu'un aller-retour CSV conserve ``index.freq``.
_OFFSET_ALIASES = {"m": ("min", 1), "h": ("h", 1), "d": ("D", 1), "w": ("D", 7)}
# résolution par défaut des DatetimeIndex ("ns" pandas 2.x, "us" pandas 3)
_DEFAULT_UNIT: str = pd.DatetimeIndex(["2000-01-01"]).unit


def default_regimes(n: int, vol: float) -> list[Regime]:
    """Mélange par défaut : un tiers haussier, un tiers baissier (plus volatil), le reste
    latéral (plus calme). Les longueurs somment exactement à ``n``."""
    third = n // 3
    return [
        (third, 0.0004, vol),
        (third, -0.0003, vol * 1.5),
        (n - 2 * third, 0.0, vol * 0.7),
    ]


def _freq_to_offset(freq: str | pd.Timedelta) -> str | pd.Timedelta:
    """``"1h"`` / ``"15m"`` / ``"1d"`` -> alias pandas (``"h"``, ``"15min"``, ``"D"``) ;
    un ``pd.Timedelta`` ou un alias pandas inconnu du format ccxt est renvoyé tel quel."""
    if isinstance(freq, pd.Timedelta):
        return freq
    match = _TIMEFRAME_RE.match(freq)
    if match is None:
        return freq
    count = int(match.group(1))
    if count <= 0:
        raise ValueError(f"Fréquence invalide : {freq!r}")
    alias, multiplier = _OFFSET_ALIASES[match.group(2)]
    return f"{count * multiplier}{alias}"


def _validate_regimes(regimes: Sequence[Regime], n: int) -> list[Regime]:
    """Vérifie la forme des régimes et que leurs longueurs somment à ``n``."""
    cleaned: list[Regime] = []
    for i, regime in enumerate(regimes):
        if len(regime) != 3:
            raise ValueError(
                f"Régime n°{i} invalide : attendu (n_bars, drift, vol), reçu {regime!r}."
            )
        bars, drift, sigma = regime
        if isinstance(bars, bool) or not isinstance(bars, int | np.integer) or bars < 0:
            raise ValueError(f"Régime n°{i} : n_bars doit être un entier >= 0, reçu {bars!r}.")
        if not np.isfinite(drift) or not np.isfinite(sigma) or sigma < 0:
            raise ValueError(
                f"Régime n°{i} : drift fini et vol >= 0 attendus, reçus ({drift!r}, {sigma!r})."
            )
        cleaned.append((int(bars), float(drift), float(sigma)))
    total = sum(r[0] for r in cleaned)
    if total != n:
        raise ValueError(f"La somme des longueurs des régimes ({total}) doit être égale à n ({n}).")
    return cleaned


def generate_ohlcv(
    n: int = 2000,
    seed: int | None = 0,
    start: str | datetime | pd.Timestamp = "2022-01-01",
    freq: str | pd.Timedelta = "1h",
    start_price: float = 100.0,
    regimes: Sequence[Regime] | None = None,
    vol: float = 0.01,
) -> pd.DataFrame:
    """Génère ``n`` bougies OHLCV synthétiques au format canonique du robot.

    - ``regimes`` : liste de ``(n_bars, drift_par_bar, vol_par_bar)`` enchaînés dont les
      longueurs doivent sommer à ``n`` ; par défaut :func:`default_regimes` (haussier /
      baissier / latéral) construit à partir de ``vol``.
    - ``seed`` : graine du générateur (``None`` = aléatoire, sinon reproductible).
    - ``start`` / ``freq`` : premier horodatage (UTC si naïf) et pas (``"1h"``, ``"1d"``,
      ``"15m"`` ou alias pandas).

    Garanties : ``open[0] == start_price`` exactement, ``open[t] == close[t-1]`` (pas de
    gap, comme en crypto 24/7), ``low <= min(open, close) <= max(open, close) <= high``,
    tous les prix ``> 0`` et ``volume > 0``.
    """
    if isinstance(n, bool) or not isinstance(n, int | np.integer) or n < 1:
        raise ValueError(f"`n` doit être un entier >= 1, reçu {n!r}.")
    n = int(n)
    if not np.isfinite(start_price) or start_price <= 0:
        raise ValueError(f"`start_price` doit être > 0, reçu {start_price!r}.")
    if not np.isfinite(vol) or vol < 0:
        raise ValueError(f"`vol` doit être >= 0, reçu {vol!r}.")

    regime_list = _validate_regimes(regimes if regimes is not None else default_regimes(n, vol), n)
    rng = np.random.default_rng(seed)

    # rendements log bougie par bougie, régime après régime
    log_returns = np.concatenate(
        [rng.normal(drift, sigma, size=bars) for bars, drift, sigma in regime_list] or [np.zeros(0)]
    )
    bar_vol = np.concatenate(
        [np.full(bars, sigma, dtype="float64") for bars, _drift, sigma in regime_list]
        or [np.zeros(0)]
    )

    close = float(start_price) * np.exp(np.cumsum(log_returns))
    open_ = np.concatenate([[float(start_price)], close[:-1]])
    body_high = np.maximum(open_, close)
    body_low = np.minimum(open_, close)

    # mèches multiplicatives (toujours positives) : high >= max(o, c), low <= min(o, c) > 0
    upper_wick = np.abs(rng.normal(0.0, 0.5, size=n)) * bar_vol
    lower_wick = np.abs(rng.normal(0.0, 0.5, size=n)) * bar_vol
    high = body_high * np.exp(upper_wick)
    low = body_low * np.exp(-lower_wick)

    # volume > 0, lognormal, plus élevé quand la bougie bouge beaucoup
    scale = np.where(bar_vol > 0, np.abs(log_returns) / np.where(bar_vol > 0, bar_vol, 1.0), 0.0)
    volume = rng.lognormal(mean=np.log(1000.0), sigma=0.4, size=n) * (1.0 + scale)

    start_ts = pd.Timestamp(start)
    start_ts = start_ts.tz_localize("UTC") if start_ts.tz is None else start_ts.tz_convert("UTC")
    # même résolution que pd.date_range(<chaîne>) dans la version de pandas installée
    index = pd.date_range(start_ts, periods=n, freq=_freq_to_offset(freq)).as_unit(_DEFAULT_UNIT)
    index.name = "timestamp"

    df = pd.DataFrame(
        {
            "open": open_.astype("float64"),
            "high": high.astype("float64"),
            "low": low.astype("float64"),
            "close": close.astype("float64"),
            "volume": volume.astype("float64"),
        },
        index=index,
    )
    logger.debug(
        "generate_ohlcv : %d bougies, %d régime(s), prix %.4f -> %.4f",
        n,
        len(regime_list),
        float(df["open"].iloc[0]),
        float(df["close"].iloc[-1]),
    )
    return df
