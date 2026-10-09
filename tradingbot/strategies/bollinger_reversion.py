"""Stratégie « retour à la moyenne sur bandes de Bollinger » (bollinger_reversion), achat
uniquement.

Idée : les bandes de Bollinger encadrent le prix à ``n_std`` écarts-types autour de sa
moyenne mobile. Une clôture **sous** la bande basse, confirmée par un RSI faible, signale
un excès de baisse : on achète en pariant sur un retour vers la moyenne (bande du milieu),
où l'on revend.

Règles (position souhaitée pour la bougie suivante) :

- **Entrée** (si plat) : ``close < lower`` **et** ``rsi < rsi_max`` → ``1``.
- **Sortie** (si en position) : ``close >= middle`` → ``0``. La bougie de sortie est
  toujours plate ; une nouvelle entrée n'est réévaluée qu'à la bougie suivante.
- Jamais de ``-1`` (achat uniquement) ; ``0`` tant que les indicateurs ne sont pas
  définis (période de chauffe).

Stop : sur la bougie d'entrée uniquement, ``stop_loss = close - atr_mult * ATR`` ;
``NaN`` ailleurs. Aucun look-ahead : les bandes utilisent une moyenne et un écart-type
glissants (jamais centrés, jamais calculés sur toute la série).
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from tradingbot import indicators as ind
from tradingbot.strategies import register
from tradingbot.strategies.base import Strategy

logger = logging.getLogger(__name__)

#: bougies de marge ajoutées à la plus longue fenêtre pour la période de chauffe
WARMUP_MARGIN = 5


def _as_int(value: Any, name: str, minimum: int = 1) -> int:
    """Convertit un paramètre en entier (accepte ``20`` ou ``20.0``) et vérifie ``>= minimum``."""
    if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        raise ValueError(f"Le paramètre `{name}` doit être un entier, reçu {value!r}")
    if not np.isfinite(float(value)) or float(value) != float(int(value)):
        raise ValueError(f"Le paramètre `{name}` doit être un entier, reçu {value!r}")
    result = int(value)
    if result < minimum:
        raise ValueError(f"Le paramètre `{name}` doit être >= {minimum}, reçu {result}")
    return result


def _as_float(value: Any, name: str, low: float, high: float) -> float:
    """Convertit un paramètre en flottant fini dans l'intervalle ouvert ``(low, high)``."""
    if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        raise ValueError(f"Le paramètre `{name}` doit être un nombre, reçu {value!r}")
    result = float(value)
    if not np.isfinite(result) or not (low < result < high):
        raise ValueError(
            f"Le paramètre `{name}` doit être compris entre {low:g} et {high:g} (exclus), "
            f"reçu {value!r}"
        )
    return result


@register
class BollingerReversion(Strategy):
    """Achat sous la bande de Bollinger basse (RSI faible), revente au retour à la moyenne."""

    name: ClassVar[str] = "bollinger_reversion"
    description: ClassVar[str] = (
        "Achète quand le prix sort sous la bande de Bollinger basse avec un RSI faible, et "
        "revend au retour vers la moyenne."
    )
    default_params: ClassVar[dict[str, Any]] = {
        "window": 20,
        "n_std": 2.0,
        "rsi_window": 14,
        "rsi_max": 35.0,
        "atr_window": 14,
        "atr_mult": 2.0,
    }
    param_grid: ClassVar[dict[str, list[Any]]] = {
        "window": [20, 30],
        "n_std": [1.5, 2.0, 2.5],
        "rsi_max": [30.0, 35.0, 40.0],
    }

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        p = self.params
        p["window"] = _as_int(p["window"], "window", minimum=2)
        p["n_std"] = _as_float(p["n_std"], "n_std", 0.0, float("inf"))
        p["rsi_window"] = _as_int(p["rsi_window"], "rsi_window")
        p["rsi_max"] = _as_float(p["rsi_max"], "rsi_max", 0.0, 100.0)
        p["atr_window"] = _as_int(p["atr_window"], "atr_window")
        p["atr_mult"] = _as_float(p["atr_mult"], "atr_mult", 0.0, float("inf"))

    def warmup_bars(self) -> int:
        """Plus longue fenêtre (bandes, RSI ou ATR) plus une marge de sécurité."""
        p = self.params
        return max(p["window"], p["rsi_window"] + 1, p["atr_window"] + 1) + WARMUP_MARGIN

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Signaux de retour à la moyenne sur Bollinger ; voir la docstring du module."""
        p = self.params
        rsi_max = p["rsi_max"]
        mult = p["atr_mult"]

        close_s = df["close"].astype("float64")
        bands = ind.bollinger(close_s, p["window"], p["n_std"])
        rsi_s = ind.rsi(close_s, p["rsi_window"])
        atr_s = ind.atr(df, p["atr_window"])

        valid = (
            bands["lower"].notna() & bands["middle"].notna() & rsi_s.notna() & atr_s.notna()
        ).to_numpy(copy=True)
        valid[: self.warmup_bars()] = False

        n = len(df)
        close = close_s.to_numpy().tolist()
        lower = bands["lower"].to_numpy().tolist()
        middle = bands["middle"].to_numpy().tolist()
        rsi = rsi_s.to_numpy().tolist()
        atr = atr_s.to_numpy().tolist()

        signal = np.zeros(n, dtype=np.int64)
        stop = np.full(n, np.nan)

        pos = 0
        n_entries = 0
        for t in range(n):
            if not valid[t]:
                pos = 0
                continue
            c = close[t]
            if pos == 1:
                if c >= middle[t]:
                    pos = 0  # bougie de sortie : plate, pas de ré-entrée immédiate
                    continue
                signal[t] = 1
                continue
            if c < lower[t] and rsi[t] < rsi_max:
                pos = 1
                signal[t] = 1
                stop[t] = c - mult * atr[t]
                n_entries += 1

        logger.debug(
            "bollinger_reversion(%d, %g sigma, rsi<%g) : %d bougies, %d entrées",
            p["window"],
            p["n_std"],
            rsi_max,
            n,
            n_entries,
        )

        signals = self.empty_signals(df)
        signals["signal"] = signal
        signals["stop_loss"] = stop
        return Strategy.validate_signals(df, signals)
