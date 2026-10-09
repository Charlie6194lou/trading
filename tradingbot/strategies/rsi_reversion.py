"""Stratégie « retour à la moyenne sur RSI » (rsi_reversion), achat uniquement.

Idée : dans un marché globalement haussier (clôture au-dessus d'une longue moyenne
mobile ``trend_window``), un RSI très bas signale un repli exagéré : on achète le creux
en pariant sur un rebond rapide. On revend dès que le RSI se normalise, ou après un
nombre maximal de bougies si le rebond ne vient pas (on ne s'entête pas).

Règles (position souhaitée pour la bougie suivante) :

- **Entrée** (si plat) : ``rsi < oversold`` **et** ``close > sma(trend_window)`` → ``1``.
- **Sortie** (si en position) : ``rsi > exit_level`` **ou** position tenue depuis
  ``max_hold`` bougies → ``0``. La bougie de sortie est toujours plate : une nouvelle
  entrée n'est réévaluée qu'à la bougie suivante.
- Jamais de ``-1`` (achat uniquement) ; ``0`` tant que les indicateurs ne sont pas
  définis (période de chauffe).

Stop : sur la bougie d'entrée uniquement, ``stop_loss = close - atr_mult * ATR`` ;
``NaN`` ailleurs. La logique est une petite machine à états parcourue dans l'ordre du
temps : la bougie ``t`` ne dépend que des données ``<= t`` (aucun look-ahead).
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
class RsiReversion(Strategy):
    """Achat des replis (RSI survendu) dans une tendance de fond haussière."""

    name: ClassVar[str] = "rsi_reversion"
    description: ClassVar[str] = (
        "Achète les replis (RSI survendu) dans un marché globalement haussier et revend dès "
        "que le prix se reprend ou après un délai maximal."
    )
    default_params: ClassVar[dict[str, Any]] = {
        "rsi_window": 14,
        "oversold": 30.0,
        "exit_level": 55.0,
        "trend_window": 200,
        "atr_window": 14,
        "atr_mult": 2.0,
        "max_hold": 20,
    }
    param_grid: ClassVar[dict[str, list[Any]]] = {
        "rsi_window": [7, 14],
        "oversold": [25.0, 30.0, 35.0],
        "exit_level": [50.0, 60.0],
        "max_hold": [10, 20],
    }

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        p = self.params
        p["rsi_window"] = _as_int(p["rsi_window"], "rsi_window")
        p["oversold"] = _as_float(p["oversold"], "oversold", 0.0, 100.0)
        p["exit_level"] = _as_float(p["exit_level"], "exit_level", 0.0, 100.0)
        p["trend_window"] = _as_int(p["trend_window"], "trend_window")
        p["atr_window"] = _as_int(p["atr_window"], "atr_window")
        p["atr_mult"] = _as_float(p["atr_mult"], "atr_mult", 0.0, float("inf"))
        p["max_hold"] = _as_int(p["max_hold"], "max_hold")
        if p["oversold"] >= p["exit_level"]:
            raise ValueError(
                f"`oversold` ({p['oversold']:g}) doit être strictement inférieur à "
                f"`exit_level` ({p['exit_level']:g})"
            )

    def warmup_bars(self) -> int:
        """Plus longue fenêtre (tendance, RSI ou ATR) plus une marge de sécurité."""
        p = self.params
        longest = max(p["trend_window"], p["rsi_window"] + 1, p["atr_window"] + 1)
        return longest + WARMUP_MARGIN

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Signaux de retour à la moyenne sur RSI ; voir la docstring du module."""
        p = self.params
        oversold = p["oversold"]
        exit_level = p["exit_level"]
        max_hold = p["max_hold"]
        mult = p["atr_mult"]

        close_s = df["close"].astype("float64")
        rsi_s = ind.rsi(close_s, p["rsi_window"])
        trend_s = ind.sma(close_s, p["trend_window"])
        atr_s = ind.atr(df, p["atr_window"])

        valid = (rsi_s.notna() & trend_s.notna() & atr_s.notna()).to_numpy(copy=True)
        valid[: self.warmup_bars()] = False

        n = len(df)
        rsi = rsi_s.to_numpy().tolist()
        trend = trend_s.to_numpy().tolist()
        atr = atr_s.to_numpy().tolist()
        close = close_s.to_numpy().tolist()

        signal = np.zeros(n, dtype=np.int64)
        stop = np.full(n, np.nan)

        pos = 0
        entry_t = -1
        n_entries = 0
        for t in range(n):
            if not valid[t]:
                pos = 0
                continue
            if pos == 1:
                if rsi[t] > exit_level or (t - entry_t) >= max_hold:
                    pos = 0  # bougie de sortie : plate, pas de ré-entrée immédiate
                    continue
                signal[t] = 1
                continue
            if rsi[t] < oversold and close[t] > trend[t]:
                pos = 1
                entry_t = t
                signal[t] = 1
                stop[t] = close[t] - mult * atr[t]
                n_entries += 1

        logger.debug(
            "rsi_reversion(rsi<%g, sortie>%g, max %d bougies) : %d bougies, %d entrées",
            oversold,
            exit_level,
            max_hold,
            n,
            n_entries,
        )

        signals = self.empty_signals(df)
        signals["signal"] = signal
        signals["stop_loss"] = stop
        return Strategy.validate_signals(df, signals)
