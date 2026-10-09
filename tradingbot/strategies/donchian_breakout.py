"""Stratégie « cassure de canal de Donchian » (donchian_breakout), façon « Turtles ».

Idée : quand le prix clôture au-dessus du plus haut des ``entry_window`` bougies
précédentes, quelque chose de nouveau se passe : on achète la cassure et on laisse courir.
On sort quand le prix clôture sous le plus bas des ``exit_window`` bougies précédentes
(fenêtre plus courte : on sort plus vite qu'on n'entre). Symétrique pour la baisse.

Règles (position souhaitée pour la bougie suivante) :

- **Entrée longue** (si plat) : ``close > upper(entry_window)`` → ``1``.
- **Entrée courte** (si plat) : ``close < lower(entry_window)`` → ``-1`` (traité comme
  ``0`` par le moteur si la vente à découvert est désactivée).
- **Sortie d'un long** : ``close < lower(exit_window)`` → ``0`` ; si, sur la même bougie,
  ``close < lower(entry_window)``, on se retourne directement en ``-1``. Symétrique pour
  un short (``close > upper(exit_window)``, retournement si ``close > upper(entry_window)``).
- ``0`` tant que les indicateurs ne sont pas définis (période de chauffe).

Les canaux sont calculés sur les bougies **précédentes** (``shift(1)`` dans
``tradingbot.indicators.donchian``) : la bougie courante ne peut pas déclencher sa propre
cassure. Stop : sur la bougie d'entrée uniquement, ``stop_loss = close - atr_mult * ATR``
(``close + …`` pour un short) ; ``NaN`` ailleurs. Aucun look-ahead.
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


def _as_positive_float(value: Any, name: str) -> float:
    """Convertit un paramètre en flottant strictement positif."""
    if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        raise ValueError(f"Le paramètre `{name}` doit être un nombre, reçu {value!r}")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"Le paramètre `{name}` doit être > 0, reçu {value!r}")
    return result


@register
class DonchianBreakout(Strategy):
    """Cassure du canal de Donchian : entrée sur nouveau plus haut, sortie sur plus bas récent."""

    name: ClassVar[str] = "donchian_breakout"
    description: ClassVar[str] = (
        "Achète la cassure du plus haut des N dernières bougies et sort quand le prix casse "
        "le plus bas des M dernières (méthode des « Turtles »)."
    )
    default_params: ClassVar[dict[str, Any]] = {
        "entry_window": 20,
        "exit_window": 10,
        "atr_window": 14,
        "atr_mult": 2.0,
    }
    param_grid: ClassVar[dict[str, list[Any]]] = {
        "entry_window": [20, 40, 55],
        "exit_window": [10, 20],
        "atr_mult": [2.0, 3.0],
    }

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        p = self.params
        p["entry_window"] = _as_int(p["entry_window"], "entry_window")
        p["exit_window"] = _as_int(p["exit_window"], "exit_window")
        p["atr_window"] = _as_int(p["atr_window"], "atr_window")
        p["atr_mult"] = _as_positive_float(p["atr_mult"], "atr_mult")

    def warmup_bars(self) -> int:
        """Plus longue fenêtre (canaux ou ATR) plus une marge de sécurité."""
        p = self.params
        return max(p["entry_window"], p["exit_window"], p["atr_window"]) + WARMUP_MARGIN

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Signaux de cassure de canal ; voir la docstring du module."""
        p = self.params
        mult = p["atr_mult"]

        close_s = df["close"].astype("float64")
        entry_ch = ind.donchian(df, p["entry_window"])
        exit_ch = ind.donchian(df, p["exit_window"])
        atr_s = ind.atr(df, p["atr_window"])

        valid = (
            entry_ch["upper"].notna()
            & entry_ch["lower"].notna()
            & exit_ch["upper"].notna()
            & exit_ch["lower"].notna()
            & atr_s.notna()
        ).to_numpy(copy=True)
        valid[: self.warmup_bars()] = False

        n = len(df)
        close = close_s.to_numpy().tolist()
        up_in = entry_ch["upper"].to_numpy().tolist()
        lo_in = entry_ch["lower"].to_numpy().tolist()
        up_out = exit_ch["upper"].to_numpy().tolist()
        lo_out = exit_ch["lower"].to_numpy().tolist()
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
            if pos == 1 and c < lo_out[t]:
                pos = 0
            elif pos == -1 and c > up_out[t]:
                pos = 0
            if pos == 0:
                if c > up_in[t]:
                    pos = 1
                    stop[t] = c - mult * atr[t]
                    n_entries += 1
                elif c < lo_in[t]:
                    pos = -1
                    stop[t] = c + mult * atr[t]
                    n_entries += 1
            signal[t] = pos

        logger.debug(
            "donchian_breakout(%d/%d) : %d bougies, %d entrées",
            p["entry_window"],
            p["exit_window"],
            n,
            n_entries,
        )

        signals = self.empty_signals(df)
        signals["signal"] = signal
        signals["stop_loss"] = stop
        return Strategy.validate_signals(df, signals)
