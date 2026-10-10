"""Stratégie « croisement de moyennes mobiles simples » (SMA cross).

Idée : une moyenne mobile courte (``fast``) qui passe au-dessus d'une moyenne longue
(``slow``) indique qu'une tendance haussière s'installe : on achète. Quand elle repasse
en dessous, la tendance s'essouffle : on vend et, si le moteur autorise la vente à
découvert, on se place à la baisse (sinon le ``-1`` est simplement traité comme « plat »).

Signal (position souhaitée pour la bougie suivante) :

- ``1``  tant que ``sma_fast > sma_slow`` ;
- ``-1`` tant que ``sma_fast < sma_slow`` ;
- en cas d'égalité stricte, on conserve la position précédente (pas de va-et-vient) ;
- ``0`` tant que les indicateurs ne sont pas définis (période de chauffe).

Stop : sur la bougie d'entrée (passage de 0 à ±1, ou retournement ±1 → ∓1),
``stop_loss`` vaut ``close - atr_mult * ATR`` pour un achat et ``close + atr_mult * ATR``
pour une vente à découvert ; ce stop d'entrée est ensuite **porté tel quel sur chaque
bougie** tant que la position est tenue (``signal[t] == signal[t-1] != 0``), afin qu'un
moteur ou un bot qui entre (ou ré-entre) en cours de position dispose toujours du stop ;
``NaN`` quand le signal est 0. Aucun look-ahead : la bougie ``t`` n'utilise que les
données ``<= t``.
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
class SmaCross(Strategy):
    """Suivi de tendance par croisement de deux moyennes mobiles simples."""

    name: ClassVar[str] = "sma_cross"
    description: ClassVar[str] = (
        "Suit la tendance : achète quand la moyenne mobile courte passe au-dessus de la "
        "longue, vend (ou joue la baisse) quand elle repasse en dessous."
    )
    default_params: ClassVar[dict[str, Any]] = {
        "fast": 20,
        "slow": 50,
        "atr_window": 14,
        "atr_mult": 3.0,
    }
    param_grid: ClassVar[dict[str, list[Any]]] = {
        "fast": [10, 20, 30],
        "slow": [50, 100],
        "atr_mult": [2.0, 3.0, 4.0],
    }

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        p = self.params
        p["fast"] = _as_int(p["fast"], "fast")
        p["slow"] = _as_int(p["slow"], "slow", minimum=2)
        p["atr_window"] = _as_int(p["atr_window"], "atr_window")
        p["atr_mult"] = _as_positive_float(p["atr_mult"], "atr_mult")
        if p["fast"] >= p["slow"]:
            raise ValueError(
                f"`fast` ({p['fast']}) doit être strictement inférieur à `slow` ({p['slow']})"
            )

    def warmup_bars(self) -> int:
        """Plus longue fenêtre (SMA lente ou ATR) plus une marge de sécurité."""
        p = self.params
        return max(p["slow"], p["atr_window"] + 1) + WARMUP_MARGIN

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Signaux de croisement SMA ; voir la docstring du module."""
        p = self.params
        close = df["close"].astype("float64")
        sma_fast = ind.sma(close, p["fast"])
        sma_slow = ind.sma(close, p["slow"])
        atr = ind.atr(df, p["atr_window"])

        valid = (sma_fast.notna() & sma_slow.notna() & atr.notna()).to_numpy(copy=True)
        valid[: self.warmup_bars()] = False

        fast_v = sma_fast.to_numpy()
        slow_v = sma_slow.to_numpy()
        # +1 / -1 selon le côté du croisement ; NaN en cas d'égalité ou d'indicateur
        # indéfini, puis report de la dernière valeur connue (on garde la position).
        raw = np.where(fast_v > slow_v, 1.0, np.where(fast_v < slow_v, -1.0, np.nan))
        raw[~valid] = np.nan
        held = pd.Series(raw, index=df.index).ffill().fillna(0.0).to_numpy(copy=True)
        held[~valid] = 0.0
        signal = held.astype(np.int64)

        prev = np.concatenate(([0], signal[:-1]))
        entry = (signal != 0) & (signal != prev)
        stop_at_entry = np.where(
            entry,
            close.to_numpy() - signal * p["atr_mult"] * atr.to_numpy(),
            np.nan,
        )
        # Le stop d'entrée est porté (constant) sur toute la position : chaque segment de
        # signal non nul commence par une entrée, le report avant ne traverse donc jamais
        # une position précédente ; NaN dès que le signal est 0.
        stop = pd.Series(stop_at_entry, index=df.index).ffill().to_numpy(copy=True)
        stop[signal == 0] = np.nan
        logger.debug(
            "sma_cross(%d/%d) : %d bougies, %d entrées", p["fast"], p["slow"], len(df), entry.sum()
        )

        signals = self.empty_signals(df)
        signals["signal"] = signal
        signals["stop_loss"] = stop
        return Strategy.validate_signals(df, signals)
