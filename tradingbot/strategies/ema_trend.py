"""Stratégie « tendance EMA + filtre ADX + stop suiveur chandelier » (ema_trend).

Idée : on suit la tendance donnée par deux moyennes mobiles exponentielles (``fast`` et
``slow``), mais on n'entre que si l'ADX confirme que la tendance est assez forte
(``adx > adx_threshold``) : cela évite d'acheter chaque petit croisement dans un marché
qui fait du sur-place. Une fois en position, un stop suiveur « chandelier » monte avec
le prix pour protéger les gains.

Règles (position souhaitée pour la bougie suivante) :

- **Entrée longue** : ``ema_fast > ema_slow`` **et** ``adx > adx_threshold`` → ``1``.
- **Entrée courte** (symétrique) : ``ema_fast < ema_slow`` et ``adx > adx_threshold`` → ``-1``.
  (Le moteur transforme ``-1`` en ``0`` si la vente à découvert est désactivée.)
- **Sortie** dès que l'une de ces conditions est vraie :
  1. les EMA se recroisent (``ema_fast < ema_slow`` pour un long) : on sort, et si l'ADX
     est au-dessus du seuil on se retourne directement (``-1``) sur la même bougie ;
  2. le **plus bas** de la bougie touche le stop suiveur (``low <= stop`` ; ``high >=
     stop`` pour un short) : c'est exactement la condition à laquelle le moteur de
     backtest et le bot live exécutent le stop exporté dans ``stop_loss``. On passe à
     ``0`` pour la bougie suivante ; une nouvelle entrée n'est réévaluée qu'à la bougie
     d'après (avec un nouveau stop). La stratégie et le moteur voient donc la même sortie :
     plus de ré-entrée immédiate avec un stop déjà touché.
- **L'ADX ne sert qu'à l'entrée** : une position ouverte est conservée tant que les EMA
  restent dans le bon ordre, même si l'ADX redescend sous le seuil (une tendance qui
  ralentit n'est pas une tendance qui s'inverse).
- ``0`` tant que les indicateurs ne sont pas définis (période de chauffe).

Stop : à l'entrée, ``stop = close - atr_mult * ATR`` (``close + …`` pour un short) ; puis,
à chaque bougie en position, ``stop_loss`` vaut le **maximum** depuis l'entrée de
``close - atr_mult * ATR`` (stop « chandelier », qui ne redescend jamais au sein d'une
même position ; minimum pour un short). Le moteur met à jour son stop actif à chaque
bougie où ``stop_loss`` n'est pas ``NaN``. Aucun look-ahead.
"""

from __future__ import annotations

import logging
import math
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


def _as_float(value: Any, name: str, minimum: float, strict: bool) -> float:
    """Convertit un paramètre en flottant fini ``> minimum`` (ou ``>=`` si ``strict`` est faux)."""
    if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        raise ValueError(f"Le paramètre `{name}` doit être un nombre, reçu {value!r}")
    result = float(value)
    if not np.isfinite(result) or result < minimum or (strict and result == minimum):
        op = ">" if strict else ">="
        raise ValueError(f"Le paramètre `{name}` doit être {op} {minimum:g}, reçu {value!r}")
    return result


@register
class EmaTrend(Strategy):
    """Suivi de tendance EMA filtré par l'ADX, avec stop suiveur chandelier."""

    name: ClassVar[str] = "ema_trend"
    description: ClassVar[str] = (
        "Suit la tendance (EMA) seulement quand elle est forte (ADX) et protège les gains "
        "avec un stop qui suit le prix."
    )
    default_params: ClassVar[dict[str, Any]] = {
        "fast": 20,
        "slow": 50,
        "adx_window": 14,
        "adx_threshold": 20.0,
        "atr_window": 14,
        "atr_mult": 3.0,
    }
    param_grid: ClassVar[dict[str, list[Any]]] = {
        "fast": [10, 20],
        "slow": [50, 100],
        "adx_threshold": [15.0, 20.0, 25.0],
        "atr_mult": [2.0, 3.0],
    }

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        p = self.params
        p["fast"] = _as_int(p["fast"], "fast")
        p["slow"] = _as_int(p["slow"], "slow", minimum=2)
        p["adx_window"] = _as_int(p["adx_window"], "adx_window")
        p["adx_threshold"] = _as_float(p["adx_threshold"], "adx_threshold", 0.0, strict=False)
        p["atr_window"] = _as_int(p["atr_window"], "atr_window")
        p["atr_mult"] = _as_float(p["atr_mult"], "atr_mult", 0.0, strict=True)
        if p["fast"] >= p["slow"]:
            raise ValueError(
                f"`fast` ({p['fast']}) doit être strictement inférieur à `slow` ({p['slow']})"
            )
        if p["adx_threshold"] >= 100.0:
            raise ValueError(
                f"`adx_threshold` doit être < 100 (l'ADX est borné à 100), reçu {p['adx_threshold']}"
            )

    def warmup_bars(self) -> int:
        """Deux fois la fenêtre la plus longue (les EMA et l'ADX convergent lentement) + marge."""
        p = self.params
        return max(2 * p["slow"], 2 * p["adx_window"], p["atr_window"] + 1) + WARMUP_MARGIN

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Signaux EMA/ADX avec stop chandelier ; voir la docstring du module."""
        p = self.params
        mult = p["atr_mult"]
        threshold = p["adx_threshold"]

        close_s = df["close"].astype("float64")
        ema_fast = ind.ema(close_s, p["fast"])
        ema_slow = ind.ema(close_s, p["slow"])
        adx_s = ind.adx(df, p["adx_window"])["adx"]
        atr_s = ind.atr(df, p["atr_window"])

        valid = (ema_fast.notna() & ema_slow.notna() & adx_s.notna() & atr_s.notna()).to_numpy(
            copy=True
        )
        valid[: self.warmup_bars()] = False

        n = len(df)
        fast = ema_fast.to_numpy().tolist()
        slow = ema_slow.to_numpy().tolist()
        adx = adx_s.to_numpy().tolist()
        atr = atr_s.to_numpy().tolist()
        close = close_s.to_numpy().tolist()
        low = df["low"].astype("float64").to_numpy().tolist()
        high = df["high"].astype("float64").to_numpy().tolist()

        signal = np.zeros(n, dtype=np.int64)
        stop = np.full(n, np.nan)

        pos = 0
        trail = math.nan
        n_entries = 0
        for t in range(n):
            if not valid[t]:
                pos = 0
                trail = math.nan
                continue
            c = close[t]
            trend = 1 if fast[t] > slow[t] else (-1 if fast[t] < slow[t] else 0)

            if pos != 0:
                if trend == -pos:
                    # Les EMA se sont recroisées : sortie (retournement évalué ci-dessous).
                    pos = 0
                    trail = math.nan
                elif (pos == 1 and low[t] <= trail) or (pos == -1 and high[t] >= trail):
                    # Le stop chandelier a été touché intrabar (même condition que le
                    # moteur) : plat pour la bougie suivante, nouveau stop à la ré-entrée.
                    pos = 0
                    trail = math.nan
                    continue
                else:
                    cand = c - pos * mult * atr[t]
                    trail = max(trail, cand) if pos == 1 else min(trail, cand)
                    signal[t] = pos
                    stop[t] = trail
                    continue

            # Ici la position est plate (ou vient d'être fermée par un recroisement).
            if trend != 0 and adx[t] > threshold:
                pos = trend
                trail = c - pos * mult * atr[t]
                signal[t] = pos
                stop[t] = trail
                n_entries += 1

        logger.debug(
            "ema_trend(%d/%d, adx>%g) : %d bougies, %d entrées",
            p["fast"],
            p["slow"],
            threshold,
            n,
            n_entries,
        )

        signals = self.empty_signals(df)
        signals["signal"] = signal
        signals["stop_loss"] = stop
        return Strategy.validate_signals(df, signals)
