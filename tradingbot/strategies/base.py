"""Contrat de base d'une stratégie.

Une stratégie reçoit un DataFrame de bougies OHLCV et renvoie, pour chaque bougie,
la position *souhaitée* à partir de la bougie suivante :

- ``signal``      : int dans {-1, 0, 1} (short, flat, long). Calculé uniquement avec
                    les données jusqu'à la bougie courante incluse (pas de look-ahead).
- ``stop_loss``   : float ou NaN. Prix de stop à appliquer à la position. Si une valeur
                    est donnée alors qu'une position est déjà ouverte, le moteur met à jour
                    le stop actif (permet un stop suiveur contrôlé par la stratégie).
- ``take_profit`` : float ou NaN. Même logique pour l'objectif de gain.

Le moteur de backtest / le bot live exécutent le changement de position à l'OUVERTURE
de la bougie suivante.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np
import pandas as pd

SIGNAL_COLUMNS = ("signal", "stop_loss", "take_profit")


class Strategy(ABC):
    """Classe mère de toutes les stratégies.

    Sous-classes : définir ``name``, ``description``, ``default_params`` et
    ``generate_signals``. Enregistrer avec le décorateur ``@register`` de
    :mod:`tradingbot.strategies`.
    """

    #: identifiant court utilisé en CLI (ex. "sma_cross")
    name: ClassVar[str] = "base"
    #: description en une phrase, en français, pour un débutant
    description: ClassVar[str] = ""
    #: paramètres par défaut
    default_params: ClassVar[dict[str, Any]] = {}
    #: grille de recherche pour l'optimisation / le walk-forward
    param_grid: ClassVar[dict[str, list[Any]]] = {}

    def __init__(self, **params: Any) -> None:
        unknown = set(params) - set(self.default_params)
        if unknown:
            raise ValueError(
                f"Paramètres inconnus pour la stratégie {self.name!r}: {sorted(unknown)}"
            )
        self.params: dict[str, Any] = {**self.default_params, **params}

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Renvoie un DataFrame aligné sur ``df.index`` avec les colonnes SIGNAL_COLUMNS."""

    def warmup_bars(self) -> int:
        """Nombre de bougies nécessaires avant que les signaux soient fiables."""
        return 0

    # --- utilitaires ---------------------------------------------------------

    def empty_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """DataFrame de signaux vide (flat, sans stop) à remplir par la sous-classe."""
        return pd.DataFrame(
            {
                "signal": np.zeros(len(df), dtype=np.int64),
                "stop_loss": np.full(len(df), np.nan),
                "take_profit": np.full(len(df), np.nan),
            },
            index=df.index,
        )

    @staticmethod
    def validate_signals(df: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        """Vérifie le format des signaux et normalise les types."""
        missing = [c for c in SIGNAL_COLUMNS if c not in signals.columns]
        if missing:
            raise ValueError(f"Colonnes de signaux manquantes : {missing}")
        if not signals.index.equals(df.index):
            raise ValueError("L'index des signaux doit être identique à celui des bougies")
        out = signals.loc[:, list(SIGNAL_COLUMNS)].copy()
        sig = out["signal"].fillna(0).astype(np.int64)
        if not sig.isin([-1, 0, 1]).all():
            raise ValueError("`signal` doit valoir -1, 0 ou 1")
        out["signal"] = sig
        out["stop_loss"] = out["stop_loss"].astype(float)
        out["take_profit"] = out["take_profit"].astype(float)
        return out

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.params})"
