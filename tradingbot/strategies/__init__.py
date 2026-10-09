"""Registre des stratégies disponibles.

Usage::

    from tradingbot.strategies import get_strategy, list_strategies
    strat = get_strategy("sma_cross", fast=10, slow=30)

Chaque module de stratégie s'enregistre avec ``@register`` et est importé en bas
de ce fichier pour peupler le registre.
"""

from __future__ import annotations

from typing import Any

from tradingbot.strategies.base import SIGNAL_COLUMNS, Strategy

STRATEGIES: dict[str, type[Strategy]] = {}


def register(cls: type[Strategy]) -> type[Strategy]:
    """Décorateur : ajoute la stratégie au registre, clé = ``cls.name``."""
    if not cls.name or cls.name == "base":
        raise ValueError(f"La stratégie {cls.__name__} doit définir un attribut `name`")
    if cls.name in STRATEGIES and STRATEGIES[cls.name] is not cls:
        raise ValueError(f"Stratégie déjà enregistrée : {cls.name!r}")
    STRATEGIES[cls.name] = cls
    return cls


def get_strategy(name: str, **params: Any) -> Strategy:
    """Instancie une stratégie par son nom (lève KeyError si inconnue)."""
    try:
        cls = STRATEGIES[name]
    except KeyError:
        raise KeyError(
            f"Stratégie inconnue : {name!r}. Disponibles : {', '.join(sorted(STRATEGIES))}"
        ) from None
    return cls(**params)


def list_strategies() -> list[tuple[str, str]]:
    """Liste (nom, description) triée par nom."""
    return sorted((n, c.description) for n, c in STRATEGIES.items())


# Importe les modules de stratégies pour les enregistrer (effets de bord volontaires).
# Chaque module ajouté par un contributeur doit être listé ici.
from tradingbot.strategies import (  # noqa: E402, F401
    bollinger_reversion,
    donchian_breakout,
    ema_trend,
    rsi_reversion,
    sma_cross,
)

__all__ = [
    "SIGNAL_COLUMNS",
    "STRATEGIES",
    "Strategy",
    "get_strategy",
    "list_strategies",
    "register",
]
