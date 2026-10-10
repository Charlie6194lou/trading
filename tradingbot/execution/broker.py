"""Contrat commun à tous les brokers (paper = fictif, ccxt = réel).

Conventions :
- ``amount`` est toujours en unités de l'actif de base (ex. BTC pour BTC/USDT), positif.
- Une position ``amount > 0`` est longue, ``< 0`` est courte, ``== 0`` est plate.
- Les prix et le cash sont dans la devise de cotation (ex. USDT).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

Side = Literal["buy", "sell"]


@dataclass
class Order:
    id: str
    symbol: str
    side: Side
    amount: float
    price: float  # prix moyen d'exécution
    fee: float  # frais payés, en devise de cotation
    timestamp: datetime
    # "filled", "partial" (annulé par l'exchange après exécution partielle), "rejected", "open"
    status: str = "filled"
    info: dict = field(default_factory=dict)

    @property
    def cost(self) -> float:
        """Valeur notionnelle de l'ordre (hors frais)."""
        return self.amount * self.price


@dataclass
class Position:
    symbol: str
    amount: float = 0.0  # signé : >0 long, <0 short
    entry_price: float = 0.0  # prix moyen d'entrée (0 si plat)
    opened_at: datetime | None = None

    @property
    def is_open(self) -> bool:
        return abs(self.amount) > 1e-12

    @property
    def side(self) -> int:
        return 0 if not self.is_open else (1 if self.amount > 0 else -1)

    def unrealized_pnl(self, price: float) -> float:
        return (price - self.entry_price) * self.amount


@dataclass
class Balance:
    cash: float  # devise de cotation disponible
    equity: float  # cash + valeur des positions au prix courant


class BrokerError(RuntimeError):
    """Erreur d'exécution (ordre rejeté, fonds insuffisants, API indisponible...)."""


class Broker(ABC):
    """Interface minimale utilisée par la boucle live et le paper trading."""

    @abstractmethod
    def get_balance(self, symbol: str) -> Balance:
        """Cash + équité, évaluée au dernier prix connu de ``symbol``."""

    @abstractmethod
    def get_position(self, symbol: str) -> Position:
        """Position courante (``amount == 0`` si aucune)."""

    @abstractmethod
    def get_price(self, symbol: str) -> float:
        """Dernier prix connu."""

    @abstractmethod
    def market_order(self, symbol: str, side: Side, amount: float) -> Order:
        """Passe un ordre au marché. Lève ``BrokerError`` en cas d'échec."""

    def close_position(self, symbol: str) -> Order | None:
        """Ferme la position courante (si elle existe) par un ordre au marché."""
        pos = self.get_position(symbol)
        if not pos.is_open:
            return None
        side: Side = "sell" if pos.amount > 0 else "buy"
        return self.market_order(symbol, side, abs(pos.amount))

    @staticmethod
    def now() -> datetime:
        return datetime.now(UTC)
