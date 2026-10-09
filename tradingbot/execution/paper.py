"""Broker fictif (« paper trading »).

Simule un compte au comptant : chaque ordre au marché est exécuté immédiatement au
dernier prix connu, dégradé d'un glissement défavorable (``slippage``), et des frais
proportionnels au notionnel (``fee_rate``) sont retirés du cash.

- Le prix vient de ``price_feed(symbol)`` si un flux est fourni, sinon de la dernière
  valeur passée à :meth:`PaperBroker.set_price`.
- Un achat dont le coût (frais inclus) dépasse le cash disponible est refusé
  (``BrokerError``) ; une vente supérieure à la position détenue est refusée sauf si
  ``allow_short=True`` (la position devient alors courte, sans levier : le notionnel
  vendu à découvert ne peut pas dépasser l'équité).
- Les positions portent un montant signé et un prix d'entrée moyen pondéré par les
  volumes (remis à zéro quand la position est plate).
- Chaque ordre exécuté reçoit un identifiant séquentiel ``paper-1``, ``paper-2``, ... et
  est ajouté à ``self.orders``. Un ordre refusé ne modifie rien.
- Si ``state_path`` est défini, l'état complet (cash, positions, ordres) est sauvegardé
  en JSON après chaque ordre et rechargé automatiquement à la construction si le
  fichier existe.
"""

from __future__ import annotations

import json
import logging
import math
import numbers
import os
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from tradingbot.execution.broker import Balance, Broker, BrokerError, Order, Position, Side

logger = logging.getLogger(__name__)

#: en dessous de cette quantité (en unités de l'actif), une position est considérée plate
FLAT_TOLERANCE = 1e-9
#: tolérance sur la comparaison coût / cash (erreurs d'arrondi flottant)
CASH_TOLERANCE = 1e-9
STATE_VERSION = 1
ORDER_ID_PREFIX = "paper-"


def apply_fill(position: Position, delta: float, price: float, now: datetime) -> Position:
    """Applique une exécution de ``delta`` unités (signé : + achat, - vente) à ``position``.

    Règles :
    - position plate ou retournement (le signe change) → nouvelle entrée au prix ``price`` ;
    - renforcement (même signe) → prix d'entrée moyen pondéré par les volumes ;
    - réduction partielle → prix d'entrée inchangé ;
    - position résultante nulle → position plate (prix d'entrée 0, ``opened_at`` None).

    Renvoie une nouvelle :class:`Position` (l'objet d'origine n'est pas modifié).
    """
    old = position.amount
    new = old + delta
    if abs(new) <= FLAT_TOLERANCE:
        return Position(position.symbol)
    if abs(old) <= FLAT_TOLERANCE or (old > 0) != (new > 0):
        return Position(position.symbol, new, price, now)
    if (delta > 0) == (old > 0):
        entry = (abs(old) * position.entry_price + abs(delta) * price) / abs(new)
        return Position(position.symbol, new, entry, position.opened_at)
    return Position(position.symbol, new, position.entry_price, position.opened_at)


def check_order_args(side: str, amount: float) -> None:
    """Vérifie ``side`` et ``amount`` ; lève ``BrokerError`` (message en français) sinon."""
    if side not in ("buy", "sell"):
        raise BrokerError(f"Sens d'ordre invalide : {side!r} (attendu 'buy' ou 'sell').")
    if not isinstance(amount, numbers.Real) or isinstance(amount, bool):
        raise BrokerError(f"Montant d'ordre invalide : {amount!r} (nombre attendu).")
    if not math.isfinite(amount) or amount <= 0:
        raise BrokerError(f"Montant d'ordre invalide : {amount!r} (doit être un nombre > 0).")


class PaperBroker(Broker):
    """Broker fictif : argent virtuel, exécution immédiate au prix simulé.

    Paramètres :
    - ``initial_cash`` : capital de départ en devise de cotation ;
    - ``fee_rate`` : frais par ordre, fraction du notionnel (0.001 = 0,1 %) ;
    - ``slippage`` : glissement défavorable par ordre, fraction du prix ;
    - ``price_feed`` : fonction ``symbol -> prix`` ; si ``None``, utiliser ``set_price`` ;
    - ``state_path`` : fichier JSON d'état (chargé à l'init s'il existe, sauvegardé après
      chaque ordre) ;
    - ``allow_short`` : autorise la vente à découvert (sans levier).
    """

    def __init__(
        self,
        initial_cash: float,
        fee_rate: float = 0.001,
        slippage: float = 0.0005,
        price_feed: Callable[[str], float] | None = None,
        state_path: str | Path | None = None,
        allow_short: bool = False,
    ) -> None:
        if not math.isfinite(initial_cash) or initial_cash < 0:
            raise ValueError(f"initial_cash doit être un nombre >= 0 (reçu {initial_cash!r}).")
        if not math.isfinite(fee_rate) or fee_rate < 0:
            raise ValueError(f"fee_rate doit être une fraction >= 0 (reçu {fee_rate!r}).")
        if not math.isfinite(slippage) or not 0 <= slippage < 1:
            raise ValueError(f"slippage doit être une fraction dans [0, 1[ (reçu {slippage!r}).")
        self.initial_cash = float(initial_cash)
        self.fee_rate = float(fee_rate)
        self.slippage = float(slippage)
        self.price_feed = price_feed
        self.state_path: Path | None = Path(state_path) if state_path is not None else None
        self.allow_short = bool(allow_short)

        self.cash: float = self.initial_cash
        self.positions: dict[str, Position] = {}
        self.orders: list[Order] = []
        self._prices: dict[str, float] = {}
        self._order_seq = 0

        if self.state_path is not None and self.state_path.exists():
            self.load()
            logger.info(
                "État du paper broker rechargé depuis %s (cash=%.2f, %d position(s), %d ordre(s))",
                self.state_path,
                self.cash,
                sum(1 for p in self.positions.values() if p.is_open),
                len(self.orders),
            )

    # ------------------------------------------------------------------ prix
    def set_price(self, symbol: str, price: float) -> None:
        """Fixe manuellement le dernier prix de ``symbol`` (utilisé si ``price_feed`` est None)."""
        self._prices[symbol] = self._validate_price(symbol, price)

    def get_price(self, symbol: str) -> float:
        """Dernier prix connu : ``price_feed(symbol)`` si un flux existe, sinon ``set_price``."""
        if self.price_feed is not None:
            try:
                price = self.price_feed(symbol)
            except BrokerError:
                raise
            except Exception as exc:
                raise BrokerError(f"Le flux de prix a échoué pour {symbol} : {exc}") from exc
        else:
            price = self._prices.get(symbol)
            if price is None:
                raise BrokerError(
                    f"Aucun prix connu pour {symbol} : appelez set_price(symbol, prix) "
                    "ou fournissez un price_feed."
                )
        return self._validate_price(symbol, price)

    @staticmethod
    def _validate_price(symbol: str, price: Any) -> float:
        try:
            value = float(price)
        except (TypeError, ValueError) as exc:
            raise BrokerError(f"Prix invalide pour {symbol} : {price!r}.") from exc
        if not math.isfinite(value) or value <= 0:
            raise BrokerError(f"Prix invalide pour {symbol} : {price!r} (doit être > 0).")
        return value

    # ------------------------------------------------------------------ état du compte
    def get_balance(self, symbol: str) -> Balance:
        """Cash disponible et équité = cash + Σ montant × prix courant des positions ouvertes."""
        equity = self.cash
        for sym, pos in self.positions.items():
            if pos.is_open:
                equity += pos.amount * self.get_price(sym)
        return Balance(cash=self.cash, equity=equity)

    def get_position(self, symbol: str) -> Position:
        """Copie de la position courante (plate si aucune)."""
        pos = self.positions.get(symbol)
        if pos is None or not pos.is_open:
            return Position(symbol)
        return replace(pos)

    # ------------------------------------------------------------------ ordres
    def market_order(self, symbol: str, side: Side, amount: float) -> Order:
        """Exécute immédiatement un ordre au marché au prix courant ± slippage.

        Lève ``BrokerError`` si le montant est invalide, si le prix est inconnu, si le
        cash est insuffisant (achat) ou si la vente dépasse la position détenue
        (``allow_short=False``). Un ordre refusé ne modifie ni le cash ni les positions.
        """
        check_order_args(side, amount)
        amount = float(amount)
        base = symbol.split("/")[0] if "/" in symbol else symbol
        price = self.get_price(symbol)
        exec_price = price * (1 + self.slippage) if side == "buy" else price * (1 - self.slippage)
        notional = amount * exec_price
        fee = notional * self.fee_rate
        pos = self.positions.get(symbol, Position(symbol))

        if side == "buy":
            cost = notional + fee
            if cost > self.cash + CASH_TOLERANCE:
                raise BrokerError(
                    f"Fonds insuffisants pour acheter {amount:g} {base} : coût {cost:.2f} "
                    f"(frais inclus) > cash disponible {self.cash:.2f}."
                )
            delta = amount
            cash_after = self.cash - cost
        else:
            if amount > pos.amount + FLAT_TOLERANCE:
                if not self.allow_short:
                    if pos.amount <= FLAT_TOLERANCE:
                        raise BrokerError(
                            f"Vente à découvert interdite (allow_short=False) : aucune position "
                            f"à vendre sur {symbol}."
                        )
                    raise BrokerError(
                        f"Vente de {amount:g} {base} supérieure à la position détenue "
                        f"({pos.amount:g}) sur {symbol} : vente à découvert interdite "
                        "(allow_short=False)."
                    )
                short_notional = (amount - max(pos.amount, 0.0)) * price
                equity = self.get_balance(symbol).equity
                if short_notional > equity + CASH_TOLERANCE:
                    raise BrokerError(
                        f"Vente à découvert refusée : le notionnel court ({short_notional:.2f}) "
                        f"dépasserait l'équité ({equity:.2f}) ; pas de levier en paper trading."
                    )
            delta = -amount
            cash_after = self.cash + notional - fee

        now = self.now()
        self.cash = cash_after
        self.positions[symbol] = apply_fill(pos, delta, exec_price, now)
        self._order_seq += 1
        order = Order(
            id=f"{ORDER_ID_PREFIX}{self._order_seq}",
            symbol=symbol,
            side=side,
            amount=amount,
            price=exec_price,
            fee=fee,
            timestamp=now,
            status="filled",
            info={
                "reference_price": price,
                "slippage": self.slippage,
                "fee_rate": self.fee_rate,
                "cash_after": self.cash,
            },
        )
        self.orders.append(order)
        logger.info(
            "Ordre papier %s : %s %.8g %s @ %.8g (frais %.4f) → cash %.2f, position %.8g",
            order.id,
            side,
            amount,
            symbol,
            exec_price,
            fee,
            self.cash,
            self.positions[symbol].amount,
        )
        if self.state_path is not None:
            self.save()
        return order

    # ------------------------------------------------------------------ persistance
    def state(self) -> dict[str, Any]:
        """État complet sérialisable en JSON (timestamps au format ISO 8601)."""
        return {
            "version": STATE_VERSION,
            "initial_cash": self.initial_cash,
            "cash": self.cash,
            "order_seq": self._order_seq,
            "positions": {
                sym: {
                    "symbol": pos.symbol,
                    "amount": pos.amount,
                    "entry_price": pos.entry_price,
                    "opened_at": pos.opened_at.isoformat() if pos.opened_at else None,
                }
                for sym, pos in self.positions.items()
                if pos.is_open
            },
            "orders": [
                {
                    "id": o.id,
                    "symbol": o.symbol,
                    "side": o.side,
                    "amount": o.amount,
                    "price": o.price,
                    "fee": o.fee,
                    "timestamp": o.timestamp.isoformat(),
                    "status": o.status,
                    "info": o.info,
                }
                for o in self.orders
            ],
        }

    def load_state(self, data: dict[str, Any]) -> None:
        """Restaure un état produit par :meth:`state` (``initial_cash`` reste celui de l'init)."""
        if not isinstance(data, dict):
            raise ValueError("l'état doit être un objet JSON")
        cash = float(data["cash"])
        if not math.isfinite(cash):
            raise ValueError("cash invalide")
        positions: dict[str, Position] = {}
        for sym, p in (data.get("positions") or {}).items():
            opened = p.get("opened_at")
            positions[sym] = Position(
                symbol=str(p.get("symbol", sym)),
                amount=float(p["amount"]),
                entry_price=float(p.get("entry_price", 0.0)),
                opened_at=datetime.fromisoformat(opened) if opened else None,
            )
        orders = [
            Order(
                id=str(o["id"]),
                symbol=str(o["symbol"]),
                side=o["side"],
                amount=float(o["amount"]),
                price=float(o["price"]),
                fee=float(o.get("fee", 0.0)),
                timestamp=datetime.fromisoformat(o["timestamp"]),
                status=str(o.get("status", "filled")),
                info=dict(o.get("info") or {}),
            )
            for o in data.get("orders") or []
        ]
        for o in orders:
            if o.side not in ("buy", "sell"):
                raise ValueError(f"sens d'ordre invalide : {o.side!r}")
        seq = data.get("order_seq")
        if seq is None:
            seq = max(
                (int(o.id[len(ORDER_ID_PREFIX) :]) for o in orders if _is_paper_id(o.id)),
                default=len(orders),
            )
        self.cash = cash
        self.positions = positions
        self.orders = orders
        self._order_seq = int(seq)

    def save(self) -> None:
        """Écrit l'état JSON dans ``state_path`` (écriture atomique). ``BrokerError`` sans chemin."""
        if self.state_path is None:
            raise BrokerError(
                "Aucun fichier d'état (state_path) n'est configuré pour ce PaperBroker."
            )
        path = self.state_path
        tmp = path.with_name(path.name + ".tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(
                json.dumps(self.state(), indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            os.replace(tmp, path)
        except OSError as exc:
            raise BrokerError(
                f"Impossible d'écrire l'état du paper broker dans {path} : {exc}"
            ) from exc

    def load(self) -> None:
        """Recharge l'état JSON depuis ``state_path``. ``BrokerError`` si absent ou invalide."""
        if self.state_path is None:
            raise BrokerError(
                "Aucun fichier d'état (state_path) n'est configuré pour ce PaperBroker."
            )
        path = self.state_path
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BrokerError(
                f"Impossible de lire l'état du paper broker ({path}) : {exc}"
            ) from exc
        try:
            self.load_state(data)
        except (KeyError, TypeError, ValueError) as exc:
            raise BrokerError(f"Fichier d'état du paper broker invalide ({path}) : {exc}") from exc

    def reset(self) -> None:
        """Remet le compte à zéro : cash initial, aucune position, aucun ordre (et sauvegarde)."""
        self.cash = self.initial_cash
        self.positions = {}
        self.orders = []
        self._order_seq = 0
        logger.info("Paper broker réinitialisé (cash=%.2f)", self.cash)
        if self.state_path is not None:
            self.save()

    def __repr__(self) -> str:
        open_positions = {s: round(p.amount, 8) for s, p in self.positions.items() if p.is_open}
        return (
            f"PaperBroker(cash={self.cash:.2f}, positions={open_positions}, "
            f"orders={len(self.orders)}, allow_short={self.allow_short})"
        )


def _is_paper_id(order_id: str) -> bool:
    return order_id.startswith(ORDER_ID_PREFIX) and order_id[len(ORDER_ID_PREFIX) :].isdigit()
