"""Broker réel via ``ccxt`` (marché au comptant uniquement, ordres au marché).

Garde-fous :

- ``BrokerError`` à la construction si ``confirm_live`` est False (double sécurité avec
  la commande ``live`` de la CLI) ;
- **jamais de retrait** (``withdraw``), **jamais de levier ni de marge** : seul
  ``create_order(symbol, "market", side, amount)`` est utilisé pour trader ;
- montant arrondi à la précision du marché (``amount_to_precision``) puis vérifié
  contre les minimums ``limits.amount.min`` et ``limits.cost.min`` du marché ;
- toute exception ``ccxt.BaseError`` est encapsulée dans ``BrokerError``, après
  **masquage des secrets** (:func:`scrub_secrets`) : certains exchanges signent via la
  chaîne de requête et ccxt recopie l'URL complète dans ses messages d'erreur ;
- statut des ordres : une réponse ``canceled`` / ``rejected`` / ``expired`` **sans
  exécution** lève ``BrokerError`` (rien n'a été acheté ni vendu) ; partiellement
  exécutée, elle renvoie un :class:`~tradingbot.execution.broker.Order` de statut
  ``"partial"`` dont ``amount`` est la quantité réellement exécutée ;
- une quantité détenue inférieure au minimum du marché (``amount.min``) **ou** dont la
  valeur est sous ``cost.min`` (« poussière » invendable) est considérée plate ;
- sur les exchanges dont ``options["createMarketBuyOrderRequiresPrice"]`` est vrai
  (Coinbase, Gate...), le prix de référence est transmis à ``create_order`` pour les
  achats au marché (ccxt convertit lui-même la quantité en montant à dépenser).

L'objet ``exchange`` est injectable (faux exchange dans les tests) : ``ccxt`` n'est
instancié que lorsqu'il vaut ``None``. Les marchés sont chargés paresseusement au
premier besoin puis mis en cache.

Le prix d'entrée d'une position n'est pas fourni par un exchange spot : il est suivi
localement, en mémoire, à partir des exécutions passées par ce broker (0 si inconnu).
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

import ccxt

from tradingbot.execution.broker import Balance, Broker, BrokerError, Order, Position, Side
from tradingbot.execution.paper import apply_fill, check_order_args

logger = logging.getLogger(__name__)

__all__ = ["SECRET_MASK", "CcxtBroker", "scrub_secrets", "split_symbol"]

#: correspondance statut ccxt → statut :class:`Order`
_STATUS_MAP = {
    "closed": "filled",
    "filled": "filled",
    "open": "open",
    "canceled": "rejected",
    "cancelled": "rejected",
    "rejected": "rejected",
    "expired": "rejected",
}
#: marque qui remplace un secret dans les messages d'erreur et le journal
SECRET_MASK = "***"
#: paramètres d'URL qui transportent une clé ou une signature (exchanges signant par la
#: chaîne de requête : htx, bithumb, upbit...)
_SECRET_QUERY_RE = re.compile(
    r"([?&](?:AccessKeyId|api_key|apiKey|apikey|key|access_key|secret|signature|sign)=)[^&\s]+",
    re.IGNORECASE,
)
#: chaîne de requête complète d'une URL (horodatage, signature... sans valeur pour l'utilisateur)
_URL_QUERY_RE = re.compile(r"(https?://[^\s?\"'<>]+)\?[^\s\"'<>]*")


def scrub_secrets(text: str, secrets: Iterable[str | None] = ()) -> str:
    """Masque les secrets dans ``text`` : chaque valeur de ``secrets`` est remplacée par
    ``***``, les paramètres d'URL sensibles (``AccessKeyId=``, ``Signature=``...) sont
    masqués et la chaîne de requête de toute URL ``http(s)://`` est tronquée."""
    out = str(text)
    for secret in secrets:
        if secret and str(secret).strip():
            out = out.replace(str(secret), SECRET_MASK)
    out = _SECRET_QUERY_RE.sub(rf"\1{SECRET_MASK}", out)
    return _URL_QUERY_RE.sub(rf"\1?{SECRET_MASK}", out)


@contextmanager
def _wrap_ccxt(action: str, secrets: Iterable[str | None] = ()) -> Iterator[None]:
    """Transforme toute ``ccxt.BaseError`` levée dans le bloc en ``BrokerError`` (FR), en
    masquant les secrets dans le message (et dans l'exception d'origine, conservée comme
    ``__cause__`` : une trace au niveau debug ne doit pas les révéler non plus)."""
    try:
        yield
    except BrokerError:
        raise
    except ccxt.BaseError as exc:
        secrets = tuple(secrets)
        exc.args = tuple(scrub_secrets(a, secrets) if isinstance(a, str) else a for a in exc.args)
        raise BrokerError(
            f"Erreur de l'exchange pendant « {action} » : {type(exc).__name__}: "
            f"{scrub_secrets(str(exc), secrets)}"
        ) from exc


def split_symbol(symbol: str) -> tuple[str, str]:
    """``"BTC/USDT"`` → ``("BTC", "USDT")`` ; ``BrokerError`` si le format est invalide."""
    base, sep, rest = symbol.partition("/")
    quote = rest.split(":")[0]
    if not sep or not base or not quote:
        raise BrokerError(
            f"Symbole invalide : {symbol!r} (format attendu 'BASE/QUOTE', ex. 'BTC/USDT')."
        )
    return base, quote


def _to_float(value: Any) -> float | None:
    """Convertit en float fini, ou ``None`` si absent / non numérique."""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _balance_amount(balance: dict[str, Any], kind: str, currency: str) -> float:
    """Montant ``kind`` ('free' | 'used' | 'total') de ``currency`` dans une réponse ccxt."""
    section = balance.get(kind)
    value = section.get(currency) if isinstance(section, dict) else None
    if value is None:
        per_currency = balance.get(currency)
        value = per_currency.get(kind) if isinstance(per_currency, dict) else None
    return _to_float(value) or 0.0


def _limit(market: dict[str, Any], kind: str, bound: str) -> float | None:
    limits = market.get("limits")
    if not isinstance(limits, dict):
        return None
    entry = limits.get(kind)
    if not isinstance(entry, dict):
        return None
    value = _to_float(entry.get(bound))
    return value if value is not None and value > 0 else None


class CcxtBroker(Broker):
    """Broker réel sur un exchange ``ccxt`` (spot, ordres au marché uniquement).

    Paramètres :
    - ``exchange_id`` : identifiant ccxt (``"binance"``, ``"kraken"``...) ;
    - ``api_key`` / ``api_secret`` : clés API (jamais journalisées ; conservées en privé
      uniquement pour être masquées dans les messages d'erreur) ;
    - ``sandbox`` : active le mode test de l'exchange (``set_sandbox_mode(True)``) ;
    - ``confirm_live`` : doit valoir True, sinon ``BrokerError`` ;
    - ``exchange`` : objet exchange déjà construit (fake dans les tests).
    """

    def __init__(
        self,
        exchange_id: str,
        api_key: str,
        api_secret: str,
        sandbox: bool = False,
        confirm_live: bool = False,
        exchange: Any | None = None,
    ) -> None:
        if not confirm_live:
            raise BrokerError(
                "Refus de créer un broker réel : passez confirm_live=True pour confirmer "
                "explicitement que des ordres avec de l'argent réel peuvent être envoyés."
            )
        self.exchange_id = str(exchange_id)
        self.sandbox = bool(sandbox)
        self._secrets: tuple[str, ...] = tuple(
            str(x) for x in (api_key, api_secret) if x and str(x).strip()
        )
        if exchange is None:
            exchange = self._build_exchange(self.exchange_id, api_key, api_secret)
        if self.sandbox:
            set_sandbox = getattr(exchange, "set_sandbox_mode", None)
            if set_sandbox is None:
                raise BrokerError(
                    f"L'exchange {self.exchange_id!r} ne propose pas de mode sandbox."
                )
            with self._wrap("activation du mode sandbox"):
                set_sandbox(True)
        self.exchange = exchange
        self._markets: dict[str, Any] | None = None
        self._tracked: dict[str, Position] = {}
        logger.info(
            "CcxtBroker prêt sur %s (%s)", self.exchange_id, "sandbox" if self.sandbox else "RÉEL"
        )

    def _build_exchange(self, exchange_id: str, api_key: str, api_secret: str) -> Any:
        if exchange_id not in ccxt.exchanges:
            raise BrokerError(
                f"Exchange inconnu pour ccxt : {exchange_id!r} (exemples : binance, kraken, "
                "coinbase, bybit)."
            )
        exchange_cls = getattr(ccxt, exchange_id)
        with self._wrap(f"construction de l'exchange {exchange_id}"):
            return exchange_cls({"apiKey": api_key, "secret": api_secret, "enableRateLimit": True})

    def _wrap(self, action: str) -> Any:
        """``_wrap_ccxt`` avec les secrets de ce broker à masquer."""
        return _wrap_ccxt(action, self._secrets)

    def scrub(self, text: str) -> str:
        """Masque la clé et le secret API de ce broker dans ``text`` (voir :func:`scrub_secrets`)."""
        return scrub_secrets(text, self._secrets)

    # ------------------------------------------------------------------ marchés
    def _load_markets(self) -> dict[str, Any]:
        if self._markets is None:
            with self._wrap("chargement des marchés"):
                markets = self.exchange.load_markets()
            self._markets = dict(markets or {})
            logger.info(
                "Marchés chargés sur %s : %d symboles", self.exchange_id, len(self._markets)
            )
        return self._markets

    def _market(self, symbol: str) -> dict[str, Any]:
        market = self._load_markets().get(symbol)
        if not isinstance(market, dict):
            raise BrokerError(f"Symbole {symbol!r} inconnu sur l'exchange {self.exchange_id}.")
        return market

    # ------------------------------------------------------------------ lecture
    def get_price(self, symbol: str) -> float:
        """Dernier prix via ``fetch_ticker`` (``last``, sinon ``close``)."""
        with self._wrap(f"récupération du prix de {symbol}"):
            ticker = self.exchange.fetch_ticker(symbol)
        ticker = ticker if isinstance(ticker, dict) else {}
        for key in ("last", "close"):
            price = _to_float(ticker.get(key))
            if price is not None and price > 0:
                return price
        raise BrokerError(
            f"Prix indisponible pour {symbol} : le ticker ne contient ni 'last' ni 'close'."
        )

    def _fetch_balance(self) -> dict[str, Any]:
        with self._wrap("récupération du solde"):
            balance = self.exchange.fetch_balance()
        return balance if isinstance(balance, dict) else {}

    def get_balance(self, symbol: str) -> Balance:
        """Cash = devise de cotation libre ; équité = cash + total de l'actif de base × prix."""
        base, quote = split_symbol(symbol)
        balance = self._fetch_balance()
        cash = _balance_amount(balance, "free", quote)
        base_total = _balance_amount(balance, "total", base)
        equity = cash
        if base_total > 0:
            equity += base_total * self.get_price(symbol)
        return Balance(cash=cash, equity=equity)

    def get_position(self, symbol: str) -> Position:
        """Position spot (longue uniquement) d'après le solde total de l'actif de base.

        Une quantité inférieure au minimum négociable du marché (``limits.amount.min``),
        ou dont la valeur au dernier prix est sous le notionnel minimal
        (``limits.cost.min``), est de la « poussière » invendable : elle est considérée
        comme plate. Le prix d'entrée vient du suivi local des exécutions (0 si inconnu).
        """
        base, quote = split_symbol(symbol)
        amount = max(_balance_amount(self._fetch_balance(), "total", base), 0.0)
        market = self._market(symbol)
        min_amount = _limit(market, "amount", "min")
        if min_amount is not None and amount < min_amount:
            amount = 0.0
        min_cost = _limit(market, "cost", "min")
        if amount > 0 and min_cost is not None:
            value = amount * self.get_price(symbol)
            if value < min_cost:
                logger.debug(
                    "Poussière ignorée sur %s : %.8g %s ≈ %.2f %s < minimum %g %s",
                    symbol,
                    amount,
                    base,
                    value,
                    quote,
                    min_cost,
                    quote,
                )
                amount = 0.0
        if amount <= 0:
            self._tracked.pop(symbol, None)
            return Position(symbol)
        tracked = self._tracked.get(symbol)
        if tracked is None:
            return Position(symbol, amount, 0.0, None)
        return Position(symbol, amount, tracked.entry_price, tracked.opened_at)

    # ------------------------------------------------------------------ ordres
    def market_order(self, symbol: str, side: Side, amount: float) -> Order:
        """Envoie un ordre au marché après arrondi et vérification des minimums du marché."""
        check_order_args(side, amount)
        base, quote = split_symbol(symbol)
        market = self._market(symbol)

        with self._wrap(f"arrondi du montant pour {symbol}"):
            rounded = self.exchange.amount_to_precision(symbol, amount)
        amount_p = _to_float(rounded)
        if amount_p is None or amount_p <= 0:
            raise BrokerError(
                f"Montant {amount:g} {base} trop petit : il devient nul après arrondi à la "
                f"précision du marché {symbol}."
            )
        min_amount = _limit(market, "amount", "min")
        if min_amount is not None and amount_p < min_amount:
            raise BrokerError(
                f"Montant {amount_p:g} {base} inférieur au minimum de l'exchange "
                f"({min_amount:g} {base}) pour {symbol} : ordre non envoyé."
            )
        reference_price: float | None = None
        min_cost = _limit(market, "cost", "min")
        if min_cost is not None:
            reference_price = self.get_price(symbol)
            notional = amount_p * reference_price
            if notional < min_cost:
                raise BrokerError(
                    f"Notionnel {notional:.2f} {quote} inférieur au minimum de l'exchange "
                    f"({min_cost:g} {quote}) pour {symbol} : ordre non envoyé."
                )

        needs_price = side == "buy" and self._market_buy_requires_price()
        if needs_price and reference_price is None:
            reference_price = self.get_price(symbol)

        logger.info("Envoi ordre %s %s %.8g %s", side, "market", amount_p, symbol)
        with self._wrap(f"ordre {side} {amount_p:g} {symbol}"):
            if needs_price:
                # ccxt convertit lui-même quantité × prix en montant à dépenser (quote).
                response = self.exchange.create_order(
                    symbol, "market", side, amount_p, reference_price
                )
            else:
                response = self.exchange.create_order(symbol, "market", side, amount_p)
        order = self._parse_order(
            response if isinstance(response, dict) else {},
            symbol=symbol,
            side=side,
            requested_amount=amount,
            sent_amount=amount_p,
            reference_price=reference_price,
        )
        if order.status == "rejected":
            raise BrokerError(
                f"Ordre {order.id} ({side} {amount_p:g} {symbol}) rejeté par l'exchange "
                f"(statut {order.info.get('exchange_status')!r}) : aucune exécution, "
                "rien n'a été acheté ni vendu."
            )
        if order.status == "partial":
            logger.warning(
                "Ordre %s %s exécuté partiellement : %.8g sur %.8g %s (statut %r)",
                self.exchange_id,
                order.id,
                order.amount,
                amount_p,
                base,
                order.info.get("exchange_status"),
            )
        self._track_fill(order)
        logger.info(
            "Ordre %s %s : %s %.8g %s @ %.8g (frais %.4f %s, statut %s)",
            self.exchange_id,
            order.id,
            side,
            order.amount,
            symbol,
            order.price,
            order.fee,
            quote,
            order.status,
        )
        return order

    def _market_buy_requires_price(self) -> bool:
        """True si l'exchange exige un prix pour un achat au marché (``options`` ccxt)."""
        options = getattr(self.exchange, "options", None)
        if not isinstance(options, dict):
            return False
        return bool(options.get("createMarketBuyOrderRequiresPrice", False))

    def _parse_order(
        self,
        response: dict[str, Any],
        *,
        symbol: str,
        side: Side,
        requested_amount: float,
        sent_amount: float,
        reference_price: float | None,
    ) -> Order:
        """Convertit la réponse ccxt de ``create_order`` en :class:`Order`.

        Statut : ``"filled"`` (ou ``"open"``) selon la réponse ; ``"rejected"`` si
        l'exchange a annulé / rejeté l'ordre sans rien exécuter (``amount`` = 0) ;
        ``"partial"`` s'il l'a annulé après une exécution partielle (``amount`` = quantité
        exécutée). Une réponse acceptée sans champ ``filled`` suppose ``sent_amount``.
        """
        base, quote = split_symbol(symbol)
        timestamp = self._parse_timestamp(response)

        raw_status = response.get("status")
        status = _STATUS_MAP.get(str(raw_status).lower(), "filled") if raw_status else "filled"
        filled = _to_float(response.get("filled"))
        if status == "rejected":
            filled = max(filled or 0.0, 0.0)
            if filled > 0:
                status = "partial"
        elif filled is None or filled <= 0:
            logger.warning(
                "Réponse de l'exchange sans quantité exécutée pour %s ; on suppose %.8g",
                symbol,
                sent_amount,
            )
            filled = sent_amount

        price = _to_float(response.get("average"))
        if price is None or price <= 0:
            price = _to_float(response.get("price"))
        if price is None or price <= 0:
            cost = _to_float(response.get("cost"))
            price = cost / filled if cost is not None and cost > 0 and filled > 0 else None
        if price is None or price <= 0:
            price = reference_price if reference_price is not None else self.get_price(symbol)

        fee_cost, fee_currency = self._extract_fee(response)
        if fee_cost and fee_currency == base:
            fee_cost *= price
        elif fee_cost and fee_currency not in (None, quote):
            logger.warning(
                "Frais payés en %s (non converti en %s) pour l'ordre %s",
                fee_currency,
                quote,
                response.get("id"),
            )
            fee_cost = 0.0

        order_id = response.get("id")
        order_id = (
            str(order_id)
            if order_id not in (None, "")
            else f"ccxt-{int(timestamp.timestamp() * 1000)}"
        )
        return Order(
            id=order_id,
            symbol=symbol,
            side=side,
            amount=filled,
            price=price,
            fee=fee_cost,
            timestamp=timestamp,
            status=status,
            info={
                "requested_amount": requested_amount,
                "sent_amount": sent_amount,
                "fee_currency": fee_currency,
                "exchange_status": raw_status,
                "raw": response,
            },
        )

    def _parse_timestamp(self, response: dict[str, Any]) -> datetime:
        ms = _to_float(response.get("timestamp"))
        if ms is not None and ms > 0:
            return datetime.fromtimestamp(ms / 1000.0, tz=UTC)
        iso = response.get("datetime")
        if isinstance(iso, str) and iso:
            try:
                parsed = datetime.fromisoformat(iso.replace("Z", "+00:00"))
            except ValueError:
                parsed = None
            if parsed is not None:
                return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        return self.now()

    @staticmethod
    def _extract_fee(response: dict[str, Any]) -> tuple[float, str | None]:
        fee = response.get("fee")
        if isinstance(fee, dict):
            cost = _to_float(fee.get("cost"))
            if cost is not None:
                return abs(cost), fee.get("currency")
        fees = response.get("fees")
        if isinstance(fees, list) and fees:
            total = 0.0
            currency: str | None = None
            for entry in fees:
                if not isinstance(entry, dict):
                    continue
                cost = _to_float(entry.get("cost"))
                if cost is None:
                    continue
                total += abs(cost)
                currency = currency or entry.get("currency")
            return total, currency
        return 0.0, None

    def _track_fill(self, order: Order) -> None:
        """Suivi local du prix d'entrée (spot long) à partir des exécutions de ce broker
        (toute quantité réellement exécutée compte, même sur un ordre partiel)."""
        if order.amount <= 0:
            return
        delta = order.amount if order.side == "buy" else -order.amount
        current = self._tracked.get(order.symbol, Position(order.symbol))
        updated = apply_fill(current, delta, order.price, order.timestamp)
        if updated.is_open and updated.amount > 0:
            self._tracked[order.symbol] = updated
        else:
            self._tracked.pop(order.symbol, None)

    def __repr__(self) -> str:
        return f"CcxtBroker(exchange_id={self.exchange_id!r}, sandbox={self.sandbox})"
