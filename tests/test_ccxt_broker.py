"""Tests du CcxtBroker avec un faux exchange (aucun réseau) : garde confirm_live, appels
effectués, arrondi à la précision, minimums du marché, mapping des réponses (ordres
rejetés / partiels), poussière invendable, prix exigé pour les achats au marché, erreurs
ccxt encapsulées avec masquage des secrets, jamais de retrait ni de levier."""

from __future__ import annotations

import copy
import logging
import math
from datetime import UTC, datetime
from decimal import ROUND_DOWN, Decimal
from typing import Any

import ccxt
import pytest

from tradingbot.execution.broker import BrokerError, Order, Position
from tradingbot.execution.ccxt_broker import CcxtBroker, scrub_secrets, split_symbol

BTC = "BTC/USDT"
FORBIDDEN_CALLS = {"withdraw", "set_leverage", "set_margin_mode", "transfer", "borrow_margin"}


def default_markets() -> dict[str, Any]:
    return {
        BTC: {
            "symbol": BTC,
            "base": "BTC",
            "quote": "USDT",
            "precision": {"amount": 5, "price": 2},
            "limits": {"amount": {"min": 0.00001, "max": 9000}, "cost": {"min": 5.0, "max": None}},
        },
        "ETH/USDT": {
            "symbol": "ETH/USDT",
            "base": "ETH",
            "quote": "USDT",
            "precision": {"amount": 4, "price": 2},
            "limits": {"amount": {"min": None, "max": None}, "cost": {"min": None, "max": None}},
        },
    }


def default_balance(usdt_free: float = 1000.0, btc_total: float = 0.5) -> dict[str, Any]:
    return {
        "free": {"USDT": usdt_free, "BTC": btc_total},
        "used": {"USDT": 0.0, "BTC": 0.0},
        "total": {"USDT": usdt_free, "BTC": btc_total},
        "USDT": {"free": usdt_free, "used": 0.0, "total": usdt_free},
        "BTC": {"free": btc_total, "used": 0.0, "total": btc_total},
        "info": {},
    }


def default_order_response(**overrides: Any) -> dict[str, Any]:
    resp: dict[str, Any] = {
        "id": "123",
        "symbol": BTC,
        "type": "market",
        "side": "buy",
        "amount": 0.12345,
        "filled": 0.12345,
        "remaining": 0.0,
        "average": 20010.5,
        "price": None,
        "cost": 0.12345 * 20010.5,
        "fee": {"cost": 2.47, "currency": "USDT"},
        "fees": [{"cost": 2.47, "currency": "USDT"}],
        "status": "closed",
        "timestamp": 1_700_000_000_000,
        "datetime": "2023-11-14T22:13:20.000Z",
        "info": {},
    }
    resp.update(overrides)
    return resp


class FakeExchange:
    """Imite la surface ccxt utilisée par le broker et enregistre chaque appel."""

    def __init__(
        self,
        markets: dict[str, Any] | None = None,
        ticker: dict[str, Any] | None = None,
        balance: dict[str, Any] | None = None,
        order_responses: list[dict[str, Any]] | None = None,
        errors: dict[str, Exception] | None = None,
    ) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.markets_data = markets if markets is not None else default_markets()
        self.ticker = ticker if ticker is not None else {"last": 20_000.0, "close": 19_990.0}
        self.balance = balance if balance is not None else default_balance()
        self.order_responses = list(order_responses or [default_order_response()])
        self.errors = dict(errors or {})
        self.sandbox: bool | None = None

    # -- utilitaires de test -------------------------------------------------
    def _record(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append((name, args, kwargs))
        if name in self.errors:
            raise self.errors[name]

    def count(self, name: str) -> int:
        return sum(1 for n, _, _ in self.calls if n == name)

    def args_of(self, name: str) -> list[tuple[Any, ...]]:
        return [a for n, a, _ in self.calls if n == name]

    # -- surface ccxt -----------------------------------------------------------
    def load_markets(self) -> dict[str, Any]:
        self._record("load_markets")
        return copy.deepcopy(self.markets_data)

    def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        self._record("fetch_ticker", symbol)
        return dict(self.ticker)

    def fetch_balance(self) -> dict[str, Any]:
        self._record("fetch_balance")
        return copy.deepcopy(self.balance)

    def amount_to_precision(self, symbol: str, amount: float) -> str:
        self._record("amount_to_precision", symbol, amount)
        decimals = self.markets_data[symbol]["precision"]["amount"]
        quantum = Decimal(1).scaleb(-decimals)
        return str(Decimal(repr(amount)).quantize(quantum, rounding=ROUND_DOWN))

    def create_order(
        self,
        symbol: str,
        type: str,  # noqa: A002 - nom imposé par ccxt
        side: str,
        amount: float,
        price: float | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._record("create_order", symbol, type, side, amount, price, params)
        resp = (
            self.order_responses.pop(0)
            if len(self.order_responses) > 1
            else self.order_responses[0]
        )
        return dict(resp)

    def set_sandbox_mode(self, enabled: bool) -> None:
        self._record("set_sandbox_mode", enabled)
        self.sandbox = enabled

    def withdraw(self, *args: Any, **kwargs: Any) -> None:
        self._record("withdraw", *args, **kwargs)
        raise AssertionError("withdraw ne doit jamais être appelé")

    def set_leverage(self, *args: Any, **kwargs: Any) -> None:
        self._record("set_leverage", *args, **kwargs)
        raise AssertionError("set_leverage ne doit jamais être appelé")

    def set_margin_mode(self, *args: Any, **kwargs: Any) -> None:
        self._record("set_margin_mode", *args, **kwargs)
        raise AssertionError("set_margin_mode ne doit jamais être appelé")


def make_broker(fake: FakeExchange | None = None, **kwargs: Any) -> tuple[CcxtBroker, FakeExchange]:
    fake = fake if fake is not None else FakeExchange()
    params = {"sandbox": False, "confirm_live": True, "exchange": fake}
    params.update(kwargs)
    return CcxtBroker("binance", "api-key", "api-secret", **params), fake


# ============================================================================ construction
def test_confirm_live_guard() -> None:
    fake = FakeExchange()
    with pytest.raises(BrokerError, match="confirm_live"):
        CcxtBroker("binance", "k", "s", exchange=fake)
    with pytest.raises(BrokerError, match="confirm_live"):
        CcxtBroker("binance", "k", "s", confirm_live=False, sandbox=True, exchange=fake)
    assert fake.calls == []


def test_injected_exchange_skips_ccxt_and_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_: Any, **__: Any) -> None:
        raise AssertionError("ccxt.binance ne doit pas être instancié quand exchange est injecté")

    monkeypatch.setattr(ccxt, "binance", boom)
    broker, fake = make_broker()
    assert broker.exchange is fake
    assert fake.calls == [], "aucun appel réseau à la construction (marchés chargés paresseusement)"
    assert "binance" in repr(broker)


def test_builds_ccxt_exchange_with_expected_config(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[dict[str, Any]] = []

    class FakeBinance(FakeExchange):
        def __init__(self, config: dict[str, Any]) -> None:
            super().__init__()
            created.append(config)

    monkeypatch.setattr(ccxt, "binance", FakeBinance)
    broker = CcxtBroker("binance", "my-key", "my-secret", confirm_live=True)
    assert created == [{"apiKey": "my-key", "secret": "my-secret", "enableRateLimit": True}]
    assert isinstance(broker.exchange, FakeBinance)
    assert broker.exchange.count("set_sandbox_mode") == 0
    assert broker.exchange.sandbox is None

    broker_sb = CcxtBroker("binance", "my-key", "my-secret", sandbox=True, confirm_live=True)
    assert broker_sb.exchange.args_of("set_sandbox_mode") == [(True,)]
    assert broker_sb.exchange.sandbox is True
    assert broker_sb.sandbox is True


def test_unknown_exchange_id_rejected() -> None:
    with pytest.raises(BrokerError, match="Exchange inconnu"):
        CcxtBroker("pas-un-exchange", "k", "s", confirm_live=True)


def test_sandbox_applied_to_injected_exchange() -> None:
    _, fake = make_broker(sandbox=True)
    assert fake.args_of("set_sandbox_mode") == [(True,)]
    assert fake.sandbox is True


def test_sandbox_without_support_rejected() -> None:
    class NoSandbox:
        pass

    with pytest.raises(BrokerError, match="sandbox"):
        CcxtBroker("binance", "k", "s", sandbox=True, confirm_live=True, exchange=NoSandbox())


def test_markets_loaded_lazily_and_cached() -> None:
    broker, fake = make_broker()
    assert fake.count("load_markets") == 0
    broker.get_position(BTC)
    broker.get_position(BTC)
    broker.market_order(BTC, "buy", 0.01)
    assert fake.count("load_markets") == 1


# ============================================================================ lecture
def test_get_price_uses_last() -> None:
    broker, fake = make_broker()
    assert broker.get_price(BTC) == 20_000.0
    assert fake.args_of("fetch_ticker") == [(BTC,)]


def test_get_price_falls_back_to_close() -> None:
    broker, _ = make_broker(FakeExchange(ticker={"last": None, "close": 19_990.0}))
    assert broker.get_price(BTC) == 19_990.0


def test_get_price_missing_raises() -> None:
    broker, _ = make_broker(FakeExchange(ticker={"last": None, "close": None, "bid": 1.0}))
    with pytest.raises(BrokerError, match="Prix indisponible"):
        broker.get_price(BTC)


def test_get_balance_cash_and_equity() -> None:
    broker, fake = make_broker()
    bal = broker.get_balance(BTC)
    assert bal.cash == 1000.0
    assert bal.equity == pytest.approx(1000.0 + 0.5 * 20_000.0)  # 11 000
    assert fake.count("fetch_balance") == 1
    assert fake.count("fetch_ticker") == 1


def test_get_balance_without_base_skips_ticker() -> None:
    broker, fake = make_broker(FakeExchange(balance=default_balance(btc_total=0.0)))
    bal = broker.get_balance(BTC)
    assert bal.cash == 1000.0 and bal.equity == 1000.0
    assert fake.count("fetch_ticker") == 0


def test_get_balance_missing_currency_is_zero() -> None:
    broker, _ = make_broker(FakeExchange(balance={"free": {}, "total": {}, "info": {}}))
    bal = broker.get_balance(BTC)
    assert bal.cash == 0.0 and bal.equity == 0.0


def test_get_balance_uses_per_currency_section_as_fallback() -> None:
    broker, _ = make_broker(
        FakeExchange(balance={"USDT": {"free": 42.0, "total": 42.0}, "BTC": {"total": 0.0}})
    )
    assert broker.get_balance(BTC).cash == 42.0


def test_get_position_from_balance_without_tracking() -> None:
    broker, _ = make_broker()
    pos = broker.get_position(BTC)
    assert pos == Position(BTC, 0.5, 0.0, None)
    assert pos.is_open and pos.side == 1


def test_get_position_dust_is_flat() -> None:
    broker, fake = make_broker(FakeExchange(balance=default_balance(btc_total=0.000001)))
    pos = broker.get_position(BTC)
    assert pos == Position(BTC)
    assert not pos.is_open
    assert fake.count("fetch_ticker") == 0  # sous amount.min : pas besoin du prix


def test_get_position_dust_under_min_cost_is_flat() -> None:
    """F9 : 0,00003 BTC >= amount.min (0,00001) mais 0,60 USDT < cost.min (5) : invendable,
    donc plat (sinon le bot « tient » une position qu'il ne peut jamais clôturer)."""
    broker, fake = make_broker(FakeExchange(balance=default_balance(btc_total=0.00003)))
    pos = broker.get_position(BTC)
    assert pos == Position(BTC)
    assert fake.count("fetch_ticker") == 1
    # juste au-dessus du minimum de notionnel : position réelle (0,0003 × 20 000 = 6 USDT)
    fake.balance = default_balance(btc_total=0.0003)
    assert broker.get_position(BTC).amount == pytest.approx(0.0003)
    # sans cost.min : amount.min seul décide, et le ticker n'est pas consulté
    fake2 = FakeExchange(balance=default_balance(btc_total=0.00003))
    fake2.markets_data[BTC]["limits"]["cost"]["min"] = None
    broker2, _ = make_broker(fake2)
    assert broker2.get_position(BTC).amount == pytest.approx(0.00003)
    assert fake2.count("fetch_ticker") == 0


def test_get_position_never_short_on_spot() -> None:
    broker, _ = make_broker(FakeExchange(balance=default_balance(btc_total=-0.3)))
    assert broker.get_position(BTC).amount == 0.0


def test_unknown_symbol_rejected() -> None:
    broker, fake = make_broker()
    with pytest.raises(BrokerError, match="inconnu"):
        broker.market_order("DOGE/USDT", "buy", 1)
    assert fake.count("create_order") == 0


def test_split_symbol() -> None:
    assert split_symbol("BTC/USDT") == ("BTC", "USDT")
    assert split_symbol("BTC/USDT:USDT") == ("BTC", "USDT")
    for bad in ("BTCUSDT", "/USDT", "BTC/"):
        with pytest.raises(BrokerError, match="Symbole invalide"):
            split_symbol(bad)


# ============================================================================ ordres
def test_market_order_rounds_amount_and_maps_response() -> None:
    broker, fake = make_broker()
    order = broker.market_order(BTC, "buy", 0.123456789)

    assert fake.args_of("amount_to_precision") == [(BTC, 0.123456789)]
    assert fake.args_of("create_order") == [(BTC, "market", "buy", 0.12345, None, None)]
    sent = fake.args_of("create_order")[0][3]
    assert isinstance(sent, float)

    assert isinstance(order, Order)
    assert order.id == "123"
    assert order.symbol == BTC
    assert order.side == "buy"
    assert order.amount == 0.12345
    assert order.price == 20010.5
    assert order.fee == 2.47
    assert order.cost == pytest.approx(0.12345 * 20010.5)
    assert order.status == "filled"
    assert order.timestamp == datetime(2023, 11, 14, 22, 13, 20, tzinfo=UTC)
    assert order.info["requested_amount"] == 0.123456789
    assert order.info["sent_amount"] == 0.12345
    assert order.info["fee_currency"] == "USDT"
    assert order.info["raw"]["id"] == "123"


def test_precision_truncates_instead_of_rounding() -> None:
    broker, fake = make_broker()
    broker.market_order(BTC, "buy", 0.123459999)
    assert fake.args_of("create_order")[0][3] == 0.12345


def test_min_cost_check_uses_ticker_before_sending() -> None:
    broker, fake = make_broker()
    broker.market_order(BTC, "buy", 0.01)  # 0,01 × 20 000 = 200 >= 5
    names = [n for n, _, _ in fake.calls]
    assert names.index("fetch_ticker") < names.index("create_order")


def test_min_amount_rejection() -> None:
    fake = FakeExchange()
    fake.markets_data[BTC]["limits"]["amount"]["min"] = 0.001
    broker, _ = make_broker(fake)
    # 0,0005 passe l'arrondi (0,00050) mais reste sous le minimum 0,001
    with pytest.raises(BrokerError, match="0.0005 BTC inférieur au minimum"):
        broker.market_order(BTC, "buy", 0.0005)
    assert fake.count("create_order") == 0
    assert fake.count("fetch_ticker") == 0  # refusé avant même de consulter le prix
    # juste au-dessus du minimum : accepté
    broker.market_order(BTC, "buy", 0.001)
    assert fake.args_of("create_order") == [(BTC, "market", "buy", 0.001, None, None)]


def test_min_cost_rejection() -> None:
    broker, fake = make_broker()
    # 0,0001 × 20 000 = 2 USDT < 5 USDT
    with pytest.raises(BrokerError, match="Notionnel 2.00 USDT inférieur au minimum"):
        broker.market_order(BTC, "buy", 0.0001)
    assert fake.count("create_order") == 0
    assert fake.count("fetch_ticker") == 1


def test_no_limits_means_no_ticker_call() -> None:
    broker, fake = make_broker(
        FakeExchange(order_responses=[default_order_response(symbol="ETH/USDT", average=1500.0)])
    )
    order = broker.market_order("ETH/USDT", "buy", 0.12345678)
    assert fake.args_of("create_order") == [("ETH/USDT", "market", "buy", 0.1234, None, None)]
    assert fake.count("fetch_ticker") == 0
    assert order.price == 1500.0


def test_amount_rounding_to_zero_rejected() -> None:
    fake = FakeExchange()
    fake.markets_data[BTC]["limits"]["amount"]["min"] = None
    broker, _ = make_broker(fake)
    with pytest.raises(BrokerError, match="nul après arrondi"):
        broker.market_order(BTC, "buy", 0.000001)
    assert fake.count("create_order") == 0


@pytest.mark.parametrize("bad_amount", [0, -1, math.nan, math.inf, "1", None])
def test_invalid_amount_rejected_before_any_call(bad_amount: object) -> None:
    broker, fake = make_broker()
    with pytest.raises(BrokerError, match="Montant d'ordre invalide"):
        broker.market_order(BTC, "buy", bad_amount)  # type: ignore[arg-type]
    assert fake.calls == []


def test_invalid_side_rejected_before_any_call() -> None:
    broker, fake = make_broker()
    with pytest.raises(BrokerError, match="Sens d'ordre invalide"):
        broker.market_order(BTC, "long", 1)  # type: ignore[arg-type]
    assert fake.calls == []


def test_sell_order_sent_with_sell_side() -> None:
    broker, fake = make_broker(
        FakeExchange(order_responses=[default_order_response(side="sell", average=19_900.0)])
    )
    order = broker.market_order(BTC, "sell", 0.25)
    assert fake.args_of("create_order") == [(BTC, "market", "sell", 0.25, None, None)]
    assert order.side == "sell" and order.price == 19_900.0


def test_market_buy_passes_reference_price_when_exchange_requires_it() -> None:
    """F27 : Coinbase, Gate... exigent un prix pour un achat au marché (ccxt convertit la
    quantité en montant à dépenser) ; sans ce prix ccxt refuse l'ordre avant tout appel."""
    fake = FakeExchange(
        order_responses=[default_order_response(), default_order_response(side="sell")]
    )
    fake.options = {"createMarketBuyOrderRequiresPrice": True}
    broker, _ = make_broker(fake)
    broker.market_order(BTC, "buy", 0.01)
    assert fake.args_of("create_order") == [(BTC, "market", "buy", 0.01, 20_000.0, None)]
    assert fake.count("fetch_ticker") == 1  # le prix du min cost est réutilisé
    broker.market_order(BTC, "sell", 0.01)  # une vente n'a pas besoin de prix
    assert fake.args_of("create_order")[1] == (BTC, "market", "sell", 0.01, None, None)


def test_market_buy_fetches_price_when_required_and_no_min_cost() -> None:
    fake = FakeExchange(order_responses=[default_order_response(symbol="ETH/USDT", average=1500.0)])
    fake.options = {"createMarketBuyOrderRequiresPrice": True}
    fake.ticker = {"last": 1500.0}
    broker, _ = make_broker(fake)
    broker.market_order("ETH/USDT", "buy", 0.1)  # ETH/USDT n'a pas de cost.min
    assert fake.args_of("create_order") == [("ETH/USDT", "market", "buy", 0.1, 1500.0, None)]
    assert fake.count("fetch_ticker") == 1


def test_market_buy_without_price_when_option_false_or_absent() -> None:
    fake = FakeExchange()
    fake.options = {"createMarketBuyOrderRequiresPrice": False}
    broker, _ = make_broker(fake)
    broker.market_order(BTC, "buy", 0.01)
    assert fake.args_of("create_order") == [(BTC, "market", "buy", 0.01, None, None)]


# ============================================================================ mapping de la réponse
def test_fee_in_base_currency_converted_to_quote() -> None:
    resp = default_order_response(fee={"cost": 0.0001, "currency": "BTC"}, average=20_000.0)
    broker, _ = make_broker(FakeExchange(order_responses=[resp]))
    order = broker.market_order(BTC, "buy", 0.1)
    assert order.fee == pytest.approx(2.0)  # 0,0001 BTC × 20 000
    assert order.info["fee_currency"] == "BTC"


def test_fee_in_third_currency_is_not_counted() -> None:
    resp = default_order_response(fee={"cost": 0.01, "currency": "BNB"})
    broker, _ = make_broker(FakeExchange(order_responses=[resp]))
    order = broker.market_order(BTC, "buy", 0.1)
    assert order.fee == 0.0
    assert order.info["fee_currency"] == "BNB"


def test_fees_list_used_when_fee_missing() -> None:
    resp = default_order_response(
        fee=None, fees=[{"cost": 1.0, "currency": "USDT"}, {"cost": 0.5, "currency": "USDT"}]
    )
    broker, _ = make_broker(FakeExchange(order_responses=[resp]))
    assert broker.market_order(BTC, "buy", 0.1).fee == pytest.approx(1.5)


def test_no_fee_information_means_zero() -> None:
    resp = default_order_response(fee=None, fees=[])
    broker, _ = make_broker(FakeExchange(order_responses=[resp]))
    order = broker.market_order(BTC, "buy", 0.1)
    assert order.fee == 0.0 and order.info["fee_currency"] is None


def test_price_fallback_to_price_then_cost_then_ticker() -> None:
    r_price = default_order_response(average=None, price=19_999.0)
    r_cost = default_order_response(average=None, price=None, cost=2000.0, filled=0.1)
    r_none = default_order_response(average=None, price=None, cost=None)
    broker, fake = make_broker(FakeExchange(order_responses=[r_price, r_cost, r_none]))
    assert broker.market_order(BTC, "buy", 0.1).price == 19_999.0
    assert broker.market_order(BTC, "buy", 0.1).price == pytest.approx(20_000.0)
    order = broker.market_order(BTC, "buy", 0.1)
    assert order.price == 20_000.0  # dernier recours : le ticker récupéré pour le min cost
    assert fake.count("fetch_ticker") == 3


def test_missing_filled_assumes_sent_amount() -> None:
    resp = default_order_response(filled=None)
    broker, _ = make_broker(FakeExchange(order_responses=[resp]))
    order = broker.market_order(BTC, "buy", 0.2)
    assert order.amount == 0.2


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("closed", "filled"),
        ("open", "open"),
        ("canceled", "partial"),  # annulé APRÈS une exécution partielle (filled 0,12345)
        (None, "filled"),
        ("ZZZ", "filled"),
    ],
)
def test_status_mapping(raw: str | None, expected: str) -> None:
    broker, _ = make_broker(FakeExchange(order_responses=[default_order_response(status=raw)]))
    order = broker.market_order(BTC, "buy", 0.1)
    assert order.status == expected
    assert order.info["exchange_status"] == raw


@pytest.mark.parametrize("raw", ["canceled", "cancelled", "rejected", "expired"])
@pytest.mark.parametrize("filled", [0.0, None])
def test_rejected_order_without_fill_raises(raw: str, filled: float | None) -> None:
    """F4 : un ordre annulé / rejeté sans exécution n'est pas un ordre exécuté."""
    resp = default_order_response(status=raw, filled=filled, average=None, cost=None)
    broker, fake = make_broker(FakeExchange(order_responses=[resp]))
    with pytest.raises(BrokerError, match="rejeté par l'exchange") as exc_info:
        broker.market_order(BTC, "buy", 0.1)
    assert raw in str(exc_info.value) and "123" in str(exc_info.value)
    assert fake.count("create_order") == 1
    # rien n'est suivi localement
    fake.balance = default_balance(btc_total=0.1)
    assert broker.get_position(BTC).entry_price == 0.0


def test_partially_filled_then_canceled_order_reports_executed_amount(
    caplog: pytest.LogCaptureFixture,
) -> None:
    resp = default_order_response(status="canceled", filled=0.04, average=20_000.0)
    broker, fake = make_broker(FakeExchange(order_responses=[resp]))
    with caplog.at_level(logging.WARNING, logger="tradingbot.execution.ccxt_broker"):
        order = broker.market_order(BTC, "buy", 0.1)
    assert order.status == "partial"
    assert order.amount == 0.04
    assert order.info["sent_amount"] == 0.1
    assert "exécuté partiellement" in caplog.text
    # la quantité réellement exécutée est suivie (prix d'entrée)
    fake.balance = default_balance(btc_total=0.04)
    pos = broker.get_position(BTC)
    assert pos.amount == 0.04 and pos.entry_price == pytest.approx(20_000.0)


def test_closed_order_with_smaller_fill_keeps_actual_amount() -> None:
    resp = default_order_response(status="closed", filled=0.07)
    broker, _ = make_broker(FakeExchange(order_responses=[resp]))
    order = broker.market_order(BTC, "buy", 0.1)
    assert order.status == "filled" and order.amount == 0.07


def test_timestamp_fallbacks() -> None:
    r_iso = default_order_response(timestamp=None, datetime="2024-01-02T03:04:05Z")
    r_none = default_order_response(timestamp=None, datetime=None)
    broker, _ = make_broker(FakeExchange(order_responses=[r_iso, r_none]))
    before = datetime.now(UTC)
    assert broker.market_order(BTC, "buy", 0.1).timestamp == datetime(
        2024, 1, 2, 3, 4, 5, tzinfo=UTC
    )
    ts = broker.market_order(BTC, "buy", 0.1).timestamp
    assert ts.tzinfo is not None and before <= ts <= datetime.now(UTC)


def test_missing_id_gets_generated() -> None:
    broker, _ = make_broker(FakeExchange(order_responses=[default_order_response(id=None)]))
    assert broker.market_order(BTC, "buy", 0.1).id == "ccxt-1700000000000"


# ============================================================================ suivi local du prix d'entrée
def test_entry_price_tracked_from_fills() -> None:
    r1 = default_order_response(id="a", filled=0.1, average=20_000.0, timestamp=1_700_000_000_000)
    r2 = default_order_response(id="b", filled=0.1, average=21_000.0, timestamp=1_700_000_060_000)
    r3 = default_order_response(id="c", filled=0.05, average=25_000.0, side="sell")
    r4 = default_order_response(id="d", filled=0.15, average=25_000.0, side="sell")
    fake = FakeExchange(order_responses=[r1, r2, r3, r4], balance=default_balance(btc_total=0.2))
    broker, _ = make_broker(fake)

    broker.market_order(BTC, "buy", 0.1)
    pos = broker.get_position(BTC)
    assert pos.amount == 0.2  # quantité : solde de l'exchange (source de vérité)
    assert pos.entry_price == pytest.approx(20_000.0)
    assert pos.opened_at == datetime.fromtimestamp(1_700_000_000, tz=UTC)

    broker.market_order(BTC, "buy", 0.1)
    assert broker.get_position(BTC).entry_price == pytest.approx(20_500.0)  # VWAP
    assert broker.get_position(BTC).opened_at == datetime.fromtimestamp(1_700_000_000, tz=UTC)

    broker.market_order(BTC, "sell", 0.05)
    assert broker.get_position(BTC).entry_price == pytest.approx(20_500.0)  # réduction : inchangé

    broker.market_order(BTC, "sell", 0.15)  # suivi local plat → prix d'entrée inconnu
    fake.balance = default_balance(btc_total=0.0)
    assert broker.get_position(BTC) == Position(BTC)


def test_tracking_reset_when_exchange_balance_is_flat() -> None:
    fake = FakeExchange(balance=default_balance(btc_total=0.0))
    broker, _ = make_broker(fake)
    broker.market_order(BTC, "buy", 0.1)
    assert broker.get_position(BTC) == Position(BTC)
    fake.balance = default_balance(btc_total=0.1)
    assert broker.get_position(BTC).entry_price == 0.0  # le suivi a été oublié


# ============================================================================ erreurs ccxt
def test_ccxt_errors_are_wrapped_with_cause() -> None:
    cause = ccxt.InsufficientFunds("binance Account has insufficient balance")
    broker, fake = make_broker(FakeExchange(errors={"create_order": cause}))
    with pytest.raises(BrokerError, match="InsufficientFunds") as exc_info:
        broker.market_order(BTC, "buy", 0.1)
    assert exc_info.value.__cause__ is cause
    assert "Erreur de l'exchange" in str(exc_info.value)
    assert fake.count("create_order") == 1


def test_network_error_on_ticker_wrapped() -> None:
    broker, _ = make_broker(FakeExchange(errors={"fetch_ticker": ccxt.NetworkError("timeout")}))
    with pytest.raises(BrokerError, match="prix de BTC/USDT"):
        broker.get_price(BTC)
    with pytest.raises(BrokerError):
        broker.get_balance(BTC)


def test_load_markets_error_wrapped() -> None:
    broker, _ = make_broker(
        FakeExchange(errors={"load_markets": ccxt.ExchangeError("maintenance")})
    )
    with pytest.raises(BrokerError, match="chargement des marchés"):
        broker.market_order(BTC, "buy", 0.1)
    with pytest.raises(BrokerError):
        broker.get_position(BTC)


def test_api_key_scrubbed_from_error_messages(caplog: pytest.LogCaptureFixture) -> None:
    """F28 : ccxt recopie l'URL signée (clé en paramètre sur htx, upbit...) dans ses erreurs."""
    key, secret = "LEAKME_APIKEY_0123456789", "U1VQRVJTRUNSRVQ="
    cause = ccxt.ExchangeNotAvailable(
        f"htx GET https://api.huobi.pro/v1/account/accounts?AccessKeyId={key}"
        f"&Signature=abc%3D&Timestamp=2024 503 <html>cloudflare</html> secret={secret}"
    )
    fake = FakeExchange(errors={"fetch_balance": cause})
    broker = CcxtBroker("htx", key, secret, confirm_live=True, exchange=fake)
    with pytest.raises(BrokerError) as exc_info:
        broker.get_balance(BTC)
    text = str(exc_info.value)
    assert key not in text and secret not in text
    assert "***" in text and "ExchangeNotAvailable" in text and "api.huobi.pro" in text
    # l'exception d'origine (chaînée, visible au niveau debug) est elle aussi masquée
    assert exc_info.value.__cause__ is cause
    assert key not in str(cause) and secret not in str(cause)
    assert key not in repr(broker)


def test_scrub_secrets_helper() -> None:
    text = (
        "GET https://x.test/a?apiKey=K1&sign=S1&ts=5 then https://y.test/b?foo=bar key=K1 "
        "plain K2 and {'msg': 'ok'}"
    )
    out = scrub_secrets(text, ["K1", "K2", "", None])
    assert "K1" not in out and "K2" not in out
    assert out.startswith("GET https://x.test/a?*** then https://y.test/b?*** ")
    assert "{'msg': 'ok'}" in out
    assert scrub_secrets("rien à cacher", []) == "rien à cacher"
    assert scrub_secrets("?access_key=abc&Signature=def", []) == "?access_key=***&Signature=***"


def test_balance_error_wrapped() -> None:
    broker, _ = make_broker(
        FakeExchange(errors={"fetch_balance": ccxt.AuthenticationError("bad key")})
    )
    with pytest.raises(BrokerError, match="AuthenticationError"):
        broker.get_balance(BTC)


def test_sandbox_error_wrapped() -> None:
    fake = FakeExchange(errors={"set_sandbox_mode": ccxt.NotSupported("no sandbox")})
    with pytest.raises(BrokerError, match="sandbox"):
        make_broker(fake, sandbox=True)


def test_broker_error_from_exchange_passes_through_unchanged() -> None:
    inner = BrokerError("déjà en français")
    broker, _ = make_broker(FakeExchange(errors={"fetch_ticker": inner}))
    with pytest.raises(BrokerError) as exc_info:
        broker.get_price(BTC)
    assert exc_info.value is inner


# ============================================================================ close_position (contrat)
def test_close_position_sells_whole_balance() -> None:
    resp = default_order_response(side="sell", filled=0.5, average=20_000.0)
    broker, fake = make_broker(FakeExchange(order_responses=[resp]))
    order = broker.close_position(BTC)
    assert order is not None
    assert fake.args_of("create_order") == [(BTC, "market", "sell", 0.5, None, None)]
    assert order.side == "sell" and order.amount == 0.5


def test_close_position_when_flat_sends_nothing() -> None:
    broker, fake = make_broker(FakeExchange(balance=default_balance(btc_total=0.0)))
    assert broker.close_position(BTC) is None
    assert fake.count("create_order") == 0


# ============================================================================ sécurité
def test_never_withdraws_nor_sets_leverage() -> None:
    broker, fake = make_broker(sandbox=True)
    broker.get_price(BTC)
    broker.get_balance(BTC)
    broker.get_position(BTC)
    broker.market_order(BTC, "buy", 0.01)
    broker.close_position(BTC)
    called = {name for name, _, _ in fake.calls}
    assert not (called & FORBIDDEN_CALLS)
    assert called == {
        "set_sandbox_mode",
        "fetch_ticker",
        "fetch_balance",
        "load_markets",
        "amount_to_precision",
        "create_order",
    }
    # uniquement des ordres au marché, jamais de params (levier, marge, etc.)
    assert [a[1] for a in fake.args_of("create_order")] == ["market", "market"]
    assert all(a[5] is None for a in fake.args_of("create_order"))


def test_api_secrets_not_exposed_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeBinance(FakeExchange):
        def __init__(self, config: dict[str, Any]) -> None:
            super().__init__()

    monkeypatch.setattr(ccxt, "binance", FakeBinance)
    broker = CcxtBroker("binance", "SECRET-KEY", "SECRET-SECRET", confirm_live=True)
    assert "SECRET" not in repr(broker)
    assert not hasattr(broker, "api_secret")
