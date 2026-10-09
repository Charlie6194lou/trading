"""Tests du PaperBroker : cash / frais / glissement calculés à la main, refus, shorts,
prix moyen d'entrée, persistance JSON, close_position hérité du contrat."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tradingbot.execution.broker import Balance, BrokerError, Order, Position
from tradingbot.execution.paper import PaperBroker, apply_fill, check_order_args

BTC = "BTC/USDT"
ETH = "ETH/USDT"
T0 = datetime(2024, 3, 10, 12, 0, tzinfo=UTC)


@pytest.fixture
def fixed_clock(monkeypatch: pytest.MonkeyPatch) -> list[datetime]:
    """Horloge contrôlée : la liste contient l'instant courant (modifiable par le test)."""
    clock = [T0]
    monkeypatch.setattr(PaperBroker, "now", staticmethod(lambda: clock[0]))
    return clock


@pytest.fixture
def broker() -> PaperBroker:
    """10 000 USDT, frais 0,1 %, glissement 0,05 %, BTC à 100."""
    b = PaperBroker(initial_cash=10_000.0, fee_rate=0.001, slippage=0.0005)
    b.set_price(BTC, 100.0)
    return b


@pytest.fixture
def short_broker() -> PaperBroker:
    b = PaperBroker(initial_cash=10_000.0, fee_rate=0.001, slippage=0.0005, allow_short=True)
    b.set_price(BTC, 100.0)
    return b


# ============================================================================ construction
def test_constructor_validates_arguments() -> None:
    with pytest.raises(ValueError, match="initial_cash"):
        PaperBroker(initial_cash=-1)
    with pytest.raises(ValueError, match="initial_cash"):
        PaperBroker(initial_cash=math.nan)
    with pytest.raises(ValueError, match="fee_rate"):
        PaperBroker(initial_cash=100, fee_rate=-0.1)
    with pytest.raises(ValueError, match="slippage"):
        PaperBroker(initial_cash=100, slippage=1.0)


def test_initial_state_is_flat(broker: PaperBroker) -> None:
    assert broker.cash == 10_000.0
    assert broker.orders == []
    assert broker.get_position(BTC) == Position(BTC)
    assert not broker.get_position(BTC).is_open
    assert broker.get_balance(BTC) == Balance(cash=10_000.0, equity=10_000.0)
    assert "PaperBroker" in repr(broker)


# ============================================================================ prix
def test_set_price_and_get_price(broker: PaperBroker) -> None:
    assert broker.get_price(BTC) == 100.0
    broker.set_price(BTC, 101.5)
    assert broker.get_price(BTC) == 101.5


def test_unknown_price_raises_french_error() -> None:
    b = PaperBroker(initial_cash=100)
    with pytest.raises(BrokerError, match="Aucun prix connu"):
        b.get_price(BTC)
    with pytest.raises(BrokerError):
        b.market_order(BTC, "buy", 1)


@pytest.mark.parametrize("bad", [0, -5, math.nan, math.inf, "abc", None])
def test_invalid_prices_rejected(bad: object) -> None:
    b = PaperBroker(initial_cash=100)
    with pytest.raises(BrokerError, match="Prix invalide"):
        b.set_price(BTC, bad)  # type: ignore[arg-type]


def test_price_feed_takes_precedence_over_manual_prices() -> None:
    prices = {BTC: 250.0}
    b = PaperBroker(initial_cash=10_000, price_feed=lambda s: prices[s])
    b.set_price(BTC, 1.0)  # ignoré : le flux fait foi
    assert b.get_price(BTC) == 250.0
    prices[BTC] = 260.0
    assert b.get_price(BTC) == 260.0


def test_price_feed_failure_is_wrapped() -> None:
    def feed(symbol: str) -> float:
        raise KeyError(symbol)

    b = PaperBroker(initial_cash=10_000, price_feed=feed)
    with pytest.raises(BrokerError, match="flux de prix") as exc_info:
        b.get_price(BTC)
    assert isinstance(exc_info.value.__cause__, KeyError)


# ============================================================================ achats / ventes
def test_buy_applies_slippage_and_fee(broker: PaperBroker, fixed_clock: list[datetime]) -> None:
    order = broker.market_order(BTC, "buy", 10)
    # prix exécuté 100 × 1,0005 = 100,05 ; notionnel 1 000,5 ; frais 1,0005 ; coût 1 001,5005
    assert order.id == "paper-1"
    assert order.symbol == BTC
    assert order.side == "buy"
    assert order.amount == 10
    assert order.price == pytest.approx(100.05)
    assert order.fee == pytest.approx(1.0005)
    assert order.cost == pytest.approx(1000.5)
    assert order.status == "filled"
    assert order.timestamp == T0
    assert order.info["reference_price"] == 100.0
    assert order.info["cash_after"] == pytest.approx(8998.4995)
    assert broker.cash == pytest.approx(8998.4995)
    assert broker.orders == [order]

    pos = broker.get_position(BTC)
    assert pos.amount == pytest.approx(10)
    assert pos.entry_price == pytest.approx(100.05)
    assert pos.opened_at == T0
    assert pos.side == 1

    bal = broker.get_balance(BTC)
    assert bal.cash == pytest.approx(8998.4995)
    assert bal.equity == pytest.approx(8998.4995 + 10 * 100.0)  # 9 998,4995


def test_round_trip_costs_fees_and_slippage(broker: PaperBroker) -> None:
    broker.market_order(BTC, "buy", 10)
    sell = broker.market_order(BTC, "sell", 10)
    # vente à 100 × 0,9995 = 99,95 ; notionnel 999,5 ; frais 0,9995 ; encaissé 998,5005
    assert sell.id == "paper-2"
    assert sell.price == pytest.approx(99.95)
    assert sell.fee == pytest.approx(0.9995)
    assert broker.cash == pytest.approx(8998.4995 + 998.5005)  # 9 997,0
    assert broker.cash == pytest.approx(9997.0)
    assert not broker.get_position(BTC).is_open
    assert broker.get_position(BTC).entry_price == 0.0
    assert broker.get_position(BTC).opened_at is None
    assert broker.get_balance(BTC).equity == pytest.approx(9997.0)
    assert [o.id for o in broker.orders] == ["paper-1", "paper-2"]


def test_zero_fee_zero_slippage() -> None:
    b = PaperBroker(initial_cash=10_000, fee_rate=0.0, slippage=0.0)
    b.set_price(BTC, 100.0)
    b.market_order(BTC, "buy", 10)
    assert b.cash == pytest.approx(9000.0)
    assert b.get_balance(BTC).equity == pytest.approx(10_000.0)
    b.market_order(BTC, "sell", 10)
    assert b.cash == pytest.approx(10_000.0)


def test_equity_follows_price(broker: PaperBroker) -> None:
    broker.market_order(BTC, "buy", 10)
    broker.set_price(BTC, 120.0)
    assert broker.get_balance(BTC).equity == pytest.approx(8998.4995 + 1200.0)
    assert broker.get_position(BTC).unrealized_pnl(120.0) == pytest.approx((120 - 100.05) * 10)
    broker.set_price(BTC, 80.0)
    assert broker.get_balance(BTC).equity == pytest.approx(8998.4995 + 800.0)


def test_buy_rejected_when_cash_insufficient(broker: PaperBroker) -> None:
    # 100 × 100,05 = 10 005 + frais 10,005 = 10 015,005 > 10 000
    with pytest.raises(BrokerError, match="Fonds insuffisants"):
        broker.market_order(BTC, "buy", 100)
    # un refus ne change rien
    assert broker.cash == 10_000.0
    assert broker.orders == []
    assert not broker.get_position(BTC).is_open
    # le prochain ordre accepté reprend bien la numérotation à 1
    assert broker.market_order(BTC, "buy", 1).id == "paper-1"


def test_buy_exactly_affordable_within_tolerance() -> None:
    b = PaperBroker(initial_cash=1001.5005, fee_rate=0.001, slippage=0.0005)
    b.set_price(BTC, 100.0)
    b.market_order(BTC, "buy", 10)  # coût exact = 1 001,5005
    assert b.cash == pytest.approx(0.0, abs=1e-9)
    # un centime de plus et c'est refusé
    with pytest.raises(BrokerError):
        b.market_order(BTC, "buy", 0.001)


def test_sell_without_position_rejected(broker: PaperBroker) -> None:
    with pytest.raises(BrokerError, match="découvert"):
        broker.market_order(BTC, "sell", 1)
    assert broker.cash == 10_000.0
    assert broker.orders == []


def test_sell_more_than_held_rejected(broker: PaperBroker) -> None:
    broker.market_order(BTC, "buy", 10)
    with pytest.raises(BrokerError, match="supérieure à la position"):
        broker.market_order(BTC, "sell", 10.5)
    assert broker.get_position(BTC).amount == pytest.approx(10)
    assert len(broker.orders) == 1


def test_partial_sell_keeps_entry_price(broker: PaperBroker) -> None:
    broker.market_order(BTC, "buy", 10)
    broker.market_order(BTC, "sell", 4)
    pos = broker.get_position(BTC)
    assert pos.amount == pytest.approx(6)
    assert pos.entry_price == pytest.approx(100.05)
    # 8 998,4995 + 4 × 99,95 × (1 - 0,001) = 8 998,4995 + 399,4002
    assert broker.cash == pytest.approx(8998.4995 + 399.4002)


def test_vwap_entry_price_when_adding(broker: PaperBroker, fixed_clock: list[datetime]) -> None:
    broker.market_order(BTC, "buy", 10)  # 10 @ 100,05
    fixed_clock[0] = T0 + timedelta(hours=1)
    broker.set_price(BTC, 110.0)
    broker.market_order(BTC, "buy", 10)  # 10 @ 110,055 ; coût 1 100,55 + 1,10055
    pos = broker.get_position(BTC)
    assert pos.amount == pytest.approx(20)
    assert pos.entry_price == pytest.approx((1000.5 + 1100.55) / 20)  # 105,0525
    assert pos.entry_price == pytest.approx(105.0525)
    assert pos.opened_at == T0  # la date d'ouverture est celle de la première entrée
    assert broker.cash == pytest.approx(8998.4995 - 1101.65055)  # 7 896,84895

    broker.market_order(BTC, "sell", 5)  # 5 @ 109,945 ; encaissé 549,725 - 0,549725
    pos = broker.get_position(BTC)
    assert pos.amount == pytest.approx(15)
    assert pos.entry_price == pytest.approx(105.0525)
    assert broker.cash == pytest.approx(7896.84895 + 549.175275)  # 8 446,024225


def test_float_dust_is_snapped_to_flat(broker: PaperBroker) -> None:
    broker.market_order(BTC, "buy", 0.1)
    broker.market_order(BTC, "buy", 0.2)
    broker.market_order(BTC, "sell", 0.3)  # 0,1 + 0,2 - 0,3 != 0 en flottant
    pos = broker.get_position(BTC)
    assert pos.amount == 0.0
    assert not pos.is_open
    assert pos.entry_price == 0.0


@pytest.mark.parametrize("bad_amount", [0, -1, math.nan, math.inf, "3", None, True])
def test_invalid_amount_rejected(broker: PaperBroker, bad_amount: object) -> None:
    with pytest.raises(BrokerError, match="Montant d'ordre invalide"):
        broker.market_order(BTC, "buy", bad_amount)  # type: ignore[arg-type]
    assert broker.orders == []


def test_invalid_side_rejected(broker: PaperBroker) -> None:
    with pytest.raises(BrokerError, match="Sens d'ordre invalide"):
        broker.market_order(BTC, "hold", 1)  # type: ignore[arg-type]


def test_get_position_returns_a_copy(broker: PaperBroker) -> None:
    broker.market_order(BTC, "buy", 10)
    pos = broker.get_position(BTC)
    pos.amount = 999.0
    assert broker.get_position(BTC).amount == pytest.approx(10)


def test_equity_includes_every_open_symbol(broker: PaperBroker) -> None:
    broker.market_order(BTC, "buy", 10)  # cash 8 998,4995
    broker.set_price(ETH, 10.0)
    broker.market_order(ETH, "buy", 50)  # 50 × 10,005 = 500,25 + 0,50025 → cash 8 497,74925
    assert broker.cash == pytest.approx(8497.74925)
    expected_equity = 8497.74925 + 10 * 100.0 + 50 * 10.0
    assert broker.get_balance(BTC).equity == pytest.approx(expected_equity)
    assert broker.get_balance(ETH).equity == pytest.approx(expected_equity)
    assert broker.get_position(ETH).amount == pytest.approx(50)
    assert broker.get_position(BTC).amount == pytest.approx(10)


# ============================================================================ short
def test_short_sell_and_buy_back(short_broker: PaperBroker, fixed_clock: list[datetime]) -> None:
    b = short_broker
    sell = b.market_order(BTC, "sell", 5)
    # 5 × 99,95 = 499,75 ; frais 0,49975 ; cash 10 000 + 499,25025
    assert sell.price == pytest.approx(99.95)
    assert b.cash == pytest.approx(10_499.25025)
    pos = b.get_position(BTC)
    assert pos.amount == pytest.approx(-5)
    assert pos.side == -1
    assert pos.entry_price == pytest.approx(99.95)
    assert pos.opened_at == T0
    assert b.get_balance(BTC).equity == pytest.approx(10_499.25025 - 500.0)  # 9 999,25025

    b.set_price(BTC, 90.0)
    assert b.get_balance(BTC).equity == pytest.approx(10_499.25025 - 450.0)
    buy = b.market_order(BTC, "buy", 5)
    # 5 × 90,045 = 450,225 ; frais 0,450225 ; coût 450,675225
    assert buy.price == pytest.approx(90.045)
    assert b.cash == pytest.approx(10_499.25025 - 450.675225)  # 10 048,575025
    assert b.cash == pytest.approx(10_048.575025)
    assert not b.get_position(BTC).is_open
    assert b.get_balance(BTC).equity == pytest.approx(10_048.575025)


def test_flip_long_to_short_resets_entry(
    short_broker: PaperBroker, fixed_clock: list[datetime]
) -> None:
    b = short_broker
    b.market_order(BTC, "buy", 10)  # cash 8 998,4995, long 10 @ 100,05
    fixed_clock[0] = T0 + timedelta(hours=2)
    b.market_order(BTC, "sell", 15)  # 15 × 99,95 = 1 499,25 ; frais 1,49925
    pos = b.get_position(BTC)
    assert pos.amount == pytest.approx(-5)
    assert pos.entry_price == pytest.approx(99.95)
    assert pos.opened_at == T0 + timedelta(hours=2)
    assert b.cash == pytest.approx(8998.4995 + 1497.75075)  # 10 496,25025


def test_short_adds_use_vwap(short_broker: PaperBroker) -> None:
    b = short_broker
    b.market_order(BTC, "sell", 5)  # @ 99,95
    b.set_price(BTC, 80.0)
    b.market_order(BTC, "sell", 5)  # @ 79,96
    pos = b.get_position(BTC)
    assert pos.amount == pytest.approx(-10)
    assert pos.entry_price == pytest.approx((5 * 99.95 + 5 * 79.96) / 10)  # 89,955


def test_short_notional_capped_by_equity(short_broker: PaperBroker) -> None:
    b = short_broker
    # 101 × 100 = 10 100 > équité 10 000 → refus ; 100 × 100 = 10 000 → accepté
    with pytest.raises(BrokerError, match="pas de levier"):
        b.market_order(BTC, "sell", 101)
    assert b.orders == []
    b.market_order(BTC, "sell", 100)
    assert b.cash == pytest.approx(10_000 + 9995.0 - 9.995)  # 19 985,005
    assert b.get_balance(BTC).equity == pytest.approx(19_985.005 - 10_000.0)


def test_buying_while_short_needs_cash(short_broker: PaperBroker) -> None:
    b = short_broker
    b.market_order(BTC, "sell", 5)  # cash 10 499,25025
    b.set_price(BTC, 100_000.0)  # le short explose : racheter 5 coûte 500 250 + frais
    with pytest.raises(BrokerError, match="Fonds insuffisants"):
        b.market_order(BTC, "buy", 5)


# ============================================================================ close_position
def test_close_position_long(broker: PaperBroker) -> None:
    broker.market_order(BTC, "buy", 10)
    order = broker.close_position(BTC)
    assert isinstance(order, Order)
    assert order.side == "sell"
    assert order.amount == pytest.approx(10)
    assert order.id == "paper-2"
    assert not broker.get_position(BTC).is_open
    assert broker.cash == pytest.approx(9997.0)


def test_close_position_when_flat_returns_none(broker: PaperBroker) -> None:
    assert broker.close_position(BTC) is None
    assert broker.orders == []


def test_close_position_short(short_broker: PaperBroker) -> None:
    short_broker.market_order(BTC, "sell", 5)
    order = short_broker.close_position(BTC)
    assert order is not None
    assert order.side == "buy"
    assert order.amount == pytest.approx(5)
    assert not short_broker.get_position(BTC).is_open


# ============================================================================ persistance
def test_state_round_trip_via_file(tmp_path: Path, fixed_clock: list[datetime]) -> None:
    path = tmp_path / "state" / "paper.json"
    b1 = PaperBroker(initial_cash=10_000, state_path=path)
    b1.set_price(BTC, 100.0)
    assert not path.exists()
    b1.market_order(BTC, "buy", 10)
    assert path.exists(), "sauvegarde automatique après chaque ordre"

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["cash"] == pytest.approx(8998.4995)
    assert data["positions"][BTC]["amount"] == pytest.approx(10)
    assert data["positions"][BTC]["entry_price"] == pytest.approx(100.05)
    assert data["positions"][BTC]["opened_at"] == T0.isoformat()
    assert data["orders"][0]["id"] == "paper-1"
    assert data["orders"][0]["timestamp"] == "2024-03-10T12:00:00+00:00"

    b2 = PaperBroker(initial_cash=10_000, state_path=str(path))
    assert b2.cash == pytest.approx(8998.4995)
    pos = b2.get_position(BTC)
    assert pos.amount == pytest.approx(10)
    assert pos.entry_price == pytest.approx(100.05)
    assert pos.opened_at == T0
    assert pos.opened_at is not None and pos.opened_at.tzinfo is not None
    assert len(b2.orders) == 1
    assert b2.orders[0] == b1.orders[0]
    assert b2.orders[0].timestamp == T0

    # la numérotation reprend après l'ordre rechargé
    b2.set_price(BTC, 100.0)
    assert b2.market_order(BTC, "sell", 10).id == "paper-2"
    assert b2.cash == pytest.approx(9997.0)
    b3 = PaperBroker(initial_cash=10_000, state_path=path)
    assert b3.cash == pytest.approx(9997.0)
    assert [o.id for o in b3.orders] == ["paper-1", "paper-2"]
    assert not b3.get_position(BTC).is_open


def test_state_dict_round_trip_without_file(broker: PaperBroker) -> None:
    broker.market_order(BTC, "buy", 3)
    state = json.loads(json.dumps(broker.state()))
    other = PaperBroker(initial_cash=1.0)
    other.load_state(state)
    assert other.cash == pytest.approx(broker.cash)
    assert other.get_position(BTC) == broker.get_position(BTC)
    assert other.orders == broker.orders
    assert other.initial_cash == 1.0  # initial_cash reste celui du constructeur


def test_order_seq_recovered_from_ids_when_missing(broker: PaperBroker) -> None:
    broker.market_order(BTC, "buy", 1)
    broker.market_order(BTC, "buy", 1)
    state = broker.state()
    del state["order_seq"]
    other = PaperBroker(initial_cash=10_000)
    other.load_state(state)
    other.set_price(BTC, 100.0)
    assert other.market_order(BTC, "buy", 1).id == "paper-3"


def test_reset_clears_everything_and_saves(tmp_path: Path) -> None:
    path = tmp_path / "paper.json"
    b = PaperBroker(initial_cash=5_000, state_path=path)
    b.set_price(BTC, 100.0)
    b.market_order(BTC, "buy", 2)
    b.reset()
    assert b.cash == 5_000.0
    assert b.orders == []
    assert not b.get_position(BTC).is_open
    assert b.get_price(BTC) == 100.0  # les prix (données de marché) sont conservés
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["cash"] == 5_000.0 and data["orders"] == [] and data["positions"] == {}
    assert b.market_order(BTC, "buy", 1).id == "paper-1"


def test_save_and_load_require_state_path(broker: PaperBroker) -> None:
    with pytest.raises(BrokerError, match="state_path"):
        broker.save()
    with pytest.raises(BrokerError, match="state_path"):
        broker.load()


def test_corrupted_state_file_raises_broker_error(tmp_path: Path) -> None:
    path = tmp_path / "paper.json"
    path.write_text("{pas du json", encoding="utf-8")
    with pytest.raises(BrokerError, match="Impossible de lire"):
        PaperBroker(initial_cash=100, state_path=path)
    path.write_text(json.dumps({"positions": {}}), encoding="utf-8")  # cash manquant
    with pytest.raises(BrokerError, match="invalide"):
        PaperBroker(initial_cash=100, state_path=path)
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(BrokerError, match="invalide"):
        PaperBroker(initial_cash=100, state_path=path)


# ============================================================================ helpers
def test_apply_fill_rules() -> None:
    flat = Position(BTC)
    opened = apply_fill(flat, 2.0, 100.0, T0)
    assert opened == Position(BTC, 2.0, 100.0, T0)
    added = apply_fill(opened, 2.0, 110.0, T0 + timedelta(hours=1))
    assert added.amount == 4.0 and added.entry_price == pytest.approx(105.0)
    assert added.opened_at == T0
    reduced = apply_fill(added, -1.0, 50.0, T0)
    assert reduced.amount == 3.0 and reduced.entry_price == pytest.approx(105.0)
    closed = apply_fill(reduced, -3.0, 50.0, T0)
    assert closed == Position(BTC)
    flipped = apply_fill(reduced, -5.0, 90.0, T0 + timedelta(hours=3))
    assert flipped == Position(BTC, -2.0, 90.0, T0 + timedelta(hours=3))
    # l'objet d'origine n'est jamais modifié
    assert opened.amount == 2.0


def test_check_order_args() -> None:
    check_order_args("buy", 1.5)
    check_order_args("sell", 3)
    with pytest.raises(BrokerError):
        check_order_args("buy", 0)
    with pytest.raises(BrokerError):
        check_order_args("short", 1)


def test_numpy_amounts_accepted(broker: PaperBroker) -> None:
    import numpy as np

    broker.market_order(BTC, "buy", np.float64(2.0))
    broker.market_order(BTC, "buy", np.int64(1))
    assert broker.get_position(BTC).amount == pytest.approx(3.0)
