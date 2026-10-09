"""Tests de la boucle live (:class:`LiveTrader`).

Tout est injecté : source de données factice (tranches de ``make_ohlcv`` qui avancent
d'une bougie), stratégie scriptée (signal fixé par horodatage), ``PaperBroker`` avec
``set_price``, ``RiskManager``, notificateur en mémoire, ``sleep`` et horloge factices.
Les montants attendus (taille, cash après frais et glissement) sont calculés à la main.

Hypothèses chiffrées : cash 10 000, frais 0,1 %, glissement 0,05 %, BTC à 100,
risk_per_trade 1 % → avec un stop à 95, taille = 10 000 × 0,01 / 5 = 20 unités.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from conftest import make_ohlcv

from tradingbot.execution.broker import BrokerError, Order, Side
from tradingbot.execution.paper import PaperBroker
from tradingbot.live import runner as runner_module
from tradingbot.live.notifier import Notifier
from tradingbot.live.runner import (
    LAST_BAR_FILE,
    POSITION_FILE,
    RISK_FILE,
    LiveTrader,
    PositionRecord,
    default_poll_seconds,
    normalise_ohlcv,
    parse_timeframe,
    stop_hit,
)
from tradingbot.risk import RiskConfig, RiskManager
from tradingbot.strategies.base import Strategy

BTC = "BTC/USDT"
N_BARS = 100
START = 60  # la première itération voit les bougies 0..59 (la 59 vient de clôturer)
ONE_HOUR = pd.Timedelta(hours=1)
NAN = math.nan

# --- valeurs calculées à la main --------------------------------------------------------
# Achat 20 @ 100 × 1,0005 = 100,05 : notionnel 2 001, frais 2,001 → cash 7 996,999
CASH_AFTER_OPEN = 10_000.0 - 20 * 100.05 - 20 * 100.05 * 0.001
EQUITY_AFTER_OPEN = CASH_AFTER_OPEN + 20 * 100.0  # 9 996,999
# Vente 20 @ 110 × 0,9995 = 109,945 : notionnel 2 198,9, frais 2,1989 → cash 10 193,7001
CASH_AFTER_CLOSE_110 = CASH_AFTER_OPEN + 20 * 109.945 - 20 * 109.945 * 0.001
# Vente 20 @ 94 × 0,9995 = 93,953 : notionnel 1 879,06, frais 1,87906 → cash 9 874,17994
CASH_AFTER_STOP_94 = CASH_AFTER_OPEN + 20 * 93.953 - 20 * 93.953 * 0.001
# Vente 20 @ 121 × 0,9995 = 120,9395 : notionnel 2 418,79, frais 2,41879 → 10 413,37021
CASH_AFTER_TP_121 = CASH_AFTER_OPEN + 20 * 120.9395 - 20 * 120.9395 * 0.001
# Sans stop : taille = 10 000 × 0,95 / 100 = 95 ; achat 95 @ 100,05 : notionnel 9 504,75,
# frais 9,50475 → cash 485,74525
CASH_AFTER_FULL_OPEN = 10_000.0 - 95 * 100.05 * 1.001
# Kill switch à 75 : équité 485,74525 + 95 × 75 = 7 610,74525 (drawdown 23,9 % > 20 %) ;
# vente 95 @ 74,9625 : notionnel 7 121,4375, frais 7,1214375 → cash 7 600,0613125
EQUITY_AT_75 = CASH_AFTER_FULL_OPEN + 95 * 75.0
CASH_AFTER_KILL = CASH_AFTER_FULL_OPEN + 95 * 74.9625 * 0.999
# Short 20 @ 100 × 0,9995 = 99,95 : notionnel 1 999, frais 1,999 → cash 11 997,001
CASH_AFTER_SHORT = 10_000.0 + 20 * 99.95 * 0.999
# Rachat 20 @ 106 × 1,0005 = 106,053 : notionnel 2 121,06, frais 2,12106 → 9 873,81994
CASH_AFTER_SHORT_STOP = CASH_AFTER_SHORT - 20 * 106.053 * 1.001


# ============================================================================ fakes
class FakeDataSource:
    """Renvoie ``df.iloc[:pos]`` (tranche de bougies clôturées) ; ``advance`` ou
    ``auto_advance`` fait avancer d'une bougie par appel."""

    def __init__(self, df: pd.DataFrame, start: int = START, auto_advance: bool = False) -> None:
        self.df = df
        self.pos = start
        self.auto_advance = auto_advance
        self.calls = 0
        self.last_n: int | None = None

    def __call__(self, n_bars: int) -> pd.DataFrame:
        self.calls += 1
        self.last_n = n_bars
        end = min(self.pos, len(self.df))
        out = self.df.iloc[max(0, end - n_bars) : end].copy()
        if self.auto_advance:
            self.pos += 1
        return out

    def advance(self, k: int = 1) -> None:
        self.pos += k


class ScriptedStrategy(Strategy):
    """Stratégie dont les signaux sont fixés par horodatage : ``{ts: (signal, stop, tp)}``.

    Déterministe par bougie (la même bougie redonne le même signal), sans look-ahead.
    """

    name = "scripted"
    description = "Stratégie de test aux signaux scriptés."
    default_params: dict[str, Any] = {}

    def __init__(
        self, script: dict[pd.Timestamp, tuple[int, float, float]] | None = None, warmup: int = 0
    ) -> None:
        super().__init__()
        self.script = script or {}
        self.warmup = warmup
        self.calls = 0

    def warmup_bars(self) -> int:
        return self.warmup

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        self.calls += 1
        out = self.empty_signals(df)
        for ts, (sig, stop, tp) in self.script.items():
            if ts in out.index:
                out.loc[ts, "signal"] = sig
                out.loc[ts, "stop_loss"] = stop
                out.loc[ts, "take_profit"] = tp
        return out


class ExplodingStrategy(ScriptedStrategy):
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        raise RuntimeError("stratégie cassée")


class ListNotifier(Notifier):
    def __init__(self) -> None:
        self.messages: list[str] = []

    def send(self, text: str) -> bool:
        self.messages.append(text)
        return True


class FailingBroker(PaperBroker):
    """PaperBroker dont les ordres échouent tant que ``failing`` est True."""

    failing = True

    def market_order(self, symbol: str, side: Side, amount: float) -> Order:
        if self.failing:
            raise BrokerError("panne API simulée")
        return super().market_order(symbol, side, amount)


class FakeSleep:
    def __init__(self, raise_after: int | None = None, on_call: Any = None) -> None:
        self.calls: list[float] = []
        self.raise_after = raise_after
        self.on_call = on_call

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        if self.on_call is not None:
            self.on_call(len(self.calls))
        if self.raise_after is not None and len(self.calls) >= self.raise_after:
            raise KeyboardInterrupt


# ============================================================================ fixtures
@pytest.fixture
def candles() -> pd.DataFrame:
    """100 bougies horaires à partir du 2024-01-01 00:00 UTC."""
    return make_ohlcv(n=N_BARS, seed=42)


def at(candles: pd.DataFrame, i: int) -> pd.Timestamp:
    return candles.index[i]


def closed_at(candles: pd.DataFrame, i: int) -> pd.Timestamp:
    """Instant où la bougie ``i`` vient de clôturer (= ouverture de la bougie ``i+1``)."""
    return candles.index[i] + ONE_HOUR


def make_risk(**overrides: Any) -> RiskManager:
    cfg: dict[str, Any] = {
        "risk_per_trade": 0.01,
        "max_position_fraction": 0.95,
        "max_drawdown": 0.20,
        "daily_loss_limit": 0.05,
        "allow_short": False,
        "min_trade_notional": 10.0,
    }
    cfg.update(overrides)
    return RiskManager(RiskConfig(**cfg))


def make_broker(allow_short: bool = False, **kwargs: Any) -> PaperBroker:
    b = PaperBroker(
        initial_cash=10_000.0, fee_rate=0.001, slippage=0.0005, allow_short=allow_short, **kwargs
    )
    b.set_price(BTC, 100.0)
    return b


class Setup:
    """Petit conteneur : trader + dépendances injectées."""

    def __init__(
        self,
        tmp_path: Path,
        candles: pd.DataFrame,
        script: dict[pd.Timestamp, tuple[int, float, float]] | None = None,
        *,
        strategy: Strategy | None = None,
        broker: PaperBroker | None = None,
        risk: RiskManager | None = None,
        allow_short: bool = False,
        auto_advance: bool = False,
        lookback: int = 500,
        sleep: Any = None,
        clock: Any = None,
        state_dir: Path | None = None,
    ) -> None:
        self.candles = candles
        self.source = FakeDataSource(candles, auto_advance=auto_advance)
        self.strategy = strategy if strategy is not None else ScriptedStrategy(script)
        self.broker = broker if broker is not None else make_broker()
        self.risk = risk if risk is not None else make_risk()
        self.notifier = ListNotifier()
        self.state_dir = state_dir if state_dir is not None else tmp_path / "state"
        self.sleep = sleep if sleep is not None else FakeSleep()
        self.trader = LiveTrader(
            strategy=self.strategy,
            broker=self.broker,
            symbol=BTC,
            timeframe="1h",
            risk=self.risk,
            data_source=self.source,
            state_dir=self.state_dir,
            lookback=lookback,
            notifier=self.notifier,
            allow_short=allow_short,
            sleep=self.sleep,
            clock=clock,
        )

    def step(self, i: int) -> dict[str, Any]:
        """Itération à l'instant où la bougie ``i`` vient de clôturer."""
        return self.trader.step(closed_at(self.candles, i))

    def read(self, name: str) -> Any:
        return json.loads((self.state_dir / name).read_text(encoding="utf-8"))


def long_script(candles: pd.DataFrame, *bars: int, stop: float = 95.0, tp: float = 120.0) -> dict:
    return {at(candles, i): (1, stop, tp) for i in bars}


# ============================================================================ premier pas
def test_first_step_opens_long_with_expected_size_and_cash(
    tmp_path: Path, candles: pd.DataFrame
) -> None:
    s = Setup(tmp_path, candles, long_script(candles, 59))
    state = s.step(59)

    assert state["action"] == "open"
    assert state["signal"] == 1
    assert state["position"] == 1
    assert state["amount"] == pytest.approx(20.0)
    assert state["timestamp"] == at(candles, 59).isoformat()
    assert state["now"] == closed_at(candles, 59).isoformat()
    assert state["stop_loss"] == 95.0
    assert state["take_profit"] == 120.0
    assert state["price"] == 100.0
    assert state["order"] == "paper-1"
    assert state["equity"] == pytest.approx(EQUITY_AFTER_OPEN)
    assert state["reason"] == "entrée sur signal"
    assert set(state) >= {"timestamp", "signal", "action", "position", "equity", "reason"}

    assert len(s.broker.orders) == 1
    order = s.broker.orders[0]
    assert order.side == "buy"
    assert order.amount == pytest.approx(20.0)
    assert order.price == pytest.approx(100.05)
    assert order.fee == pytest.approx(2.001)
    assert s.broker.cash == pytest.approx(CASH_AFTER_OPEN)
    assert s.broker.get_position(BTC).amount == pytest.approx(20.0)
    assert s.source.last_n == 500
    assert s.strategy.calls == 1

    # notification : une ligne en français
    assert len(s.notifier.messages) == 1
    msg = s.notifier.messages[0]
    assert msg.startswith("[BTC/USDT] Entrée LONG")
    assert "stop 95" in msg and "objectif 120" in msg and "USDT" in msg


def test_state_files_are_written_after_open(tmp_path: Path, candles: pd.DataFrame) -> None:
    s = Setup(tmp_path, candles, long_script(candles, 59))
    s.step(59)

    last = s.read(LAST_BAR_FILE)
    assert last["last_bar"] == at(candles, 59).isoformat()
    assert last["signal"] == 1
    assert last["symbol"] == BTC and last["timeframe"] == "1h"

    pos = s.read(POSITION_FILE)
    assert pos["side"] == 1
    assert pos["amount"] == pytest.approx(20.0)
    assert pos["entry_price"] == pytest.approx(100.05)
    assert pos["stop_loss"] == 95.0
    assert pos["take_profit"] == 120.0
    assert pos["opened_at"] == closed_at(candles, 59).isoformat()

    risk = s.read(RISK_FILE)
    assert risk["kill_switch"] is False
    assert risk["peak_equity"] == pytest.approx(10_000.0)
    assert not list(s.state_dir.glob("*.tmp"))


def test_same_candle_again_is_skipped_without_new_order(
    tmp_path: Path, candles: pd.DataFrame
) -> None:
    s = Setup(tmp_path, candles, long_script(candles, 59))
    s.step(59)
    state = s.step(59)

    assert state["action"] == "skip"
    assert state["reason"] == "bougie déjà traitée"
    assert state["signal"] == 1  # dernier signal connu
    assert state["position"] == 1
    assert state["timestamp"] == at(candles, 59).isoformat()
    assert len(s.broker.orders) == 1
    assert s.strategy.calls == 1  # la stratégie n'est même pas recalculée
    assert len(s.notifier.messages) == 1  # pas de notification pour un skip


def test_next_candle_with_signal_zero_closes(tmp_path: Path, candles: pd.DataFrame) -> None:
    s = Setup(tmp_path, candles, long_script(candles, 59))
    s.step(59)
    s.source.advance()
    s.broker.set_price(BTC, 110.0)
    state = s.step(60)

    assert state["action"] == "close"
    assert state["signal"] == 0
    assert state["position"] == 0
    assert state["amount"] == 0.0
    assert state["stop_loss"] is None
    assert state["reason"] == "signal de sortie"
    assert state["order"] == "paper-2"
    assert len(s.broker.orders) == 2
    sell = s.broker.orders[1]
    assert sell.side == "sell"
    assert sell.amount == pytest.approx(20.0)
    assert sell.price == pytest.approx(109.945)
    assert s.broker.cash == pytest.approx(CASH_AFTER_CLOSE_110)
    assert state["equity"] == pytest.approx(CASH_AFTER_CLOSE_110)
    assert not s.broker.get_position(BTC).is_open
    assert s.read(POSITION_FILE)["side"] == 0
    assert s.read(LAST_BAR_FILE)["last_bar"] == at(candles, 60).isoformat()
    assert s.notifier.messages[-1].startswith("[BTC/USDT] Sortie sur signal : VENTE 20")


def test_hold_when_signal_unchanged(tmp_path: Path, candles: pd.DataFrame) -> None:
    s = Setup(tmp_path, candles, long_script(candles, 59, 60))
    s.step(59)
    s.source.advance()
    state = s.step(60)
    assert state["action"] == "hold"
    assert state["reason"] == "position conservée"
    assert state["position"] == 1
    assert len(s.broker.orders) == 1
    assert len(s.notifier.messages) == 1


def test_flat_and_no_signal_is_hold(tmp_path: Path, candles: pd.DataFrame) -> None:
    s = Setup(tmp_path, candles, {})
    state = s.step(59)
    assert state["action"] == "hold"
    assert state["signal"] == 0
    assert state["position"] == 0
    assert s.broker.orders == []
    assert s.notifier.messages == []
    assert s.read(LAST_BAR_FILE)["last_bar"] == at(candles, 59).isoformat()


# ============================================================================ stops
def test_stop_crossed_between_candles_closes(tmp_path: Path, candles: pd.DataFrame) -> None:
    s = Setup(tmp_path, candles, long_script(candles, 59))
    s.step(59)
    s.broker.set_price(BTC, 94.0)  # même bougie, prix sous le stop 95
    state = s.step(59)

    assert state["action"] == "stop_loss"
    assert state["position"] == 0
    assert "Stop touché" in state["reason"] and "94" in state["reason"] and "95" in state["reason"]
    assert state["order"] == "paper-2"
    assert len(s.broker.orders) == 2
    assert s.broker.orders[1].price == pytest.approx(93.953)
    assert s.broker.cash == pytest.approx(CASH_AFTER_STOP_94)
    assert s.read(POSITION_FILE)["side"] == 0
    assert s.notifier.messages[-1].startswith("[BTC/USDT] Stop touché")

    # toujours la même bougie : rien d'autre à faire
    again = s.step(59)
    assert again["action"] == "skip"
    assert len(s.broker.orders) == 2


def test_stop_not_triggered_above_level(tmp_path: Path, candles: pd.DataFrame) -> None:
    s = Setup(tmp_path, candles, long_script(candles, 59))
    s.step(59)
    s.broker.set_price(BTC, 95.5)
    assert s.step(59)["action"] == "skip"
    s.broker.set_price(BTC, 95.0)  # égalité = touché
    assert s.step(59)["action"] == "stop_loss"


def test_take_profit_crossed_closes(tmp_path: Path, candles: pd.DataFrame) -> None:
    s = Setup(tmp_path, candles, long_script(candles, 59))
    s.step(59)
    s.broker.set_price(BTC, 121.0)
    state = s.step(59)
    assert state["action"] == "take_profit"
    assert "Objectif atteint" in state["reason"]
    assert s.broker.cash == pytest.approx(CASH_AFTER_TP_121)
    assert state["position"] == 0


def test_stop_exit_on_new_candle_does_not_reenter_before_next_candle(
    tmp_path: Path, candles: pd.DataFrame
) -> None:
    s = Setup(tmp_path, candles, long_script(candles, 59, 60, 61))
    s.step(59)
    s.source.advance()
    s.broker.set_price(BTC, 94.0)
    state = s.step(60)
    assert state["action"] == "stop_loss"
    assert len(s.broker.orders) == 2
    # la bougie 60 est considérée traitée : pas de ré-entrée dans la même bougie
    assert s.trader.last_bar == at(candles, 60)
    s.broker.set_price(BTC, 100.0)
    assert s.step(60)["action"] == "skip"
    assert len(s.broker.orders) == 2
    # bougie suivante, le signal demande toujours un long : ré-entrée
    s.source.advance()
    state = s.step(61)
    assert state["action"] == "open"
    assert len(s.broker.orders) == 3


def test_strategy_can_move_the_stop_while_holding(tmp_path: Path, candles: pd.DataFrame) -> None:
    script = {at(candles, 59): (1, 95.0, NAN), at(candles, 60): (1, 98.0, NAN)}
    s = Setup(tmp_path, candles, script)
    s.step(59)
    s.source.advance()
    state = s.step(60)
    assert state["action"] == "hold"
    assert "stop mis à jour à 98" in state["reason"]
    assert s.trader.position.stop_loss == 98.0
    assert s.trader.position.take_profit is None
    assert s.read(POSITION_FILE)["stop_loss"] == 98.0
    s.broker.set_price(BTC, 97.0)
    assert s.step(60)["action"] == "stop_loss"


# ============================================================================ shorts
def test_short_signal_ignored_without_allow_short(tmp_path: Path, candles: pd.DataFrame) -> None:
    s = Setup(tmp_path, candles, {at(candles, 59): (-1, 105.0, 80.0)})
    state = s.step(59)
    assert state["action"] == "hold"
    assert state["signal"] == 0
    assert "short ignoré" in state["reason"]
    assert s.broker.orders == []
    assert s.read(LAST_BAR_FILE)["signal"] == 0


def test_short_opens_and_stops_when_allowed(tmp_path: Path, candles: pd.DataFrame) -> None:
    s = Setup(
        tmp_path,
        candles,
        {at(candles, 59): (-1, 105.0, 80.0)},
        broker=make_broker(allow_short=True),
        risk=make_risk(allow_short=True),
        allow_short=True,
    )
    state = s.step(59)
    assert state["action"] == "open"
    assert state["signal"] == -1
    assert state["position"] == -1
    assert state["amount"] == pytest.approx(20.0)  # 10 000 × 1 % / |100 - 105|
    assert s.broker.orders[0].side == "sell"
    assert s.broker.orders[0].price == pytest.approx(99.95)
    assert s.broker.cash == pytest.approx(CASH_AFTER_SHORT)
    assert s.notifier.messages[0].startswith("[BTC/USDT] Entrée SHORT")

    s.broker.set_price(BTC, 106.0)  # au-dessus du stop d'un short
    state = s.step(59)
    assert state["action"] == "stop_loss"
    assert s.broker.orders[1].side == "buy"
    assert s.broker.cash == pytest.approx(CASH_AFTER_SHORT_STOP)


def test_short_refused_by_risk_manager(tmp_path: Path, candles: pd.DataFrame) -> None:
    s = Setup(
        tmp_path,
        candles,
        {at(candles, 59): (-1, 105.0, 80.0)},
        broker=make_broker(allow_short=True),
        risk=make_risk(allow_short=False),
        allow_short=True,
    )
    state = s.step(59)
    assert state["action"] == "refused"
    assert "vente à découvert" in state["reason"]
    assert s.broker.orders == []
    assert any("Refus du gestionnaire de risque" in m for m in s.notifier.messages)


# ============================================================================ garde-fous
def test_kill_switch_closes_position_and_blocks_reentry(
    tmp_path: Path, candles: pd.DataFrame
) -> None:
    script = {at(candles, i): (1, NAN, NAN) for i in (59, 60, 61)}
    s = Setup(tmp_path, candles, script)
    state = s.step(59)
    assert state["action"] == "open"
    assert state["amount"] == pytest.approx(95.0)  # 10 000 × 0,95 / 100
    assert s.broker.cash == pytest.approx(CASH_AFTER_FULL_OPEN)
    assert "sans stop" in s.notifier.messages[0]

    s.source.advance()
    s.broker.set_price(BTC, 75.0)
    state = s.step(60)
    assert state["action"] == "kill_switch"
    assert state["position"] == 0
    assert "position clôturée" in state["reason"]
    assert s.risk.kill_switch_active()
    assert len(s.broker.orders) == 2
    assert s.broker.orders[1].side == "sell"
    assert s.broker.cash == pytest.approx(CASH_AFTER_KILL)
    assert state["equity"] == pytest.approx(CASH_AFTER_KILL)
    assert s.notifier.messages[-1].startswith("[BTC/USDT] KILL SWITCH")
    assert s.read(RISK_FILE)["kill_switch"] is True

    s.source.advance()
    state = s.step(61)
    assert state["action"] == "kill_switch"
    assert "ouverture refusée" in state["reason"]
    assert len(s.broker.orders) == 2


def test_daily_limit_closes_and_blocks_for_the_day(tmp_path: Path, candles: pd.DataFrame) -> None:
    script = {at(candles, i): (1, NAN, NAN) for i in (59, 60, 61)}
    s = Setup(tmp_path, candles, script, risk=make_risk(max_drawdown=None, daily_loss_limit=0.05))
    s.step(59)
    s.source.advance()
    s.broker.set_price(BTC, 90.0)  # équité 9 035,74 : -9,6 % sur la journée
    state = s.step(60)
    assert state["action"] == "daily_limit"
    assert state["position"] == 0
    assert not s.risk.kill_switch_active()
    assert len(s.broker.orders) == 2
    assert s.notifier.messages[-1].startswith("[BTC/USDT] LIMITE JOURNALIÈRE")

    s.source.advance()
    state = s.step(61)  # même jour UTC
    assert state["action"] == "daily_limit"
    assert "ouverture refusée" in state["reason"]
    assert len(s.broker.orders) == 2


def test_blocked_but_flat_and_no_signal_is_a_quiet_hold(
    tmp_path: Path, candles: pd.DataFrame
) -> None:
    s = Setup(tmp_path, candles, {})
    s.risk.load_state({**s.risk.state(), "kill_switch": True})
    state = s.step(59)
    assert state["action"] == "hold"
    assert "kill switch" in state["reason"]
    assert s.notifier.messages == []


def test_size_too_small_skips(tmp_path: Path, candles: pd.DataFrame) -> None:
    s = Setup(tmp_path, candles, long_script(candles, 59), risk=make_risk(min_trade_notional=5000))
    state = s.step(59)  # 20 unités × 100 = 2 000 < 5 000
    assert state["action"] == "skip"
    assert state["reason"] == "taille trop petite"
    assert s.broker.orders == []
    assert s.read(LAST_BAR_FILE)["last_bar"] == at(candles, 59).isoformat()


def test_warmup_period_skips_without_trading(tmp_path: Path, candles: pd.DataFrame) -> None:
    strategy = ScriptedStrategy(long_script(candles, 59), warmup=60)  # 60 bougies <= 60
    s = Setup(tmp_path, candles, strategy=strategy)
    state = s.step(59)
    assert state["action"] == "skip"
    assert "chauffe" in state["reason"]
    assert s.broker.orders == []
    assert strategy.calls == 0
    assert s.step(59)["reason"] == "bougie déjà traitée"
    s.source.advance()  # 61 bougies > 60 : la stratégie est consultée
    s.step(60)
    assert strategy.calls == 1


# ============================================================================ erreurs
def test_broker_error_returns_error_state_and_retries_next_call(
    tmp_path: Path, candles: pd.DataFrame
) -> None:
    broker = FailingBroker(initial_cash=10_000.0, fee_rate=0.001, slippage=0.0005)
    broker.set_price(BTC, 100.0)
    s = Setup(tmp_path, candles, long_script(candles, 59), broker=broker)
    state = s.step(59)
    assert state["action"] == "error"
    assert "panne API simulée" in state["reason"]
    assert state["position"] == 0
    assert broker.orders == []
    assert s.notifier.messages[-1].startswith("[BTC/USDT] ERREUR broker")
    # la bougie n'est pas marquée traitée : nouvel essai à l'appel suivant
    assert s.trader.last_bar is None
    broker.failing = False
    assert s.step(59)["action"] == "open"
    assert len(broker.orders) == 1


def test_data_source_failure_is_reported_as_error(tmp_path: Path, candles: pd.DataFrame) -> None:
    s = Setup(tmp_path, candles, long_script(candles, 59))

    def broken(n: int) -> pd.DataFrame:
        raise RuntimeError("réseau indisponible")

    s.trader.data_source = broken
    state = s.step(59)
    assert state["action"] == "error"
    assert "Récupération des bougies impossible" in state["reason"]
    assert "réseau indisponible" in state["reason"]
    assert state["timestamp"] is None
    assert s.broker.orders == []


def test_invalid_candles_are_reported_as_error(tmp_path: Path, candles: pd.DataFrame) -> None:
    s = Setup(tmp_path, candles, long_script(candles, 59))
    s.trader.data_source = lambda n: pd.DataFrame({"foo": [1.0]})
    state = s.step(59)
    assert state["action"] == "error"
    assert "Bougies invalides" in state["reason"]


def test_unknown_price_is_reported_as_error(tmp_path: Path, candles: pd.DataFrame) -> None:
    broker = PaperBroker(initial_cash=10_000.0)  # aucun prix fixé
    s = Setup(tmp_path, candles, long_script(candles, 59), broker=broker)
    state = s.step(59)
    assert state["action"] == "error"
    assert "Aucun prix connu" in state["reason"]


# ============================================================================ persistance
def test_new_instance_resumes_without_duplicate_orders(
    tmp_path: Path, candles: pd.DataFrame
) -> None:
    s = Setup(tmp_path, candles, long_script(candles, 59))
    s.step(59)
    assert len(s.broker.orders) == 1

    # redémarrage : nouveau LiveTrader, même broker, même répertoire d'état
    s2 = Setup(
        tmp_path,
        candles,
        long_script(candles, 59),
        broker=s.broker,
        risk=make_risk(),
        state_dir=s.state_dir,
    )
    assert s2.trader.last_bar == at(candles, 59)
    assert s2.trader.last_signal == 1
    assert s2.trader.position == PositionRecord(
        side=1,
        amount=20.0,
        entry_price=pytest.approx(100.05),  # type: ignore[arg-type]
        stop_loss=95.0,
        take_profit=120.0,
        opened_at=closed_at(candles, 59).isoformat(),
    )
    assert s2.risk.peak_equity == pytest.approx(10_000.0)

    state = s2.step(59)
    assert state["action"] == "skip"
    assert len(s.broker.orders) == 1

    # le stop mémorisé est toujours actif après redémarrage
    s.broker.set_price(BTC, 94.0)
    assert s2.step(59)["action"] == "stop_loss"
    assert len(s.broker.orders) == 2


def test_kill_switch_survives_restart(tmp_path: Path, candles: pd.DataFrame) -> None:
    script = {at(candles, i): (1, NAN, NAN) for i in (59, 60, 61)}
    s = Setup(tmp_path, candles, script)
    s.step(59)
    s.source.advance()
    s.broker.set_price(BTC, 75.0)
    assert s.step(60)["action"] == "kill_switch"

    # redémarrage avec un RiskManager neuf : le kill switch est relu depuis risk.json
    s2 = Setup(tmp_path, candles, script, broker=s.broker, risk=make_risk(), state_dir=s.state_dir)
    assert s2.risk.kill_switch_active()
    s2.source.advance(2)
    state = s2.step(61)
    assert state["action"] == "kill_switch"
    assert "ouverture refusée" in state["reason"]
    assert len(s.broker.orders) == 2


def test_paper_broker_state_file_and_trader_state_resume_together(
    tmp_path: Path, candles: pd.DataFrame
) -> None:
    broker_path = tmp_path / "paper.json"
    broker = make_broker(state_path=broker_path)
    s = Setup(tmp_path, candles, long_script(candles, 59), broker=broker)
    s.step(59)

    broker2 = make_broker(state_path=broker_path)  # recharge cash + position
    assert broker2.cash == pytest.approx(CASH_AFTER_OPEN)
    s2 = Setup(tmp_path, candles, long_script(candles, 59), broker=broker2, state_dir=s.state_dir)
    assert s2.step(59)["action"] == "skip"
    assert len(broker2.orders) == 1


def test_position_adopted_from_broker_when_state_missing(
    tmp_path: Path, candles: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    broker = make_broker()
    broker.market_order(BTC, "buy", 10.0)  # position prise « à la main » avant le bot
    s = Setup(tmp_path, candles, long_script(candles, 59), broker=broker)
    with caplog.at_level(logging.WARNING, logger="tradingbot.live.runner"):
        state = s.step(59)
    assert "position du broker adoptée" in caplog.text
    assert state["action"] == "hold"
    assert state["position"] == 1
    assert "stop mis à jour à 95" in state["reason"]
    assert s.trader.position.amount == pytest.approx(10.0)
    assert s.trader.position.entry_price == pytest.approx(100.05)
    assert s.trader.position.stop_loss == 95.0
    assert len(broker.orders) == 1
    assert s.read(POSITION_FILE)["amount"] == pytest.approx(10.0)


def test_stale_position_record_is_cleared_when_broker_is_flat(
    tmp_path: Path, candles: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / POSITION_FILE).write_text(
        json.dumps(
            {"side": 1, "amount": 5, "entry_price": 100, "stop_loss": 95, "take_profit": None}
        ),
        encoding="utf-8",
    )
    s = Setup(tmp_path, candles, {}, state_dir=state_dir)
    assert s.trader.position.side == 1
    with caplog.at_level(logging.WARNING, logger="tradingbot.live.runner"):
        state = s.step(59)
    assert "broker est plat" in caplog.text
    assert state["action"] == "hold"
    assert state["position"] == 0
    assert s.broker.orders == []
    assert s.read(POSITION_FILE)["side"] == 0


def test_corrupted_state_files_are_ignored(
    tmp_path: Path, candles: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / LAST_BAR_FILE).write_text("{pas du json", encoding="utf-8")
    (state_dir / POSITION_FILE).write_text(json.dumps({"side": 7}), encoding="utf-8")
    (state_dir / RISK_FILE).write_text("[]", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="tradingbot.live.runner"):
        s = Setup(tmp_path, candles, long_script(candles, 59), state_dir=state_dir)
    assert s.trader.last_bar is None
    assert s.trader.position == PositionRecord()
    assert "illisible" in caplog.text
    assert s.step(59)["action"] == "open"


# ============================================================================ bougies
def test_unclosed_last_candle_is_dropped(tmp_path: Path, candles: pd.DataFrame) -> None:
    s = Setup(tmp_path, candles, long_script(candles, 59))
    # 30 min après l'ouverture de la bougie 59 : elle n'est pas clôturée
    state = s.trader.step(at(candles, 59) + pd.Timedelta(minutes=30))
    assert state["action"] == "hold"
    assert state["timestamp"] == at(candles, 58).isoformat()
    assert s.broker.orders == []
    # à la clôture exacte de la bougie 59, elle est prise en compte
    state = s.step(59)
    assert state["action"] == "open"
    assert state["timestamp"] == at(candles, 59).isoformat()


def test_all_candles_unclosed_gives_skip(tmp_path: Path, candles: pd.DataFrame) -> None:
    s = Setup(tmp_path, candles, long_script(candles, 59))
    state = s.trader.step(at(candles, 0))  # avant toute clôture
    assert state["action"] == "skip"
    assert "aucune bougie clôturée" in state["reason"]
    assert state["timestamp"] is None
    assert s.broker.orders == []
    assert not (s.state_dir / LAST_BAR_FILE).exists()


def test_naive_now_is_treated_as_utc(tmp_path: Path, candles: pd.DataFrame) -> None:
    s = Setup(tmp_path, candles, long_script(candles, 59))
    naive = closed_at(candles, 59).tz_localize(None).to_pydatetime()
    assert naive.tzinfo is None
    state = s.trader.step(naive)
    assert state["action"] == "open"
    assert state["now"] == closed_at(candles, 59).isoformat()


def test_clock_is_used_when_now_is_omitted(tmp_path: Path, candles: pd.DataFrame) -> None:
    s = Setup(tmp_path, candles, long_script(candles, 59), clock=lambda: closed_at(candles, 59))
    state = s.trader.step()
    assert state["action"] == "open"
    assert s.read(LAST_BAR_FILE)["updated_at"] == closed_at(candles, 59).isoformat()


def test_lookback_limits_requested_bars(tmp_path: Path, candles: pd.DataFrame) -> None:
    s = Setup(tmp_path, candles, long_script(candles, 59), lookback=30)
    s.step(59)
    assert s.source.last_n == 30


# ============================================================================ run()
def test_run_calls_step_max_iterations_times_and_sleeps_between(
    tmp_path: Path, candles: pd.DataFrame
) -> None:
    s = Setup(tmp_path, candles, {}, auto_advance=True)
    s.trader.run(poll_seconds=5, max_iterations=3)
    assert s.source.calls == 3
    assert s.sleep.calls == [5, 5]
    assert s.trader.last_bar == at(candles, 61)


def test_run_uses_default_poll_seconds(tmp_path: Path, candles: pd.DataFrame) -> None:
    s = Setup(tmp_path, candles, {}, auto_advance=True)
    s.trader.run(max_iterations=2)
    assert s.sleep.calls == [300.0]  # 1h / 12


def test_run_stops_cleanly_on_keyboard_interrupt(
    tmp_path: Path, candles: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    s = Setup(tmp_path, candles, long_script(candles, 59), sleep=FakeSleep(raise_after=1))
    with caplog.at_level(logging.INFO, logger="tradingbot.live.runner"):
        s.trader.run(poll_seconds=1, max_iterations=10)  # ne lève pas
    assert s.source.calls == 1
    assert s.sleep.calls == [1]
    assert "Arrêt demandé" in caplog.text
    assert "toujours ouverte" in caplog.text  # la position long reste mémorisée
    assert s.read(POSITION_FILE)["side"] == 1


def test_run_stops_when_kill_switch_triggers(tmp_path: Path, candles: pd.DataFrame) -> None:
    prices = {"p": 100.0}
    broker = PaperBroker(
        initial_cash=10_000.0, fee_rate=0.001, slippage=0.0005, price_feed=lambda s: prices["p"]
    )

    def crash(call_number: int) -> None:
        prices["p"] = 75.0

    s = Setup(
        tmp_path,
        candles,
        {at(candles, 59): (1, NAN, NAN)},
        broker=broker,
        auto_advance=True,
        sleep=FakeSleep(on_call=crash),
    )
    s.trader.run(poll_seconds=2, max_iterations=10)
    assert s.source.calls == 2  # ouverture, puis kill switch → arrêt
    assert s.sleep.calls == [2]
    assert s.risk.kill_switch_active()
    assert not broker.get_position(BTC).is_open
    assert any("arrêt de la boucle" in m for m in s.notifier.messages)


def test_run_survives_unexpected_exception(
    tmp_path: Path, candles: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    s = Setup(tmp_path, candles, strategy=ExplodingStrategy(), auto_advance=True)
    with caplog.at_level(logging.ERROR, logger="tradingbot.live.runner"):
        s.trader.run(poll_seconds=1, max_iterations=2)
    assert s.source.calls == 2
    assert "Erreur inattendue" in caplog.text
    assert sum("ERREUR inattendue" in m for m in s.notifier.messages) == 2


def test_run_zero_iterations_does_nothing(tmp_path: Path, candles: pd.DataFrame) -> None:
    s = Setup(tmp_path, candles, {})
    s.trader.run(poll_seconds=1, max_iterations=0)
    assert s.source.calls == 0


@pytest.mark.parametrize("bad", [0, -1, math.nan])
def test_run_rejects_bad_poll_seconds(tmp_path: Path, candles: pd.DataFrame, bad: float) -> None:
    s = Setup(tmp_path, candles, {})
    with pytest.raises(ValueError, match="poll_seconds"):
        s.trader.run(poll_seconds=bad, max_iterations=1)


def test_notifier_failure_does_not_break_step(
    tmp_path: Path, candles: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    class Exploding(Notifier):
        def send(self, text: str) -> bool:
            raise RuntimeError("telegram HS")

    s = Setup(tmp_path, candles, long_script(candles, 59))
    s.trader.notifier = Exploding()
    with caplog.at_level(logging.WARNING, logger="tradingbot.live.runner"):
        assert s.step(59)["action"] == "open"
    assert "telegram HS" in caplog.text


# ============================================================================ fonctions pures
@pytest.mark.parametrize(
    ("side", "price", "stop", "tp", "expected"),
    [
        (1, 94.0, 95.0, 120.0, "stop_loss"),
        (1, 95.0, 95.0, 120.0, "stop_loss"),
        (1, 121.0, 95.0, 120.0, "take_profit"),
        (1, 100.0, 95.0, 120.0, None),
        (1, 100.0, None, None, None),
        (1, 50.0, 95.0, 40.0, "stop_loss"),  # stop ET objectif : stop (pessimiste)
        (-1, 106.0, 105.0, 80.0, "stop_loss"),
        (-1, 79.0, 105.0, 80.0, "take_profit"),
        (-1, 100.0, 105.0, 80.0, None),
        (0, 0.0, 95.0, 120.0, None),
    ],
)
def test_stop_hit(side: int, price: float, stop: Any, tp: Any, expected: str | None) -> None:
    assert stop_hit(side, price, stop, tp) == expected


@pytest.mark.parametrize(
    ("timeframe", "expected"), [("1m", 10.0), ("1h", 300.0), ("4h", 1200.0), ("1d", 7200.0)]
)
def test_default_poll_seconds(timeframe: str, expected: float) -> None:
    assert default_poll_seconds(timeframe) == expected


def test_parse_timeframe_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner_module, "_timeframe_to_timedelta", None)
    assert parse_timeframe("4h") == pd.Timedelta(hours=4)
    assert parse_timeframe("15m") == pd.Timedelta(minutes=15)
    assert parse_timeframe("1d") == pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="Timeframe invalide"):
        parse_timeframe("1M")
    with pytest.raises(ValueError, match="Timeframe invalide"):
        parse_timeframe("0h")


def test_normalise_ohlcv_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner_module, "_ensure_ohlcv", None)
    idx = pd.DatetimeIndex(["2024-01-01 01:00", "2024-01-01 00:00", "2024-01-01 01:00"])
    raw = pd.DataFrame(
        {"Open": [1, 2, 3], "High": [2, 3, 4], "Low": [0, 1, 2], "Close": [1, 2, 3.5], "Volume": 1},
        index=idx,
    )
    out = normalise_ohlcv(raw)
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert str(out.index.tz) == "UTC"
    assert out.index.is_monotonic_increasing and out.index.is_unique
    assert out["close"].tolist() == [2.0, 3.5]  # doublon : dernière occurrence gardée
    assert out["open"].dtype == np.float64
    with pytest.raises(ValueError, match="manquantes"):
        normalise_ohlcv(raw.drop(columns=["Close"]))
    with pytest.raises(ValueError, match="DatetimeIndex"):
        normalise_ohlcv(raw.reset_index(drop=True))


def test_position_record_roundtrip() -> None:
    rec = PositionRecord(1, 2.5, 101.0, 95.0, None, "2024-01-01T00:00:00+00:00")
    assert PositionRecord.from_dict(rec.to_dict()) == rec
    assert PositionRecord.from_dict({"side": 0, "amount": 3}) == PositionRecord()
    assert PositionRecord.from_dict({"side": -1, "amount": -2, "stop_loss": NAN}) == PositionRecord(
        side=-1, amount=2.0
    )
    assert PositionRecord.from_dict("n'importe quoi") == PositionRecord()
    with pytest.raises(ValueError, match="side"):
        PositionRecord.from_dict({"side": 2})


def test_constructor_validation(tmp_path: Path, candles: pd.DataFrame) -> None:
    kwargs: dict[str, Any] = {
        "strategy": ScriptedStrategy(),
        "broker": make_broker(),
        "symbol": BTC,
        "timeframe": "1h",
        "risk": make_risk(),
        "data_source": FakeDataSource(candles),
        "state_dir": tmp_path / "s",
    }
    with pytest.raises(ValueError, match="lookback"):
        LiveTrader(**{**kwargs, "lookback": 0})
    with pytest.raises(ValueError, match="Timeframe"):
        LiveTrader(**{**kwargs, "timeframe": "1x"})
    with pytest.raises(ValueError, match="symbole"):
        LiveTrader(**{**kwargs, "symbol": "  "})
    with pytest.raises(TypeError, match="data_source"):
        LiveTrader(**{**kwargs, "data_source": candles})
    trader = LiveTrader(**kwargs)
    assert trader.quote == "USDT"
    assert "BTC/USDT" in repr(trader)
    assert not (tmp_path / "s").exists()  # créé seulement à la première écriture


def test_default_clock_is_utc_now(tmp_path: Path, candles: pd.DataFrame) -> None:
    s = Setup(tmp_path, candles, {})
    before = datetime.now(UTC)
    now = s.trader._clock()
    assert now.tzinfo is not None
    assert now >= before
