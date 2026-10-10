"""Tests du moteur de backtest : petites séries construites à la main, résultats calculés
à la main (frais, slippage, stops, kill switch...)."""

from __future__ import annotations

import json
import logging
import math
import time
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest
from conftest import make_ohlcv

from tradingbot.backtest import metrics as metrics_mod
from tradingbot.backtest.engine import (
    TRADE_COLUMNS,
    VOL_WINDOW,
    BacktestConfig,
    BacktestResult,
    Trade,
    _fallback_periods_per_year,
    _fallback_validate_signals,
    _minimal_ensure_ohlcv,
    _realized_vol,
    run_backtest,
)
from tradingbot.data.synthetic import generate_ohlcv
from tradingbot.risk import RiskConfig, RiskManager
from tradingbot.strategies import get_strategy

try:
    from tradingbot.strategies.base import Strategy
except ImportError:  # le paquet strategies peut être incomplet pendant le développement

    class Strategy(ABC):  # type: ignore[no-redef]
        """Doublure minimale du contrat de ``tradingbot.strategies.base.Strategy``."""

        name: ClassVar[str] = "base"
        description: ClassVar[str] = ""
        default_params: ClassVar[dict[str, Any]] = {}

        def __init__(self, **params: Any) -> None:
            unknown = set(params) - set(self.default_params)
            if unknown:
                raise ValueError(f"Paramètres inconnus : {sorted(unknown)}")
            self.params: dict[str, Any] = {**self.default_params, **params}

        @abstractmethod
        def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame: ...

        def warmup_bars(self) -> int:
            return 0

        def empty_signals(self, df: pd.DataFrame) -> pd.DataFrame:
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
            return _fallback_validate_signals(df, signals)


CASH = 10_000.0
FEE = 0.001
SLIP = 0.001
NAN = math.nan


class ListStrategy(Strategy):
    """Stratégie de test : signaux, stops et objectifs donnés explicitement."""

    name = "liste"
    description = "signaux fournis explicitement (tests)"
    default_params = {"signals": (), "stops": None, "takes": None, "warmup": 0}

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self.empty_signals(df)
        signals = list(self.params["signals"])
        assert len(signals) == len(df), "autant de signaux que de bougies"
        out["signal"] = np.asarray(signals, dtype=np.int64)
        if self.params["stops"] is not None:
            out["stop_loss"] = np.asarray(self.params["stops"], dtype=np.float64)
        if self.params["takes"] is not None:
            out["take_profit"] = np.asarray(self.params["takes"], dtype=np.float64)
        return out

    def warmup_bars(self) -> int:
        return int(self.params["warmup"])


def make_df(
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    start: str = "2024-01-01",
    freq: str = "1h",
) -> pd.DataFrame:
    n = len(opens)
    assert len(highs) == len(lows) == len(closes) == n
    index = pd.date_range(start=start, periods=n, freq=freq, tz="UTC")
    df = pd.DataFrame(
        {
            "open": np.asarray(opens, dtype=float),
            "high": np.asarray(highs, dtype=float),
            "low": np.asarray(lows, dtype=float),
            "close": np.asarray(closes, dtype=float),
            "volume": np.ones(n),
        },
        index=index,
    )
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    return df


def no_guard_risk(**overrides: Any) -> RiskManager:
    """Risque sans kill switch ni limite journalière (sizing seul)."""
    kwargs: dict[str, Any] = {
        "risk_per_trade": 0.01,
        "max_position_fraction": 0.95,
        "max_drawdown": None,
        "daily_loss_limit": None,
        "min_trade_notional": 10.0,
    }
    kwargs.update(overrides)
    return RiskManager(RiskConfig(**kwargs))


def max_fraction_risk(**overrides: Any) -> RiskManager:
    """Risque sans garde-fous dont le sizing sature toujours ``max_position_fraction`` :
    une position ne s'ouvre jamais sans stop, alors on prend un stop de dernier recours
    très loin (99 % du prix) et ``risk_per_trade = 1`` → taille = 95 % de l'équité."""
    kwargs: dict[str, Any] = {"risk_per_trade": 1.0, "default_stop_pct": 0.99}
    kwargs.update(overrides)
    return no_guard_risk(**kwargs)


def cfg(**overrides: Any) -> BacktestConfig:
    kwargs: dict[str, Any] = {"initial_cash": CASH, "fee_rate": FEE, "slippage": SLIP}
    kwargs.update(overrides)
    return BacktestConfig(**kwargs)


def assert_series(actual: pd.Series, expected: list[float]) -> None:
    np.testing.assert_allclose(
        actual.to_numpy(dtype=float), np.asarray(expected, float), rtol=1e-12
    )


# ============================================================================ long simple
def test_long_entry_and_exit_on_signal() -> None:
    df = make_df(
        opens=[100, 100, 110, 120, 115, 118],
        highs=[101, 106, 116, 121, 117, 120],
        lows=[99, 99, 109, 117, 114, 117],
        closes=[100, 105, 115, 118, 116, 119],
    )
    strat = ListStrategy(signals=[0, 1, 1, 0, 0, 0])
    res = run_backtest(df, strat, cfg(), max_fraction_risk())

    # Entrée à open[2] = 110 (signal de la bougie 1), slippage 0,1 % : 110,11.
    entry = 110 * (1 + SLIP)
    size = CASH * 0.95 / entry  # sizing saturé : fraction max ; notionnel = 9 500 exactement
    fee1 = size * entry * FEE  # = 9,5
    assert fee1 == pytest.approx(9.5)
    cash_after_entry = CASH - size * entry - fee1  # 490,5
    assert cash_after_entry == pytest.approx(490.5)
    # Sortie à open[4] = 115 (signal 0 à la bougie 3) : 114,885.
    exit_px = 115 * (1 - SLIP)
    fee2 = size * exit_px * FEE
    cash_final = cash_after_entry + size * exit_px - fee2

    assert_series(
        res.equity,
        [
            CASH,
            CASH,
            cash_after_entry + size * 115,
            cash_after_entry + size * 118,
            cash_final,
            cash_final,
        ],
    )
    assert list(res.positions) == [0, 0, 1, 1, 0, 0]
    assert len(res.trades) == 1
    tr = res.trades.iloc[0]
    assert tr["side"] == 1
    assert tr["entry_time"] == df.index[2]
    assert tr["exit_time"] == df.index[4]
    assert tr["entry_price"] == pytest.approx(entry)
    assert tr["exit_price"] == pytest.approx(exit_px)
    assert tr["size"] == pytest.approx(size)
    assert tr["fees"] == pytest.approx(fee1 + fee2)
    assert tr["pnl"] == pytest.approx(size * (exit_px - entry) - fee1 - fee2)
    assert tr["pnl"] == pytest.approx(cash_final - CASH)  # cohérence cash / pnl
    assert tr["pnl_pct"] == pytest.approx(tr["pnl"] / 9500.0)
    assert tr["bars_held"] == 2
    assert tr["exit_reason"] == "signal"
    # Valeurs littérales (calculées à la main) pour fixer les idées.
    assert size == pytest.approx(86.277359, abs=1e-6)
    assert cash_final == pytest.approx(10392.5624, abs=1e-3)
    assert res.metrics["n_trades"] == 1
    assert res.metrics["final_equity"] == pytest.approx(cash_final)
    assert res.metrics["total_return"] == pytest.approx(cash_final / CASH - 1)


# ============================================================================ stops
def _stop_df(open3: float = 99.0, high3: float = 101.0, low3: float = 94.0) -> pd.DataFrame:
    return make_df(
        opens=[100, 100, 100, open3, 98, 97],
        highs=[101, 101, 103, high3, 99, 98],
        lows=[99, 99, 97, low3, 96, 95],
        closes=[100, 100, 102, min(high3, max(low3, 96)), 97, 96],
    )


def test_stop_loss_hit_intrabar_exits_at_stop_minus_slippage() -> None:
    df = _stop_df()
    strat = ListStrategy(signals=[0, 1, 1, 0, 0, 0], stops=[NAN, 95, NAN, NAN, NAN, NAN])
    res = run_backtest(df, strat, cfg(), no_guard_risk())

    entry = 100 * (1 + SLIP)  # 100,1
    size = CASH * 0.01 / (entry - 95)  # risque 1 % : 100 / 5,1
    assert size == pytest.approx(100 / 5.1)
    assert size * entry < CASH * 0.95  # le plafond ne mord pas
    fee1 = size * entry * FEE
    cash1 = CASH - size * entry - fee1
    # Bougie 2 : low 97 > 95, pas de sortie. Bougie 3 : open 99 > 95, low 94 <= 95 → stop.
    exit_px = 95 * (1 - SLIP)  # 94,905
    fee2 = size * exit_px * FEE
    cash2 = cash1 + size * exit_px - fee2

    assert_series(res.equity, [CASH, CASH, cash1 + size * 102, cash2, cash2, cash2])
    assert list(res.positions) == [0, 0, 1, 1, 0, 0]
    tr = res.trades.iloc[0]
    assert tr["exit_reason"] == "stop_loss"
    assert tr["exit_price"] == pytest.approx(exit_px)
    assert tr["exit_time"] == df.index[3]
    assert tr["bars_held"] == 2
    expected_pnl = size * (exit_px - entry) - fee1 - fee2
    assert tr["pnl"] == pytest.approx(expected_pnl)
    # La perte « au stop » vaut exactement 1 % du capital, plus slippage et frais.
    assert size * (entry - 95) == pytest.approx(100.0)
    assert expected_pnl == pytest.approx(-100.0 - size * 95 * SLIP - fee1 - fee2)


def test_gap_below_stop_exits_at_open() -> None:
    df = make_df(
        opens=[100, 100, 100, 90, 91, 92],
        highs=[101, 101, 103, 92, 93, 93],
        lows=[99, 99, 97, 88, 90, 91],
        closes=[100, 100, 102, 91, 92, 92],
    )
    strat = ListStrategy(signals=[0, 1, 1, 0, 0, 0], stops=[NAN, 95, NAN, NAN, NAN, NAN])
    res = run_backtest(df, strat, cfg(), no_guard_risk())

    entry = 100 * (1 + SLIP)
    size = CASH * 0.01 / (entry - 95)
    fee1 = size * entry * FEE
    exit_px = 90 * (1 - SLIP)  # gap : sortie à l'ouverture, pas au stop
    fee2 = size * exit_px * FEE
    cash2 = CASH - size * entry - fee1 + size * exit_px - fee2
    tr = res.trades.iloc[0]
    assert tr["exit_reason"] == "stop_loss"
    assert tr["exit_price"] == pytest.approx(exit_px)
    assert res.equity.iloc[3] == pytest.approx(cash2)
    assert res.equity.iloc[-1] == pytest.approx(cash2)


def test_take_profit_hit_intrabar() -> None:
    df = make_df(
        opens=[100, 100, 100, 104, 108, 107],
        highs=[101, 101, 103, 112, 109, 108],
        lows=[99, 99, 99, 103, 106, 106],
        closes=[100, 100, 102, 109, 107, 107],
    )
    strat = ListStrategy(signals=[0, 1, 1, 0, 0, 0], takes=[NAN, 110, NAN, NAN, NAN, NAN])
    res = run_backtest(df, strat, cfg(), max_fraction_risk())

    entry = 100 * (1 + SLIP)
    size = CASH * 0.95 / entry  # sizing saturé → fraction max
    fee1 = 9.5
    cash1 = CASH - 9500.0 - fee1  # 490,5
    exit_px = 110 * (1 - SLIP)  # 109,89 (open 104 < 110, high 112 >= 110)
    fee2 = size * exit_px * FEE
    cash2 = cash1 + size * exit_px - fee2
    tr = res.trades.iloc[0]
    assert tr["exit_reason"] == "take_profit"
    assert tr["exit_price"] == pytest.approx(exit_px)
    assert tr["bars_held"] == 2
    assert_series(res.equity, [CASH, CASH, cash1 + size * 102, cash2, cash2, cash2])
    assert tr["pnl"] == pytest.approx(cash2 - CASH)
    assert tr["pnl"] > 0


def test_stop_and_take_profit_same_bar_stop_wins() -> None:
    df = make_df(
        opens=[100, 100, 100, 100, 98, 97],
        highs=[101, 101, 103, 112, 99, 98],
        lows=[99, 99, 97, 94, 96, 95],
        closes=[100, 100, 102, 96, 97, 96],
    )
    strat = ListStrategy(
        signals=[0, 1, 1, 0, 0, 0],
        stops=[NAN, 95, NAN, NAN, NAN, NAN],
        takes=[NAN, 110, NAN, NAN, NAN, NAN],
    )
    res = run_backtest(df, strat, cfg(), no_guard_risk())
    tr = res.trades.iloc[0]
    assert tr["exit_reason"] == "stop_loss"
    assert tr["exit_price"] == pytest.approx(95 * (1 - SLIP))
    assert tr["pnl"] < 0


def test_take_profit_gap_above_exits_at_open() -> None:
    df = make_df(
        opens=[100, 100, 100, 115, 114, 113],
        highs=[101, 101, 103, 116, 115, 114],
        lows=[99, 99, 99, 113, 112, 112],
        closes=[100, 100, 102, 114, 113, 113],
    )
    strat = ListStrategy(signals=[0, 1, 1, 0, 0, 0], takes=[NAN, 110, NAN, NAN, NAN, NAN])
    res = run_backtest(df, strat, cfg(slippage=0.0), no_guard_risk())
    tr = res.trades.iloc[0]
    assert tr["exit_reason"] == "take_profit"
    assert tr["exit_price"] == pytest.approx(115.0)  # gap favorable : ouverture


# ============================================================================ shorts
def _short_df() -> pd.DataFrame:
    return make_df(
        opens=[100, 100, 100, 95, 90, 92],
        highs=[101, 101, 101, 97, 93, 93],
        lows=[99, 99, 97, 92, 89, 91],
        closes=[100, 100, 98, 93, 91, 92],
    )


@pytest.mark.parametrize(
    ("config_short", "risk_short"),
    [(False, False), (True, False), (False, True)],
)
def test_allow_short_false_ignores_minus_one(config_short: bool, risk_short: bool) -> None:
    df = _short_df()
    strat = ListStrategy(signals=[0, -1, -1, 0, 0, 0])
    res = run_backtest(
        df, strat, cfg(allow_short=config_short), no_guard_risk(allow_short=risk_short)
    )
    assert len(res.trades) == 0
    assert list(res.trades.columns) == list(TRADE_COLUMNS)
    assert list(res.positions) == [0] * 6
    assert_series(res.equity, [CASH] * 6)
    assert res.metrics["n_trades"] == 0
    assert res.metrics["exposure"] == 0.0


def test_short_trade_when_allowed() -> None:
    df = _short_df()
    strat = ListStrategy(signals=[0, -1, -1, 0, 0, 0])
    res = run_backtest(df, strat, cfg(allow_short=True), max_fraction_risk(allow_short=True))

    entry = 100 * (1 - SLIP)  # vente : 99,9
    size = CASH * 0.95 / entry
    fee1 = 9.5
    cash1 = CASH + 9500.0 - fee1  # 19 490,5 : le produit de la vente entre en cash
    assert cash1 == pytest.approx(19490.5)
    exit_px = 90 * (1 + SLIP)  # rachat à open[4] : 90,09
    fee2 = size * exit_px * FEE
    cash2 = cash1 - size * exit_px - fee2

    assert_series(res.equity, [CASH, CASH, cash1 - size * 98, cash1 - size * 93, cash2, cash2])
    assert list(res.positions) == [0, 0, -1, -1, 0, 0]
    tr = res.trades.iloc[0]
    assert tr["side"] == -1
    assert tr["entry_price"] == pytest.approx(entry)
    assert tr["exit_price"] == pytest.approx(exit_px)
    assert tr["pnl"] == pytest.approx(size * (entry - exit_px) - fee1 - fee2)
    assert tr["pnl"] == pytest.approx(cash2 - CASH)
    assert tr["pnl"] > 0
    assert tr["exit_reason"] == "signal"


def test_short_gap_above_stop_and_short_take_profit() -> None:
    df = make_df(
        opens=[100, 100, 100, 108, 107, 106],
        highs=[101, 101, 102, 110, 108, 107],
        lows=[99, 99, 98, 106, 105, 104],
        closes=[100, 100, 99, 107, 106, 105],
    )
    strat = ListStrategy(signals=[0, -1, -1, 0, 0, 0], stops=[NAN, 105, NAN, NAN, NAN, NAN])
    res = run_backtest(
        df, strat, cfg(slippage=0.0, allow_short=True), no_guard_risk(allow_short=True)
    )
    tr = res.trades.iloc[0]
    size = CASH * 0.01 / 5.0  # 20 unités
    assert tr["size"] == pytest.approx(size)
    assert tr["exit_reason"] == "stop_loss"
    assert tr["exit_price"] == pytest.approx(108.0)  # gap au-dessus du stop → ouverture
    fees = size * 100 * FEE + size * 108 * FEE  # 2 + 2,16
    assert tr["pnl"] == pytest.approx(20 * (100 - 108) - fees)
    assert tr["pnl"] == pytest.approx(-164.16)

    df2 = make_df(
        opens=[100, 100, 100, 95, 93, 92],
        highs=[101, 101, 101, 96, 94, 93],
        lows=[99, 99, 97, 89, 92, 91],
        closes=[100, 100, 98, 92, 93, 92],
    )
    strat2 = ListStrategy(signals=[0, -1, -1, 0, 0, 0], takes=[NAN, 90, NAN, NAN, NAN, NAN])
    res2 = run_backtest(
        df2, strat2, cfg(slippage=0.0, allow_short=True), max_fraction_risk(allow_short=True)
    )
    tr2 = res2.trades.iloc[0]
    assert tr2["exit_reason"] == "take_profit"
    assert tr2["exit_price"] == pytest.approx(90.0)
    assert tr2["size"] == pytest.approx(95.0)
    assert tr2["pnl"] == pytest.approx(95 * 10 - 9.5 - 95 * 90 * FEE)
    assert tr2["pnl"] == pytest.approx(931.95)


def test_reversal_long_to_short_is_two_trades_with_four_fee_legs() -> None:
    df = make_df(
        opens=[100, 100, 100, 110, 105, 105],
        highs=[101, 101, 106, 111, 106, 106],
        lows=[99, 99, 99, 107, 103, 103],
        closes=[100, 100, 105, 108, 104, 104],
    )
    strat = ListStrategy(signals=[0, 1, -1, 0, 0, 0])
    res = run_backtest(df, strat, cfg(allow_short=True), max_fraction_risk(allow_short=True))

    entry1 = 100 * (1 + SLIP)
    size1 = CASH * 0.95 / entry1
    cash1 = CASH - 9500.0 - 9.5
    eq2 = cash1 + size1 * 105
    sell_px = 110 * (1 - SLIP)  # clôture du long ET ouverture du short au même prix
    fee2 = size1 * sell_px * FEE
    cash_a = cash1 + size1 * sell_px - fee2
    size2 = eq2 * 0.95 / sell_px  # taille sur l'équité de la clôture précédente
    fee3 = size2 * sell_px * FEE
    cash_b = cash_a + size2 * sell_px - fee3
    buy_px = 105 * (1 + SLIP)
    fee4 = size2 * buy_px * FEE
    cash_c = cash_b - size2 * buy_px - fee4

    assert list(res.positions) == [0, 0, 1, -1, 0, 0]
    assert_series(res.equity, [CASH, CASH, eq2, cash_b - size2 * 108, cash_c, cash_c])
    assert len(res.trades) == 2
    t1, t2 = res.trades.iloc[0], res.trades.iloc[1]
    assert t1["side"] == 1 and t1["exit_reason"] == "signal"
    assert t1["exit_time"] == df.index[3] and t2["entry_time"] == df.index[3]
    assert t1["fees"] == pytest.approx(9.5 + fee2)
    assert t2["side"] == -1
    assert t2["size"] == pytest.approx(size2)
    assert t2["fees"] == pytest.approx(fee3 + fee4)
    assert t1["pnl"] + t2["pnl"] == pytest.approx(cash_c - CASH)


# ============================================================================ chauffe / fin
def test_warmup_bars_never_trade() -> None:
    df = make_df(
        opens=[100, 101, 102, 103, 104, 105],
        highs=[101, 102, 103, 104, 105, 106],
        lows=[99, 100, 101, 102, 103, 104],
        closes=[101, 102, 103, 104, 105, 106],
    )
    strat = ListStrategy(signals=[1, 1, 1, 1, 1, 1], warmup=3)
    res = run_backtest(df, strat, cfg(), no_guard_risk())
    # Les signaux des bougies 0..2 sont ignorés ; signal[3] → entrée à open[4].
    assert list(res.positions) == [0, 0, 0, 0, 1, 1]
    assert len(res.trades) == 1
    tr = res.trades.iloc[0]
    assert tr["entry_time"] == df.index[4]
    assert tr["entry_price"] == pytest.approx(104 * (1 + SLIP))
    assert tr["exit_reason"] == "end"
    assert tr["bars_held"] == 2


def test_end_close_at_last_close_with_fees_and_slippage() -> None:
    df = make_df(
        opens=[100, 100, 100, 105, 110, 115],
        highs=[101, 101, 106, 111, 116, 121],
        lows=[99, 99, 99, 104, 109, 114],
        closes=[100, 100, 105, 110, 115, 120],
    )
    strat = ListStrategy(signals=[0, 1, 1, 1, 1, 1])
    res = run_backtest(df, strat, cfg(), max_fraction_risk())

    entry = 100 * (1 + SLIP)
    size = CASH * 0.95 / entry
    cash1 = 490.5
    exit_px = 120 * (1 - SLIP)
    fee2 = size * exit_px * FEE
    cash_final = cash1 + size * exit_px - fee2
    assert_series(
        res.equity,
        [CASH, CASH, cash1 + size * 105, cash1 + size * 110, cash1 + size * 115, cash_final],
    )
    assert list(res.positions) == [0, 0, 1, 1, 1, 1]
    tr = res.trades.iloc[0]
    assert tr["exit_reason"] == "end"
    assert tr["exit_time"] == df.index[5]
    assert tr["exit_price"] == pytest.approx(exit_px)
    assert tr["bars_held"] == 4
    assert res.metrics["final_equity"] == pytest.approx(cash_final)
    assert CASH + res.trades["pnl"].sum() == pytest.approx(res.equity.iloc[-1])


# ============================================================================ risque
def test_kill_switch_closes_at_next_open_and_blocks_reentry() -> None:
    df = make_df(
        opens=[100, 100, 100, 90, 85, 86, 87, 88],
        highs=[101, 101, 101, 92, 87, 88, 89, 90],
        lows=[99, 99, 99, 84, 83, 84, 85, 86],
        closes=[100, 100, 100, 85, 86, 87, 88, 89],
    )
    strat = ListStrategy(signals=[0, 1, 1, 1, 1, 1, 1, 1])
    risk = RiskManager(
        RiskConfig(
            max_drawdown=0.10, daily_loss_limit=None, risk_per_trade=1.0, default_stop_pct=0.99
        )
    )
    res = run_backtest(df, strat, cfg(slippage=0.0), risk)

    # Entrée à 100 (stop de dernier recours à 1, sizing saturé) : 95 unités, frais 9,5,
    # cash 490,5. Bougie 3 : équité 490,5 + 95×85 = 8 565,5 soit -14,3 % depuis le plus
    # haut (10 000) > 10 % → kill switch.
    assert res.equity.iloc[2] == pytest.approx(9990.5)
    assert res.equity.iloc[3] == pytest.approx(8565.5)
    # Clôture à open[4] = 85 : 95×85 = 8 075, frais 8,075.
    cash_final = 490.5 + 8075 - 8.075
    assert cash_final == pytest.approx(8557.425)
    assert_series(res.equity, [CASH, CASH, 9990.5, 8565.5] + [cash_final] * 4)
    assert list(res.positions) == [0, 0, 1, 1, 0, 0, 0, 0]
    assert len(res.trades) == 1
    tr = res.trades.iloc[0]
    assert tr["exit_reason"] == "kill_switch"
    assert tr["exit_price"] == pytest.approx(85.0)
    assert tr["exit_time"] == df.index[4]
    assert tr["pnl"] == pytest.approx(-1442.575)
    assert risk.kill_switch_active()
    # Événement visible : déclenché à la clôture de la bougie 3, équité 8 565,5.
    assert len(res.events) == 1
    ev = res.events[0]
    assert ev["type"] == "kill_switch"
    assert ev["timestamp"] == df.index[3].isoformat()
    assert ev["equity"] == pytest.approx(8565.5)
    assert ev["drawdown"] == pytest.approx(1 - 8565.5 / 10_000)
    assert "drawdown 14.3 %" in ev["detail"] and "10 %" in ev["detail"]
    assert res.metrics["kill_switch_triggered"] == 1
    assert res.metrics["n_daily_limit_hits"] == 0
    assert "kill switch le 2024-01-01 03:00" in res.summary()
    json.dumps(res.events)


def test_daily_limit_closes_blocks_then_resets_next_utc_day() -> None:
    # Bougies horaires de 20:00 à 05:00 le lendemain (changement de jour à l'indice 4).
    df = make_df(
        opens=[100, 100, 100, 93, 94, 95, 96, 97, 98, 99],
        highs=[101, 101, 101, 95, 96, 97, 98, 99, 100, 101],
        lows=[99, 99, 92, 92, 93, 94, 95, 96, 97, 98],
        closes=[100, 100, 93, 94, 95, 96, 97, 98, 99, 100],
        start="2024-01-01 20:00",
    )
    assert df.index[3].date() != df.index[4].date()
    strat = ListStrategy(signals=[1] * 10)
    risk = RiskManager(
        RiskConfig(
            max_drawdown=None, daily_loss_limit=0.05, risk_per_trade=1.0, default_stop_pct=0.99
        )
    )
    # Politique « immediate » : le signal persistant ré-ouvre dès que la limite est levée.
    res = run_backtest(df, strat, cfg(slippage=0.0, reentry_after_stop="immediate"), risk)

    # Entrée à open[1] = 100 : 95 unités (sizing saturé), cash 490,5. Bougie 2 : équité
    # 490,5 + 95×93 = 9 325,5 → -6,7 % sur la journée > 5 % → clôture à open[3] = 93.
    assert res.equity.iloc[2] == pytest.approx(9325.5)
    cash_after = 490.5 + 95 * 93 - 95 * 93 * FEE  # 9 316,665
    assert cash_after == pytest.approx(9316.665)
    assert res.equity.iloc[3] == pytest.approx(cash_after)
    # Clôturé à open[3] ; bloqué le reste du jour (la bougie 4 est décidée à 23:00) ;
    # le jour 2 commence à la bougie 4 → réentrée à open[5].
    assert list(res.positions) == [0, 1, 1, 0, 0, 1, 1, 1, 1, 1]
    assert list(res.trades["exit_reason"]) == ["daily_limit", "end"]
    t1, t2 = res.trades.iloc[0], res.trades.iloc[1]
    assert t1["exit_time"] == df.index[3]
    assert t1["exit_price"] == pytest.approx(93.0)
    assert t2["entry_time"] == df.index[5]
    assert t2["entry_price"] == pytest.approx(95.0)
    # Nouvelle équité de référence le jour 2 = dernière équité du jour 1.
    assert risk.day_start_equity == pytest.approx(cash_after)
    assert res.equity.iloc[4] == pytest.approx(cash_after)
    # Événement : un seul jour touché, consigné à la clôture de la bougie 2.
    assert res.daily_limit_active is True
    assert [e["type"] for e in res.events] == ["daily_limit"]
    ev = res.events[0]
    assert ev["timestamp"] == df.index[2].isoformat()
    assert ev["equity"] == pytest.approx(9325.5)
    assert "perte journalière 6.7 %" in ev["detail"] and "5 %" in ev["detail"]
    assert res.metrics["n_daily_limit_hits"] == 1
    assert res.metrics["kill_switch_triggered"] == 0
    assert "limite journalière atteinte 1 jour(s)" in res.summary()


def test_daily_limit_exit_blocks_same_side_until_signal_leaves_with_wait_policy() -> None:
    df = make_df(
        opens=[100, 100, 100, 93, 94, 95, 96, 97, 98, 99],
        highs=[101, 101, 101, 95, 96, 97, 98, 99, 100, 101],
        lows=[99, 99, 92, 92, 93, 94, 95, 96, 97, 98],
        closes=[100, 100, 93, 94, 95, 96, 97, 98, 99, 100],
        start="2024-01-01 20:00",
    )
    strat = ListStrategy(signals=[1] * 10)
    risk = RiskManager(
        RiskConfig(
            max_drawdown=None, daily_loss_limit=0.05, risk_per_trade=1.0, default_stop_pct=0.99
        )
    )
    res = run_backtest(df, strat, cfg(slippage=0.0), risk)  # reentry_after_stop = "wait"
    # Même sortie « daily_limit » à open[3] ; le signal brut reste à 1 jusqu'à la fin →
    # le côté long reste bloqué : aucune ré-entrée le jour 2.
    assert list(res.positions) == [0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    assert list(res.trades["exit_reason"]) == ["daily_limit"]
    assert res.metrics["n_daily_limit_hits"] == 1


def test_daily_limit_is_inactive_on_daily_or_slower_bars(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Sur des bougies >= 1 jour, chaque bougie est un nouveau jour UTC : la limite
    journalière n'a pas de sens et ne doit jamais se déclencher (M1)."""
    caplog.set_level(logging.INFO, logger="tradingbot.backtest.engine")
    df = make_df(
        opens=[100, 100, 100, 93, 94, 95, 96, 97],
        highs=[101, 101, 101, 95, 96, 97, 98, 99],
        lows=[99, 99, 92, 92, 93, 94, 95, 96],
        closes=[100, 100, 93, 94, 95, 96, 97, 98],
        freq="1D",
    )
    strat = ListStrategy(signals=[1] * 8)
    risk = RiskManager(
        RiskConfig(
            max_drawdown=None, daily_loss_limit=0.05, risk_per_trade=1.0, default_stop_pct=0.99
        )
    )
    res = run_backtest(df, strat, cfg(slippage=0.0), risk)
    # Bougie 2 : -6,7 % en une bougie, mais aucune sortie « daily_limit ».
    assert res.daily_limit_active is False
    assert not risk.daily_limit_active
    assert list(res.positions) == [0, 1, 1, 1, 1, 1, 1, 1]
    assert list(res.trades["exit_reason"]) == ["end"]
    assert res.events == []
    assert res.metrics["n_daily_limit_hits"] == 0
    assert "limite journalière inactive (bougies >= 1 jour)" in res.summary()
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert any(
        m.startswith("limite de perte journalière désactivée : bougies >= 1 jour") for m in messages
    )

    # Le même gestionnaire réutilisé sur des bougies horaires retrouve la limite active.
    hourly = make_df(
        opens=[100, 100, 100, 93, 94, 95, 96, 97],
        highs=[101, 101, 101, 95, 96, 97, 98, 99],
        lows=[99, 99, 92, 92, 93, 94, 95, 96],
        closes=[100, 100, 93, 94, 95, 96, 97, 98],
    )
    res_h = run_backtest(hourly, strat, cfg(slippage=0.0), risk)
    assert res_h.daily_limit_active is True and risk.daily_limit_active
    assert res_h.trades.iloc[0]["exit_reason"] == "daily_limit"


def test_risk_manager_state_is_reset_at_start() -> None:
    df = make_df(
        opens=[100, 100, 100, 101, 102, 103],
        highs=[101, 101, 102, 103, 104, 105],
        lows=[99, 99, 99, 100, 101, 102],
        closes=[100, 100, 101, 102, 103, 104],
    )
    risk = no_guard_risk(max_drawdown=0.1)
    risk.update(pd.Timestamp("2023-01-01", tz="UTC"), 10_000)
    risk.update(pd.Timestamp("2023-01-02", tz="UTC"), 5_000)
    assert risk.kill_switch_active()
    res = run_backtest(df, ListStrategy(signals=[0, 1, 1, 1, 1, 1]), cfg(), risk)
    assert len(res.trades) == 1  # l'état hérité n'a pas bloqué la simulation
    assert not risk.kill_switch_active()


def test_default_risk_manager_when_none_given() -> None:
    df = make_df(
        opens=[100, 100, 100, 101, 102, 103],
        highs=[101, 101, 102, 103, 104, 105],
        lows=[99, 99, 99, 100, 101, 102],
        closes=[100, 100, 101, 102, 103, 104],
    )
    res = run_backtest(df, ListStrategy(signals=[0, 1, 1, 1, 1, 1]), cfg(slippage=0.0))
    # Pas de stop de stratégie, pas de stop_loss_pct, ATR(14) indisponible (6 bougies) →
    # stop de dernier recours 5 % : 95 ; taille = 10 000 × 1 % / (100 - 95) = 20 unités,
    # et non 95 unités (95 % du capital sans stop).
    assert res.trades.iloc[0]["size"] == pytest.approx(20.0)


# ============================================================================ stops moteur
def test_trailing_stop_ratchets_on_highs_of_previous_bars() -> None:
    df = make_df(
        opens=[100, 100, 100, 105, 115, 112, 110],
        highs=[100, 100, 104, 120, 118, 113, 111],
        lows=[100, 100, 99, 104, 110, 105, 100],
        closes=[100, 100, 103, 118, 114, 108, 105],
    )
    strat = ListStrategy(signals=[0, 1, 1, 1, 0, 0, 0])
    res = run_backtest(
        df, strat, cfg(fee_rate=0.0, slippage=0.0, trailing_stop_pct=0.05), max_fraction_risk()
    )
    # Entrée 100, stop 95. Fin bougie 2 : plus haut 104 → stop 98,8. Fin bougie 3 : plus
    # haut 120 → stop 114. Bougie 4 : open 115 > 114, low 110 <= 114 → sortie à 114.
    tr = res.trades.iloc[0]
    assert tr["exit_reason"] == "stop_loss"
    assert tr["exit_price"] == pytest.approx(114.0)
    assert tr["exit_time"] == df.index[4]
    assert tr["size"] == pytest.approx(95.0)  # le stop suiveur ne sert pas au sizing
    assert tr["pnl"] == pytest.approx(95 * 14.0)
    # Sans frais : cash 500 après l'achat de 95 unités à 100.
    assert_series(res.equity, [CASH, CASH, 500 + 95 * 103, 500 + 95 * 118, 11330, 11330, 11330])


def test_strategy_stop_update_while_in_position() -> None:
    df = make_df(
        opens=[100, 100, 100, 101, 100, 100],
        highs=[101, 101, 103, 103, 101, 101],
        lows=[99, 99, 98, 99, 99, 99],
        closes=[100, 100, 102, 100, 100, 100],
    )
    strat = ListStrategy(signals=[0, 1, 1, 1, 0, 0], stops=[NAN, 90, 100, NAN, NAN, NAN])
    res = run_backtest(df, strat, cfg(slippage=0.0), no_guard_risk())
    # Entrée à 100 avec stop 90 : taille 100 / 10 = 10. Le stop passe à 100 à la fin de
    # la bougie 2 ; bougie 3 : low 99 <= 100 → sortie à 100. Frais 1 + 1.
    tr = res.trades.iloc[0]
    assert tr["size"] == pytest.approx(10.0)
    assert tr["exit_price"] == pytest.approx(100.0)
    assert tr["exit_reason"] == "stop_loss"
    assert tr["pnl"] == pytest.approx(-2.0)


def test_strategy_take_profit_update_while_in_position() -> None:
    df = make_df(
        opens=[100, 100, 100, 101, 100, 100],
        highs=[101, 101, 102, 104, 101, 101],
        lows=[99, 99, 99, 100, 99, 99],
        closes=[100, 100, 101, 102, 100, 100],
    )
    strat = ListStrategy(signals=[0, 1, 1, 1, 0, 0], takes=[NAN, NAN, 103, NAN, NAN, NAN])
    res = run_backtest(df, strat, cfg(fee_rate=0.0, slippage=0.0), max_fraction_risk())
    tr = res.trades.iloc[0]
    assert tr["exit_reason"] == "take_profit"
    assert tr["exit_price"] == pytest.approx(103.0)
    assert tr["pnl"] == pytest.approx(95 * 3.0)


def test_default_stop_loss_pct_used_for_sizing_and_exit() -> None:
    df = make_df(
        opens=[100, 100, 100, 99, 98, 98],
        highs=[101, 101, 102, 100, 99, 99],
        lows=[99, 99, 99, 97, 97, 97],
        closes=[100, 100, 101, 98, 98, 98],
    )
    strat = ListStrategy(signals=[0, 1, 1, 0, 0, 0])
    res = run_backtest(
        df, strat, cfg(fee_rate=0.0, slippage=0.0, stop_loss_pct=0.02), no_guard_risk()
    )
    # Stop par défaut 98 → taille = 10 000 × 1 % / 2 = 50 ; low 97 → sortie à 98.
    tr = res.trades.iloc[0]
    assert tr["size"] == pytest.approx(50.0)
    assert tr["exit_price"] == pytest.approx(98.0)
    assert tr["pnl"] == pytest.approx(-100.0)  # exactement risk_per_trade × capital
    assert res.equity.iloc[-1] == pytest.approx(9900.0)


def test_default_take_profit_pct() -> None:
    df = make_df(
        opens=[100, 100, 100, 105, 108, 108],
        highs=[101, 101, 103, 111, 109, 109],
        lows=[99, 99, 99, 104, 107, 107],
        closes=[100, 100, 102, 109, 108, 108],
    )
    strat = ListStrategy(signals=[0, 1, 1, 0, 0, 0])
    res = run_backtest(
        df, strat, cfg(fee_rate=0.0, slippage=0.0, take_profit_pct=0.10), max_fraction_risk()
    )
    tr = res.trades.iloc[0]
    assert tr["exit_reason"] == "take_profit"
    assert tr["exit_price"] == pytest.approx(110.0)
    assert tr["pnl"] == pytest.approx(95 * 10.0)


def test_wait_policy_no_reentry_while_signal_persists_after_stop() -> None:
    """Politique « wait » (défaut) : après un stop, le côté reste bloqué tant que le signal
    brut n'a pas quitté ce côté (F13 / F46 / M3)."""
    df = _stop_df()
    strat = ListStrategy(signals=[0, 1, 1, 1, 0, 0], stops=[NAN, 95, NAN, NAN, NAN, NAN])
    res = run_backtest(df, strat, cfg(slippage=0.0), no_guard_risk())
    # Stop touché bougie 3 ; signal[3] = 1 → PAS de réentrée (côté long bloqué) ;
    # signal[4] = 0 lève le blocage mais ne demande rien.
    assert list(res.positions) == [0, 0, 1, 1, 0, 0]
    assert list(res.trades["exit_reason"]) == ["stop_loss"]


def test_wait_policy_reopens_after_signal_left_the_side() -> None:
    df = make_df(
        opens=[100, 100, 100, 99, 98, 97, 97, 97],
        highs=[101, 101, 103, 101, 99, 98, 98, 98],
        lows=[99, 99, 97, 94, 96, 95, 96, 96],
        closes=[100, 100, 102, 96, 97, 96, 97, 97],
    )
    strat = ListStrategy(
        signals=[0, 1, 1, 1, 1, 0, 1, 1], stops=[NAN, 95, NAN, NAN, NAN, NAN, NAN, NAN]
    )
    res = run_backtest(df, strat, cfg(fee_rate=0.0, slippage=0.0), no_guard_risk())
    # Stop bougie 3 ; signaux 1 aux bougies 3 et 4 → bloqué ; 0 à la bougie 5 → levée ;
    # 1 à la bougie 6 → nouvelle entrée à open[7] = 97, tenue jusqu'à la fin.
    assert list(res.positions) == [0, 0, 1, 1, 0, 0, 0, 1]
    assert list(res.trades["exit_reason"]) == ["stop_loss", "end"]
    t1, t2 = res.trades.iloc[0], res.trades.iloc[1]
    # Premier trade : 20 unités (100 / 5), stop à 95 → -100 exactement (sans frais).
    assert t1["size"] == pytest.approx(20.0)
    assert t1["pnl"] == pytest.approx(-100.0)
    assert t2["entry_time"] == df.index[7]
    assert t2["entry_price"] == pytest.approx(97.0)
    # Deuxième entrée : pas de stop de stratégie, ATR indisponible → stop 5 % (92,15) ;
    # taille = 9 900 × 1 % / (97 − 92,15) = 99 / 4,85.
    assert res.equity.iloc[6] == pytest.approx(9900.0)
    assert t2["size"] == pytest.approx(99.0 / 4.85)


def test_wait_policy_flip_to_other_side_is_a_fresh_entry() -> None:
    df = make_df(
        opens=[100, 100, 100, 99, 98, 97],
        highs=[101, 101, 103, 101, 99, 98],
        lows=[99, 99, 97, 94, 96, 95],
        closes=[100, 100, 102, 96, 97, 96],
    )
    strat = ListStrategy(signals=[0, 1, 1, -1, -1, -1], stops=[NAN, 95, NAN, NAN, NAN, NAN])
    res = run_backtest(
        df,
        strat,
        cfg(fee_rate=0.0, slippage=0.0, allow_short=True),
        no_guard_risk(allow_short=True),
    )
    # Stop long bougie 3 ; le signal brut passe à -1 : le blocage du côté long ne concerne
    # pas le short → ouverture courte à open[4] = 98.
    assert list(res.positions) == [0, 0, 1, 1, -1, -1]
    assert list(res.trades["exit_reason"]) == ["stop_loss", "end"]
    assert res.trades.iloc[1]["side"] == -1
    assert res.trades.iloc[1]["entry_price"] == pytest.approx(98.0)


def test_immediate_policy_reenters_next_bar_with_a_stop() -> None:
    """Ancien comportement sur demande (``reentry_after_stop="immediate"``), mais la
    ré-entrée porte désormais un stop et une taille basée sur le risque."""
    df = _stop_df()
    strat = ListStrategy(signals=[0, 1, 1, 1, 0, 0], stops=[NAN, 95, NAN, NAN, NAN, NAN])
    res = run_backtest(
        df, strat, cfg(slippage=0.0, reentry_after_stop="immediate"), no_guard_risk()
    )
    # Stop touché bougie 3 ; signal[3] = 1 → réentrée à open[4] ; signal[4] = 0 → sortie open[5].
    assert list(res.positions) == [0, 0, 1, 1, 1, 0]
    assert list(res.trades["exit_reason"]) == ["stop_loss", "signal"]
    t2 = res.trades.iloc[1]
    assert t2["entry_time"] == df.index[4]
    assert t2["entry_price"] == pytest.approx(98.0)
    assert t2["exit_time"] == df.index[5]
    assert t2["bars_held"] == 1
    # Deuxième entrée : stop de dernier recours 5 % → 93,1 ; taille = équité × 1 % / 4,9,
    # loin des 95 % de l'équité d'autrefois.
    assert t2["size"] == pytest.approx(res.equity.iloc[3] * 0.01 / (98.0 - 93.1))
    assert t2["size"] * 98.0 < 0.25 * res.equity.iloc[3]


def test_invalid_reentry_policy_raises() -> None:
    df = make_df([100, 100], [101, 101], [99, 99], [100, 100])
    with pytest.raises(ValueError, match="reentry_after_stop"):
        run_backtest(df, ListStrategy(signals=[0, 0]), cfg(reentry_after_stop="toujours"))


# ============================================================================ stop protecteur
def _constant_range_df(n: int = 20) -> pd.DataFrame:
    """open = close = 100, high 101, low 99 : true range 2 à chaque bougie → ATR(14) = 2
    dès la bougie 14."""
    return make_df(opens=[100.0] * n, highs=[101.0] * n, lows=[99.0] * n, closes=[100.0] * n)


def test_entry_without_strategy_stop_gets_atr_default_stop_and_risk_based_size() -> None:
    df = _constant_range_df(20)
    strat = ListStrategy(signals=[0] * 15 + [1] * 5)
    res = run_backtest(df, strat, cfg(fee_rate=0.0, slippage=0.0), no_guard_risk())
    # Entrée à open[16] = 100 ; ATR(14) des bougies <= 15 vaut 2 → stop 100 − 3×2 = 94 ;
    # taille = 10 000 × 1 % / 6.
    tr = res.trades.iloc[0]
    assert tr["entry_time"] == df.index[16]
    assert tr["size"] == pytest.approx(100.0 / 6.0)
    assert tr["exit_reason"] == "end"

    # Le stop est bien actif : un plus bas à 93 (bougie 18) sort à 94, perte = 1 % du capital.
    df2 = df.copy()
    df2.loc[df2.index[18], ["low", "close"]] = [93.0, 95.0]
    res2 = run_backtest(df2, strat, cfg(fee_rate=0.0, slippage=0.0), no_guard_risk())
    tr2 = res2.trades.iloc[0]
    assert tr2["exit_reason"] == "stop_loss"
    assert tr2["exit_time"] == df2.index[18]
    assert tr2["exit_price"] == pytest.approx(94.0)
    assert tr2["pnl"] == pytest.approx(-100.0)


def test_stop_loss_pct_has_priority_over_atr_default_stop() -> None:
    df = _constant_range_df(20)
    strat = ListStrategy(signals=[0] * 15 + [1] * 5)
    res = run_backtest(
        df, strat, cfg(fee_rate=0.0, slippage=0.0, stop_loss_pct=0.02), no_guard_risk()
    )
    # stop_loss_pct 2 % → stop 98 (et non 94 via l'ATR) → taille 100 / 2 = 50.
    assert res.trades.iloc[0]["size"] == pytest.approx(50.0)


def test_strategy_stop_on_wrong_side_is_replaced_and_ignored_while_in_position() -> None:
    df = make_df(
        opens=[100, 100, 100, 100, 100, 100],
        highs=[101, 101, 101, 101, 101, 101],
        lows=[99, 99, 99, 99, 99, 99],
        closes=[100, 100, 100, 100, 100, 100],
    )
    # Stop « porté » par la stratégie à 105, au-dessus du prix d'entrée d'un long : ce
    # n'est pas un stop. À l'entrée il est remplacé (ATR indisponible → 5 % = 95), et ses
    # répétitions en position ne doivent pas forcer une sortie à l'ouverture suivante.
    strat = ListStrategy(signals=[0, 1, 1, 1, 0, 0], stops=[NAN, 105, 105, 105, NAN, NAN])
    res = run_backtest(df, strat, cfg(fee_rate=0.0, slippage=0.0), no_guard_risk())
    # Entrée open[2], tenue bougies 2-4 (signal[4] = 0 → sortie à open[5]) : jamais stoppée.
    assert list(res.positions) == [0, 0, 1, 1, 1, 0]
    tr = res.trades.iloc[0]
    assert tr["exit_reason"] == "signal"
    assert tr["exit_time"] == df.index[5]
    assert tr["size"] == pytest.approx(20.0)  # 100 / (100 − 95)


def test_entries_are_never_naked_for_every_shipped_strategy() -> None:
    """Toute entrée est dimensionnée à partir d'un stop : notionnel <= 95 % de l'équité
    et, quand le plafond ne mord pas, perte au stop = risk_per_trade × équité (M2)."""
    df = generate_ohlcv(n=3000, seed=42)
    for name in ("sma_cross", "ema_trend", "rsi_reversion", "donchian_breakout", "bollinger"):
        full = "bollinger_reversion" if name == "bollinger" else name
        strat = get_strategy(full)
        res = run_backtest(df, strat, cfg(slippage=0.0), no_guard_risk())
        assert len(res.trades) > 0, full
        loc = {ts: i for i, ts in enumerate(df.index)}
        sl = res.signals["stop_loss"].to_numpy()
        for _, tr in res.trades.iterrows():
            t = loc[tr["entry_time"]]
            eq_prev = res.equity.iloc[t - 1]
            notional = tr["size"] * tr["entry_price"]
            assert notional <= eq_prev * 0.95 * (1 + 1e-9), full
            strat_stop = sl[t - 1]
            assert np.isfinite(strat_stop), full  # le stop est porté sur toute la position
            if notional < eq_prev * 0.95 * (1 - 1e-9):
                assert tr["size"] * abs(tr["entry_price"] - strat_stop) == pytest.approx(
                    0.01 * eq_prev, rel=1e-9
                ), full


def test_ema_trend_stop_outs_are_seen_by_the_strategy_no_same_side_reentry_next_bar() -> None:
    """F14 / F48 : la stratégie sort quand le plus bas touche son chandelier, exactement
    comme le moteur. Plus aucune ré-entrée du même côté dès la bougie suivant un stop, même
    avec la politique « immediate » (le signal est à 0 sur la bougie du stop)."""
    df = generate_ohlcv(n=3000, seed=42)
    res = run_backtest(
        df,
        get_strategy("ema_trend"),
        cfg(fee_rate=0.001, slippage=0.0005, reentry_after_stop="immediate"),
        no_guard_risk(),
    )
    tr = res.trades.reset_index(drop=True)
    loc = {ts: i for i, ts in enumerate(df.index)}
    stops = tr[tr["exit_reason"] == "stop_loss"]
    assert len(stops) > 5
    sig = res.signals["signal"].to_numpy()
    for _, row in stops.iterrows():
        assert sig[loc[row["exit_time"]]] != row["side"]  # la stratégie a vu le stop
    next_bar = sum(
        1
        for i in range(1, len(tr))
        if tr.loc[i - 1, "exit_reason"] == "stop_loss"
        and tr.loc[i, "side"] == tr.loc[i - 1, "side"]
        and loc[tr.loc[i, "entry_time"]] == loc[tr.loc[i - 1, "exit_time"]] + 1
    )
    assert next_bar == 0  # avant : 32 ré-entrées immédiates sur 75 trades
    assert len(tr) < 60


# ============================================================================ gaps (F15)
def test_gap_open_beyond_take_profit_wins_over_intrabar_stop() -> None:
    df = make_df(
        opens=[100, 100, 112, 100],
        highs=[101, 101, 130, 101],
        lows=[99, 99, 80, 99],
        closes=[100, 100, 100, 100],
    )
    strat = ListStrategy(signals=[1, 1, 1, 0], stops=[90, 90, 90, NAN], takes=[110, 110, 110, NAN])
    res = run_backtest(df, strat, cfg(fee_rate=0.0, slippage=0.0), no_guard_risk())
    # Entrée open[1] = 100, stop 90, objectif 110 → 10 unités. Bougie 2 : ouverture 112 déjà
    # au-delà de l'objectif → sortie à 112 (le gap lève l'ambiguïté), même si le plus bas
    # (80) passe ensuite sous le stop.
    tr = res.trades.iloc[0]
    assert tr["size"] == pytest.approx(10.0)
    assert tr["exit_reason"] == "take_profit"
    assert tr["exit_price"] == pytest.approx(112.0)
    assert tr["pnl"] == pytest.approx(120.0)

    df_s = make_df(
        opens=[100, 100, 88, 100],
        highs=[101, 101, 120, 101],
        lows=[99, 99, 60, 99],
        closes=[100, 100, 100, 100],
    )
    strat_s = ListStrategy(
        signals=[-1, -1, -1, 0], stops=[110, 110, 110, NAN], takes=[90, 90, 90, NAN]
    )
    res_s = run_backtest(
        df_s,
        strat_s,
        cfg(fee_rate=0.0, slippage=0.0, allow_short=True),
        no_guard_risk(allow_short=True),
    )
    tr_s = res_s.trades.iloc[0]
    assert tr_s["exit_reason"] == "take_profit"
    assert tr_s["exit_price"] == pytest.approx(88.0)
    assert tr_s["pnl"] == pytest.approx(120.0)


def test_min_notional_prevents_trade() -> None:
    df = make_df(
        opens=[100, 100, 100, 101, 102, 103],
        highs=[101, 101, 102, 103, 104, 105],
        lows=[99, 99, 99, 100, 101, 102],
        closes=[100, 100, 101, 102, 103, 104],
    )
    res = run_backtest(
        df, ListStrategy(signals=[0, 1, 1, 1, 1, 1]), cfg(initial_cash=5.0), no_guard_risk()
    )
    assert len(res.trades) == 0
    assert_series(res.equity, [5.0] * 6)


# ============================================================================ look-ahead
def test_signal_at_t_executes_at_open_t_plus_1_against_manual_computation() -> None:
    opens = [100, 102, 104, 106, 108, 110, 112, 114]
    df = make_df(
        opens=opens,
        highs=[o + 3 for o in opens],
        lows=[o - 1 for o in opens],
        closes=[o + 2 for o in opens],
    )
    signals = [0, 0, 0, 1, 0, 0, 0, 0]
    res = run_backtest(df, ListStrategy(signals=signals), cfg(), max_fraction_risk())

    # Calcul manuel : signal[3] → achat à open[4] = 108 ; signal[4] = 0 → vente à open[5].
    entry = 108 * (1 + SLIP)
    size = CASH * 0.95 / entry
    cash1 = CASH - size * entry - size * entry * FEE
    exit_px = 110 * (1 - SLIP)
    cash2 = cash1 + size * exit_px - size * exit_px * FEE
    manual_equity = [CASH, CASH, CASH, CASH, cash1 + size * 110, cash2, cash2, cash2]
    assert_series(res.equity, manual_equity)
    assert list(res.positions) == [0, 0, 0, 0, 1, 0, 0, 0]
    tr = res.trades.iloc[0]
    assert tr["entry_time"] == df.index[4]
    assert tr["entry_price"] == pytest.approx(entry)

    # Modifier les bougies futures (>= 5) ne change ni l'entrée ni l'équité avant.
    df2 = df.copy()
    df2.loc[df2.index[5:], ["open", "high", "low", "close"]] *= 0.5
    res2 = run_backtest(df2, ListStrategy(signals=signals), cfg(), max_fraction_risk())
    assert_series(res2.equity.iloc[:5], manual_equity[:5])
    tr2 = res2.trades.iloc[0]
    assert tr2["entry_price"] == pytest.approx(entry)
    assert tr2["size"] == pytest.approx(size)
    assert tr2["exit_price"] == pytest.approx(55 * (1 - SLIP))


def test_result_series_share_df_index_and_shapes() -> None:
    df = make_df(
        opens=[100, 100, 100, 101, 102, 103],
        highs=[101, 101, 102, 103, 104, 105],
        lows=[99, 99, 99, 100, 101, 102],
        closes=[100, 100, 101, 102, 103, 104],
    )
    res = run_backtest(df, ListStrategy(signals=[0, 1, 0, 1, 0, 0]), cfg(), no_guard_risk())
    assert isinstance(res, BacktestResult)
    assert res.equity.index.equals(df.index)
    assert res.returns.index.equals(df.index)
    assert res.positions.index.equals(df.index)
    assert res.signals.index.equals(df.index)
    assert list(res.signals.columns) == ["signal", "stop_loss", "take_profit"]
    pd.testing.assert_series_equal(
        res.returns, res.equity.pct_change().fillna(0.0), check_names=False
    )
    assert res.returns.iloc[0] == 0.0
    assert res.positions.dtype == np.int64
    assert list(res.trades.columns) == list(TRADE_COLUMNS)
    assert res.trades["side"].dtype == np.int64
    assert res.strategy_name == "liste"
    assert res.params["signals"] == [0, 1, 0, 1, 0, 0]
    assert res.config.initial_cash == CASH
    assert res.periods_per_year == pytest.approx(8760.0)
    # Métriques : floats/ints Python, sérialisables en JSON.
    assert all(isinstance(v, float | int) for v in res.metrics.values())
    json.dumps(res.metrics)
    assert res.metrics["n_trades"] == 2
    # Buy & hold net : achat à open[0] = 100 (chauffe nulle) avec frais + slippage, revente
    # à la dernière clôture 104 avec frais + slippage.
    expected_bh = 104 * (1 - SLIP) * (1 - FEE) / (100 * (1 + SLIP) * (1 + FEE)) - 1
    assert res.metrics["buy_hold_return"] == pytest.approx(expected_bh)
    assert res.metrics["exposure"] == pytest.approx(2 / 6)
    assert res.metrics["kill_switch_triggered"] == 0
    assert res.metrics["n_daily_limit_hits"] == 0
    assert isinstance(res.metrics["kill_switch_triggered"], int)
    assert res.events == []
    assert res.warmup_bars == 0
    assert res.daily_limit_active is True


def test_buy_hold_benchmark_starts_where_the_strategy_can_trade() -> None:
    """F54 : le buy & hold de référence part de open[warmup + 1] (première exécution
    possible) et paie un aller-retour de frais + slippage, comme la stratégie."""
    opens = [100, 102, 104, 106, 108, 110, 112, 114]
    df = make_df(
        opens=opens,
        highs=[o + 3 for o in opens],
        lows=[o - 1 for o in opens],
        closes=[o + 2 for o in opens],
    )
    res = run_backtest(df, ListStrategy(signals=[1] * 8, warmup=3), cfg(), no_guard_risk())
    assert res.warmup_bars == 3
    expected = 116 * (1 - SLIP) * (1 - FEE) / (108 * (1 + SLIP) * (1 + FEE)) - 1
    assert res.metrics["buy_hold_return"] == pytest.approx(expected)
    # Une stratégie « toujours long » après la chauffe fait alors exactement le marché
    # (à la taille près) : signe identique, pas de pénalité due à la chauffe.
    assert res.trades.iloc[0]["entry_time"] == df.index[4]
    assert res.metrics["buy_hold_return"] > 0


def test_summary_is_french_text() -> None:
    df = make_df(
        opens=[100, 100, 100, 101, 102, 103],
        highs=[101, 101, 102, 103, 104, 105],
        lows=[99, 99, 99, 100, 101, 102],
        closes=[100, 100, 101, 102, 103, 104],
    )
    res = run_backtest(df, ListStrategy(signals=[0, 1, 1, 1, 1, 1]), cfg(), no_guard_risk())
    text = res.summary()
    for word in ("Résumé", "Rendement total", "Sharpe", "Drawdown maximal", "Trades", "Buy & hold"):
        assert word in text
    assert "10 000.00" in text
    assert "liste" in text


# ============================================================================ validation
def test_invalid_config_raises() -> None:
    df = make_df([100, 100], [101, 101], [99, 99], [100, 100])
    strat = ListStrategy(signals=[0, 0])
    with pytest.raises(ValueError, match="next_open"):
        run_backtest(df, strat, cfg(execute_on="close"))
    with pytest.raises(ValueError, match="initial_cash"):
        run_backtest(df, strat, cfg(initial_cash=0.0))
    with pytest.raises(ValueError, match="stop_loss_pct"):
        run_backtest(df, strat, cfg(stop_loss_pct=-0.1))


def test_invalid_signals_raise() -> None:
    df = make_df([100, 100, 100], [101, 101, 101], [99, 99, 99], [100, 100, 100])

    class BadStrategy(ListStrategy):
        def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
            out = super().generate_signals(df)
            out["signal"] = np.array([0, 2, 0], dtype=np.int64)
            return out

    with pytest.raises(ValueError, match="signal"):
        run_backtest(df, BadStrategy(signals=[0, 0, 0]), cfg())


def test_duck_typed_strategy_without_validate_signals() -> None:
    df = make_df(
        opens=[100, 100, 100, 101, 102, 103],
        highs=[101, 101, 102, 103, 104, 105],
        lows=[99, 99, 99, 100, 101, 102],
        closes=[100, 100, 101, 102, 103, 104],
    )

    class Duck:
        name = "canard"
        params = {"x": 1}

        def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "signal": [0, 1, 1, 0, 0, 0],
                    "stop_loss": [np.nan] * 6,
                    "take_profit": [np.nan] * 6,
                },
                index=df.index,
            )

    res = run_backtest(df, Duck(), cfg(), no_guard_risk())  # type: ignore[arg-type]
    assert res.strategy_name == "canard"
    assert len(res.trades) == 1
    assert list(res.positions) == [0, 0, 1, 1, 0, 0]


def test_minimal_ensure_ohlcv_normalises_and_validates() -> None:
    naive = pd.date_range("2024-01-01", periods=4, freq="1h")
    raw = pd.DataFrame(
        {
            "Open": [3, 2, 1, 1],
            "High": [4, 3, 2, 2],
            "Low": [2, 1, 0, 0],
            "Close": [3, 2, 1, 1],
            "Volume": [1, 1, 1, 1],
        },
        index=[naive[2], naive[0], naive[1], naive[1]],  # désordonné + doublon
    )
    out = _minimal_ensure_ohlcv(raw)
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert str(out.index.tz) == "UTC"
    assert out.index.is_monotonic_increasing and out.index.is_unique
    assert len(out) == 3
    assert out.index.name == "timestamp"
    assert out["open"].dtype == np.float64
    assert out.loc[naive[1].tz_localize("UTC"), "open"] == 1.0  # keep="last"
    with pytest.raises(ValueError, match="manquantes"):
        _minimal_ensure_ohlcv(raw.drop(columns="Close"))
    with pytest.raises(ValueError, match="DatetimeIndex"):
        _minimal_ensure_ohlcv(raw.reset_index(drop=True))


def test_fallback_periods_per_year() -> None:
    assert _fallback_periods_per_year(
        pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
    ) == pytest.approx(8760.0)
    assert _fallback_periods_per_year(
        pd.date_range("2024-01-01", periods=5, freq="1D", tz="UTC")
    ) == pytest.approx(365.0)
    assert _fallback_periods_per_year(
        pd.date_range("2024-01-01", periods=5, freq="15min", tz="UTC")
    ) == pytest.approx(35040.0)
    assert (
        _fallback_periods_per_year(pd.date_range("2024-01-01", periods=1, freq="1h", tz="UTC"))
        == 365.0
    )


def test_realized_vol_is_backward_looking() -> None:
    n = VOL_WINDOW + 10
    close = 100.0 * 1.01 ** np.arange(n)  # rendements log constants → volatilité nulle
    vol = _realized_vol(close, 8760.0)
    assert all(math.isnan(v) for v in vol[:VOL_WINDOW])
    assert all(abs(v) < 1e-9 for v in vol[VOL_WINDOW:])
    rng = np.random.default_rng(0)
    close2 = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    vol2 = _realized_vol(close2, 8760.0)
    close3 = close2.copy()
    close3[VOL_WINDOW + 5 :] *= 3.0  # perturber le futur ne change pas le passé
    vol3 = _realized_vol(close3, 8760.0)
    np.testing.assert_allclose(vol2[: VOL_WINDOW + 5], vol3[: VOL_WINDOW + 5])


def test_vol_target_reduces_size() -> None:
    df = make_ohlcv(n=120, seed=5, vol=0.05)
    signals = [0] * 40 + [1] * 80
    base = run_backtest(df, ListStrategy(signals=signals), cfg(), no_guard_risk())
    scaled = run_backtest(df, ListStrategy(signals=signals), cfg(), no_guard_risk(vol_target=0.5))
    assert len(base.trades) == 1 and len(scaled.trades) == 1
    assert 0 < scaled.trades.iloc[0]["size"] < base.trades.iloc[0]["size"]


def test_trade_dataclass_fields_match_columns() -> None:
    assert TRADE_COLUMNS == (
        "entry_time",
        "exit_time",
        "side",
        "entry_price",
        "exit_price",
        "size",
        "pnl",
        "pnl_pct",
        "fees",
        "bars_held",
        "exit_reason",
    )
    assert tuple(Trade.__dataclass_fields__) == TRADE_COLUMNS


# ============================================================================ performance
class FastSmaStrategy(Strategy):
    """Signal vectorisé rapide pour le test de performance (beaucoup de trades)."""

    name = "sma_rapide"
    description = "clôture au-dessus / en dessous de sa moyenne mobile"
    default_params = {"window": 20}

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self.empty_signals(df)
        sma = df["close"].rolling(self.params["window"]).mean()
        out["signal"] = np.where(df["close"] > sma, 1, -1).astype(np.int64)
        return out

    def warmup_bars(self) -> int:
        return int(self.params["window"])


def test_performance_50k_bars() -> None:
    df = make_ohlcv(n=50_000, seed=11)
    config = cfg(stop_loss_pct=0.02, take_profit_pct=0.05, trailing_stop_pct=0.03, allow_short=True)
    risk = RiskManager(RiskConfig(allow_short=True, max_drawdown=None, daily_loss_limit=None))
    t0 = time.perf_counter()
    res = run_backtest(df, FastSmaStrategy(), config, risk)
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, f"backtest trop lent : {elapsed:.2f} s"
    assert len(res.equity) == 50_000
    assert len(res.trades) > 100
    assert set(res.trades["exit_reason"]) <= {
        "signal",
        "stop_loss",
        "take_profit",
        "end",
        "kill_switch",
        "daily_limit",
    }
    # Cohérence comptable : capital final = capital initial + somme des pnl nets.
    assert res.equity.iloc[-1] == pytest.approx(CASH + res.trades["pnl"].sum(), rel=1e-9)
    assert res.metrics["n_trades"] == len(res.trades)
    assert metrics_mod.exposure(res.positions) == pytest.approx(res.metrics["exposure"])
