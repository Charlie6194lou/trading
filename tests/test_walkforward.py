"""Tests de la recherche sur grille et du walk-forward.

Les stratégies utilisées sont définies ici (indépendantes des autres modules de
stratégies) ; les bornes de fenêtres et l'équité OOS sont vérifiées à la main.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest
from conftest import make_ohlcv

from tradingbot.backtest.engine import BacktestConfig, run_backtest
from tradingbot.backtest.metrics import METRIC_KEYS, compute_metrics
from tradingbot.backtest.walkforward import (
    WINDOW_KEYS,
    WalkForwardResult,
    _finite_mean,
    _overfit_ratio,
    _sort_table,
    default_windows,
    grid_search,
    walk_forward,
)
from tradingbot.risk import RiskConfig, RiskManager

try:
    from tradingbot.strategies.base import Strategy
except ImportError:  # le paquet strategies peut être incomplet pendant le développement

    class Strategy(ABC):  # type: ignore[no-redef]
        """Doublure minimale du contrat de ``tradingbot.strategies.base.Strategy``."""

        name: ClassVar[str] = "base"
        description: ClassVar[str] = ""
        default_params: ClassVar[dict[str, Any]] = {}
        param_grid: ClassVar[dict[str, list[Any]]] = {}

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


LOGGER = "tradingbot.backtest.walkforward"
CASH = 10_000.0


# ============================================================================ stratégies de test
class SmaCrossTest(Strategy):
    """Croisement de moyennes mobiles simple (pandas rolling, aucun look-ahead)."""

    name = "sma_test"
    description = "croisement SMA pour les tests"
    default_params = {"fast": 5, "slow": 20}
    param_grid = {"fast": [3, 5], "slow": [10, 20]}

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self.empty_signals(df)
        close = df["close"]
        fast = close.rolling(int(self.params["fast"])).mean()
        slow = close.rolling(int(self.params["slow"])).mean()
        out["signal"] = (fast > slow).astype(np.int64)
        return out

    def warmup_bars(self) -> int:
        return int(self.params["slow"])


class AlwaysLong(Strategy):
    """Toujours long (signal 1 partout) : l'équité suit le prix."""

    name = "always_long"
    description = "toujours long"
    default_params = {"dummy": 1}
    param_grid = {"dummy": [1, 2]}

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self.empty_signals(df)
        out["signal"] = np.ones(len(df), dtype=np.int64)
        return out


class AlwaysLongW(Strategy):
    """Toujours long, mais avec une période de chauffe paramétrable (F51)."""

    name = "always_long_w"
    description = "toujours long après w bougies de chauffe"
    default_params = {"w": 10}
    param_grid = {"w": [10]}

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self.empty_signals(df)
        out["signal"] = np.ones(len(df), dtype=np.int64)
        return out

    def warmup_bars(self) -> int:
        return int(self.params["w"])


class Recording(Strategy):
    """Ne trade jamais mais mémorise les tranches de bougies reçues."""

    name = "recording"
    description = "enregistre les appels"
    default_params = {"n": 0}
    param_grid = {"n": [0]}
    calls: ClassVar[list[tuple[int, pd.Timestamp, pd.Timestamp]]] = []

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        Recording.calls.append((len(df), df.index[0], df.index[-1]))
        return self.empty_signals(df)

    def warmup_bars(self) -> int:
        return int(self.params["n"])


class Failing(Strategy):
    """Lève une exception pour ``k == 2``."""

    name = "failing"
    description = "échoue pour k=2"
    default_params = {"k": 1}
    param_grid = {"k": [1, 2, 3]}

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        if int(self.params["k"]) == 2:
            raise ValueError("combinaison volontairement cassée")
        out = self.empty_signals(df)
        out["signal"] = np.ones(len(df), dtype=np.int64)
        return out


def frictionless() -> tuple[BacktestConfig, RiskManager]:
    """Pas de frais, pas de glissement, 100 % du capital investi, aucun garde-fou.

    Une position ne s'ouvre jamais sans stop : le stop de dernier recours est placé très
    loin (99 % du prix) avec ``risk_per_trade = 1`` pour que la taille sature toujours
    ``max_position_fraction`` (= tout le capital).
    """
    cfg = BacktestConfig(initial_cash=CASH, fee_rate=0.0, slippage=0.0)
    risk = RiskManager(
        RiskConfig(
            risk_per_trade=1.0,
            max_position_fraction=1.0,
            max_drawdown=None,
            daily_loss_limit=None,
            min_trade_notional=0.0,
            default_stop_pct=0.99,
        )
    )
    return cfg, risk


def staircase(n: int = 12, start: float = 100.0, step: float = 1.0) -> pd.DataFrame:
    """Clôtures 100, 101, 102… (ou décroissantes si ``step < 0``) ; ``open[t] = close[t-1]``
    (pas de gap)."""
    close = start + step * np.arange(n, dtype=float)
    open_ = np.concatenate([[start], close[:-1]])
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    index = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": np.ones(n)},
        index=index,
    )


@pytest.fixture
def bars() -> pd.DataFrame:
    return make_ohlcv(n=300, seed=11)


@pytest.fixture(autouse=True)
def _clear_recording() -> None:
    Recording.calls.clear()


# ============================================================================ grid_search
def test_grid_search_one_row_per_combination_sorted(bars: pd.DataFrame) -> None:
    table = grid_search(bars, SmaCrossTest)
    assert len(table) == 4
    assert list(table.columns[:2]) == ["fast", "slow"]
    for key in METRIC_KEYS:
        assert key in table.columns
    # toutes les combinaisons sont présentes, une seule fois
    combos = {(int(f), int(s)) for f, s in zip(table["fast"], table["slow"], strict=True)}
    assert combos == {(3, 10), (3, 20), (5, 10), (5, 20)}
    # tri décroissant par Sharpe
    sharpe = table["sharpe"].to_numpy()
    assert np.all(np.diff(sharpe) <= 0)
    assert list(table.index) == [0, 1, 2, 3]
    # chaque ligne reproduit le backtest de sa combinaison ; les métriques sont calculées à
    # partir de la fin de la chauffe (F51) : rendement, drawdown et nombre de trades sont
    # identiques à run_backtest, le Sharpe n'est plus dilué par les bougies plates de chauffe
    for _, row in table.iterrows():
        strat = SmaCrossTest(fast=int(row["fast"]), slow=int(row["slow"]))
        res = run_backtest(bars, strat)
        w = strat.warmup_bars()
        assert row["total_return"] == res.metrics["total_return"]
        assert row["max_drawdown"] == res.metrics["max_drawdown"]
        assert row["n_trades"] == res.metrics["n_trades"]
        trimmed = compute_metrics(
            res.equity.iloc[w:], res.trades, res.periods_per_year, positions=res.positions.iloc[w:]
        )
        assert row["sharpe"] == pytest.approx(trimmed["sharpe"])
        assert row["exposure"] == pytest.approx(trimmed["exposure"])
        if res.metrics["n_trades"]:
            assert row["sharpe"] != res.metrics["sharpe"]


def test_grid_search_explicit_grid_and_scalar_values(bars: pd.DataFrame) -> None:
    table = grid_search(bars, SmaCrossTest, grid={"fast": 3, "slow": [10, 20]})
    assert len(table) == 2
    assert set(table["fast"].astype(int)) == {3}
    assert set(table["slow"].astype(int)) == {10, 20}


def test_grid_search_empty_grid_evaluates_defaults(bars: pd.DataFrame) -> None:
    table = grid_search(bars, SmaCrossTest, grid={})
    assert len(table) == 1
    res = run_backtest(bars, SmaCrossTest())
    assert table.loc[0, "total_return"] == res.metrics["total_return"]


def test_grid_search_cap_is_respected_and_logged(
    bars: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger=LOGGER)
    table = grid_search(bars, SmaCrossTest, max_combinations=3)
    assert len(table) == 3
    # les 3 premières combinaisons dans l'ordre du produit cartésien
    combos = {(int(f), int(s)) for f, s in zip(table["fast"], table["slow"], strict=True)}
    assert combos == {(3, 10), (3, 20), (5, 10)}
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("plafonnée à 3" in m and "1 combinaisons ignorées" in m for m in messages)


def test_grid_search_no_cap_without_warning(
    bars: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger=LOGGER)
    table = grid_search(bars, SmaCrossTest, max_combinations=None)
    assert len(table) == 4
    assert not [r for r in caplog.records if r.name == LOGGER and r.levelno >= logging.WARNING]


def test_grid_search_invalid_cap(bars: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="max_combinations"):
        grid_search(bars, SmaCrossTest, max_combinations=0)


def test_grid_search_skips_failing_combination(
    bars: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger=LOGGER)
    table = grid_search(bars, Failing)
    assert sorted(int(k) for k in table["k"]) == [1, 3]
    messages = [r.getMessage() for r in caplog.records if r.name == LOGGER]
    assert any("ignorée" in m and "'k': 2" in m for m in messages)


def test_grid_search_all_failing_returns_empty_table(bars: pd.DataFrame) -> None:
    table = grid_search(bars, Failing, grid={"k": [2]})
    assert len(table) == 0
    assert "k" in table.columns
    assert "sharpe" in table.columns


def test_grid_search_unknown_metric_warns_and_keeps_order(
    bars: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger=LOGGER)
    table = grid_search(bars, SmaCrossTest, metric="inexistante")
    assert len(table) == 4
    pairs = [(int(f), int(s)) for f, s in zip(table["fast"], table["slow"], strict=True)]
    assert pairs == [(3, 10), (3, 20), (5, 10), (5, 20)]
    assert any("inexistante" in r.getMessage() for r in caplog.records if r.name == LOGGER)


def test_grid_search_unknown_parameter_raises(bars: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="inconnus"):
        grid_search(bars, SmaCrossTest, grid={"nope": [1, 2]})


def test_grid_search_empty_values_raise(bars: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="vide"):
        grid_search(bars, SmaCrossTest, grid={"fast": []})


def test_grid_search_param_named_like_metric_raises(bars: pd.DataFrame) -> None:
    class Clash(AlwaysLong):
        name = "clash"
        default_params = {"sharpe": 1}
        param_grid = {"sharpe": [1]}

    with pytest.raises(ValueError, match="métrique"):
        grid_search(bars, Clash)


def test_sort_table_nan_last_and_params_aligned() -> None:
    table = pd.DataFrame({"a": [1, 2, 3, 4], "sharpe": [0.5, np.nan, 1.0, 0.5]})
    params = [{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}]
    ordered, ranked = _sort_table(table, "sharpe", params)
    assert ordered["a"].tolist() == [3, 1, 4, 2]  # tri stable, NaN en dernier
    assert ranked == [{"a": 3}, {"a": 1}, {"a": 4}, {"a": 2}]
    assert list(ordered.index) == [0, 1, 2, 3]


def test_grid_search_other_metric(bars: pd.DataFrame) -> None:
    table = grid_search(bars, SmaCrossTest, metric="total_return")
    values = table["total_return"].to_numpy()
    assert np.all(np.diff(values) <= 0)


# ============================================================================ default_windows
def test_default_windows_hand_computed() -> None:
    # 300 // (4 + 3) = 42 bougies de test, 300 - 4 × 42 = 132 d'entraînement
    assert default_windows(300) == (132, 42)
    assert default_windows(1000) == (432, 142)
    train, test = default_windows(300, n_windows=5, train_test_ratio=2)
    assert (train, test) == (300 - 5 * 42, 42)


def test_default_windows_gives_exactly_four_windows() -> None:
    df = make_ohlcv(n=300, seed=5)
    train, test = default_windows(300)
    result = walk_forward(df, Recording, train_bars=train, test_bars=test)
    assert result.n_windows == 4
    assert all(w["n_test_bars"] == test for w in result.windows)


def test_default_windows_too_few_bars() -> None:
    with pytest.raises(ValueError, match="Pas assez de bougies"):
        default_windows(5)
    with pytest.raises(ValueError):
        default_windows(300, n_windows=0)


# ============================================================================ walk_forward
def test_walk_forward_window_boundaries_exact(bars: pd.DataFrame) -> None:
    result = walk_forward(bars, Recording, train_bars=100, test_bars=50)
    assert isinstance(result, WalkForwardResult)
    assert result.n_windows == 4
    expected = [(0, 100, 100, 150), (50, 150, 150, 200), (100, 200, 200, 250), (150, 250, 250, 300)]
    got = [
        (w["train_start"], w["train_end"], w["test_start"], w["test_end"]) for w in result.windows
    ]
    assert got == expected
    for w in result.windows:
        for key in WINDOW_KEYS:
            assert key in w
    # horodatages cohérents avec les positions (fin inclusive pour les *_time)
    for w in result.windows:
        assert w["train_start_time"] == bars.index[w["train_start"]]
        assert w["train_end_time"] == bars.index[w["train_end"] - 1]
        assert w["test_start_time"] == bars.index[w["test_start"]]
        assert w["test_end_time"] == bars.index[w["test_end"] - 1]
    # une seule combinaison → chaque fenêtre = 1 backtest d'entraînement + 1 de test
    assert len(Recording.calls) == 8
    for k, (ts, te, ss, se) in enumerate(expected):
        n_train, first, last = Recording.calls[2 * k]
        assert (n_train, first, last) == (100, bars.index[ts], bars.index[te - 1])
        n_test, first, last = Recording.calls[2 * k + 1]
        assert (n_test, first, last) == (50, bars.index[ss], bars.index[se - 1])


def test_walk_forward_oos_equity_covers_all_test_bars(bars: pd.DataFrame) -> None:
    result = walk_forward(bars, Recording, train_bars=100, test_bars=50)
    assert len(result.oos_equity) == 200
    assert result.oos_equity.index.equals(bars.index[100:])
    assert result.oos_equity.index.is_unique
    assert result.oos_equity.index.is_monotonic_increasing
    assert result.oos_equity.iloc[0] == CASH
    # stratégie plate : équité constante, métriques neutres
    assert (result.oos_equity == CASH).all()
    assert result.oos_metrics["n_trades"] == 0
    assert result.oos_metrics["total_return"] == 0.0
    assert len(result.oos_returns) == 200
    assert len(result.oos_positions) == 200
    assert "window" in result.oos_trades.columns
    assert len(result.oos_trades) == 0


def test_walk_forward_prepends_warmup_history(bars: pd.DataFrame) -> None:
    result = walk_forward(bars, Recording, grid={"n": [7]}, train_bars=100, test_bars=50)
    assert result.n_windows == 4
    test_calls = Recording.calls[1::2]
    assert len(test_calls) == 4
    for w, (n_bars, first, last) in zip(result.windows, test_calls, strict=True):
        assert n_bars == 57  # 7 bougies d'historique + 50 de test
        assert first == bars.index[w["test_start"] - 7]
        assert last == bars.index[w["test_end"] - 1]
    # l'équité OOS ne garde que les bougies de test
    assert len(result.oos_equity) == 200
    assert result.oos_equity.index.equals(bars.index[100:])


def test_walk_forward_anchored(bars: pd.DataFrame) -> None:
    result = walk_forward(bars, Recording, train_bars=100, test_bars=50, anchored=True)
    got = [
        (w["train_start"], w["train_end"], w["test_start"], w["test_end"]) for w in result.windows
    ]
    assert got == [(0, 100, 100, 150), (0, 150, 150, 200), (0, 200, 200, 250), (0, 250, 250, 300)]
    assert result.anchored is True
    train_calls = Recording.calls[0::2]
    assert [c[0] for c in train_calls] == [100, 150, 200, 250]
    assert all(c[1] == bars.index[0] for c in train_calls)


def test_walk_forward_truncated_last_window() -> None:
    df = make_ohlcv(n=320, seed=2)
    result = walk_forward(df, Recording, train_bars=100, test_bars=50)
    assert result.n_windows == 5
    last = result.windows[-1]
    assert (last["test_start"], last["test_end"], last["n_test_bars"]) == (300, 320, 20)
    assert len(result.oos_equity) == 220
    assert result.oos_equity.index.equals(df.index[100:])


def test_walk_forward_oos_equity_hand_computed() -> None:
    """12 bougies 100→111, toujours long, sans frais : l'équité OOS suit le prix.

    Fenêtres : test [4, 8) et [8, 12). Dans chaque fenêtre le signal de la première bougie
    s'exécute à l'ouverture de la suivante (= clôture précédente), puis l'équité vaut
    ``cash × close[t] / close[test_start]``. Les deux fenêtres sont enchaînées.
    """
    df = staircase(12)
    cfg, risk = frictionless()
    result = walk_forward(df, AlwaysLong, train_bars=4, test_bars=4, config=cfg, risk=risk)
    assert result.n_windows == 2
    assert [(w["test_start"], w["test_end"]) for w in result.windows] == [(4, 8), (8, 12)]
    c = df["close"].to_numpy()
    w1 = c[7] / c[4]  # 107 / 104
    expected = [
        CASH,
        CASH * c[5] / c[4],
        CASH * c[6] / c[4],
        CASH * w1,
        CASH * w1,  # première bougie de la 2e fenêtre : pas encore en position
        CASH * w1 * c[9] / c[8],
        CASH * w1 * c[10] / c[8],
        CASH * w1 * c[11] / c[8],
    ]
    assert result.oos_equity.index.equals(df.index[4:])
    np.testing.assert_allclose(result.oos_equity.to_numpy(), expected, rtol=1e-12)
    assert result.oos_equity.iloc[0] == CASH
    assert result.oos_metrics["total_return"] == pytest.approx((107 / 104) * (111 / 108) - 1)
    assert result.oos_metrics["final_equity"] == pytest.approx(expected[-1])
    # buy & hold OOS : close[11] / close[4] - 1
    assert result.oos_metrics["buy_hold_return"] == pytest.approx(111 / 104 - 1)
    # un trade par fenêtre, clôturé à la fin de la fenêtre, numéroté
    assert result.oos_trades["window"].tolist() == [1, 2]
    assert result.oos_trades["exit_reason"].tolist() == ["end", "end"]
    assert result.oos_trades["entry_time"].tolist() == [df.index[5], df.index[9]]
    assert result.oos_metrics["n_trades"] == 2
    # rendements OOS : 0 à chaque début de fenêtre, puis rendements du prix
    rets = result.oos_returns.to_numpy()
    assert rets[0] == 0.0
    assert rets[4] == 0.0
    assert rets[1] == pytest.approx(105 / 104 - 1)
    assert rets[7] == pytest.approx(111 / 110 - 1)
    # positions OOS : plat sur la 1re bougie de chaque fenêtre, long ensuite
    assert result.oos_positions.tolist() == [0, 1, 1, 1, 0, 1, 1, 1]


def test_walk_forward_in_sample_metric_matches_grid_search() -> None:
    df = staircase(12)
    cfg, risk = frictionless()
    result = walk_forward(
        df, AlwaysLong, train_bars=4, test_bars=4, config=cfg, risk=risk, metric="total_return"
    )
    # dummy=1 et dummy=2 donnent le même résultat : tri stable → dummy=1 gagne
    assert all(w["best_params"] == {"dummy": 1} for w in result.windows)
    table = grid_search(df.iloc[0:4], AlwaysLong, config=cfg, risk=risk, metric="total_return")
    assert result.windows[0]["is_metric"] == pytest.approx(table.loc[0, "total_return"])
    # entraînement [0, 4) : entrée à open[1] = 100, sortie à close[3] = 103 → +3 %
    assert result.windows[0]["is_metric"] == pytest.approx(0.03)
    # test [4, 8) : 107 / 104 - 1
    assert result.windows[0]["oos_metric"] == pytest.approx(107 / 104 - 1)
    assert result.windows[0]["oos_return"] == pytest.approx(107 / 104 - 1)
    assert result.windows[1]["oos_metric"] == pytest.approx(111 / 108 - 1)
    assert result.is_metrics_mean == pytest.approx((0.03 + (107 / 104 - 1)) / 2)
    assert result.oos_metrics_mean == pytest.approx(((107 / 104 - 1) + (111 / 108 - 1)) / 2)
    assert result.overfit_ratio == pytest.approx(result.oos_metrics_mean / result.is_metrics_mean)
    assert result.metric == "total_return"


def test_walk_forward_with_sma_strategy(bars: pd.DataFrame) -> None:
    result = walk_forward(bars, SmaCrossTest, train_bars=100, test_bars=50)
    assert result.n_windows == 4
    for w in result.windows:
        assert set(w["best_params"]) == {"fast", "slow"}
        assert w["best_params"]["fast"] in (3, 5)
        assert w["best_params"]["slow"] in (10, 20)
        assert isinstance(w["is_metric"], float)
        assert isinstance(w["oos_metric"], float)
    assert len(result.oos_equity) == 200
    assert result.oos_equity.iloc[0] == CASH
    assert result.strategy_name == "sma_test"
    assert result.periods_per_year == pytest.approx(8760.0)
    for key in METRIC_KEYS:
        assert key in result.oos_metrics
    # tous les trades OOS sont dans la période de test
    if len(result.oos_trades):
        assert (result.oos_trades["entry_time"] >= bars.index[100]).all()


def test_walk_forward_validation(bars: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Pas assez de bougies"):
        walk_forward(bars, Recording, train_bars=250, test_bars=51)
    with pytest.raises(ValueError, match="strictement positifs"):
        walk_forward(bars, Recording, train_bars=0, test_bars=50)
    with pytest.raises(ValueError, match="strictement positifs"):
        walk_forward(bars, Recording, train_bars=100, test_bars=-1)
    with pytest.raises(ValueError, match="obligatoires"):
        walk_forward(bars, Recording)
    with pytest.raises(ValueError, match="inconnus"):
        walk_forward(bars, Recording, grid={"zzz": [1]}, train_bars=100, test_bars=50)
    # exactement train + test bougies : une seule fenêtre, autorisé
    result = walk_forward(bars, Recording, train_bars=250, test_bars=50)
    assert result.n_windows == 1
    assert (result.windows[0]["test_start"], result.windows[0]["test_end"]) == (250, 300)


def test_walk_forward_all_combinations_failing_raises(bars: pd.DataFrame) -> None:
    with pytest.raises(RuntimeError, match="Fenêtre 1"):
        walk_forward(bars, Failing, grid={"k": [2]}, train_bars=100, test_bars=50)


def test_walk_forward_cap_logged_once(bars: pd.DataFrame, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger=LOGGER)
    walk_forward(bars, SmaCrossTest, train_bars=100, test_bars=50, max_combinations=2)
    caps = [r for r in caplog.records if r.name == LOGGER and "plafonnée" in r.getMessage()]
    assert len(caps) == 1


def test_walk_forward_warmup_longer_than_train_warns(
    bars: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger=LOGGER)
    walk_forward(bars, Recording, grid={"n": [120]}, train_bars=100, test_bars=50)
    msgs = [r.getMessage() for r in caplog.records if r.name == LOGGER]
    assert sum("chauffe" in m for m in msgs) == 1


def test_overfit_ratio() -> None:
    assert _overfit_ratio(1.0, 0.5) == pytest.approx(0.5)
    assert _overfit_ratio(2.0, 2.0) == pytest.approx(1.0)
    assert _overfit_ratio(1.0, -0.5) == pytest.approx(-0.5)  # OOS négatif : ratio négatif
    assert math.isnan(_overfit_ratio(0.0, 0.5))
    assert math.isnan(_overfit_ratio(-1.0, 0.5))  # signes opposés : pas de sens
    assert math.isnan(_overfit_ratio(-1.0, 0.0))
    assert math.isnan(_overfit_ratio(math.nan, 0.5))
    assert math.isnan(_overfit_ratio(1.0, math.nan))
    assert math.isnan(_overfit_ratio(math.inf, 1.0))
    assert math.isnan(_overfit_ratio(2.0, math.inf))  # F56 : inf n'est plus propagé
    # F56 : métrique négative « plus grand = mieux » (max_drawdown) : |IS| / |OOS|
    assert _overfit_ratio(-0.0749, -0.0751) == pytest.approx(0.0749 / 0.0751)
    assert _overfit_ratio(-0.05, -0.10) == pytest.approx(0.5)  # OOS deux fois pire
    assert _overfit_ratio(-0.10, -0.05) == pytest.approx(2.0)  # OOS deux fois meilleur


def test_finite_mean_skips_inf_and_weights() -> None:
    assert _finite_mean([1.0, 3.0]) == pytest.approx(2.0)
    assert _finite_mean([math.inf, 2.0]) == pytest.approx(2.0)  # F56
    assert _finite_mean([math.nan, 2.0, -math.inf]) == pytest.approx(2.0)
    assert math.isnan(_finite_mean([math.nan, math.inf]))
    assert math.isnan(_finite_mean([]))
    # F57 : pondération ; poids nuls ignorés
    assert _finite_mean([1.0, 3.0], [3.0, 1.0]) == pytest.approx(1.5)
    assert _finite_mean([1.0, 3.0, 100.0], [1.0, 1.0, 0.0]) == pytest.approx(2.0)
    assert _finite_mean([math.inf, 3.0], [10.0, 1.0]) == pytest.approx(3.0)


def test_walk_forward_max_drawdown_metric_has_a_ratio() -> None:
    """F56 : max_drawdown est documentée comme métrique valide : le ratio n'est plus n/d."""
    df = staircase(16)
    cfg, risk = frictionless()
    result = walk_forward(
        df, AlwaysLong, train_bars=8, test_bars=4, config=cfg, risk=risk, metric="max_drawdown"
    )
    assert result.metric == "max_drawdown"
    # escalier montant : aucun drawdown, moyennes nulles → ratio sans objet
    assert result.is_metrics_mean == 0.0 and math.isnan(result.overfit_ratio)
    bars = make_ohlcv(n=300, seed=21, vol=0.02)
    result = walk_forward(bars, AlwaysLong, train_bars=100, test_bars=50, metric="max_drawdown")
    assert result.is_metrics_mean < 0 and result.oos_metrics_mean < 0
    assert result.overfit_ratio == pytest.approx(result.is_metrics_mean / result.oos_metrics_mean)
    assert "n/d" not in result.summary().split("Ratio OOS / IS")[1].splitlines()[0]


def test_summary_is_french_and_explains_ratio() -> None:
    df = staircase(12)
    cfg, risk = frictionless()
    result = walk_forward(
        df, AlwaysLong, train_bars=4, test_bars=4, config=cfg, risk=risk, metric="total_return"
    )
    text = result.summary()
    assert "Walk-forward" in text
    assert "always_long" in text
    assert "Fenêtres : 2" in text
    assert "Moyenne IS" in text and "Moyenne OOS" in text
    assert "Ratio OOS / IS" in text
    assert "Lecture" in text and "sur-apprentissage" in text and "robuste" in text
    assert "dummy=1" in text
    assert "Équité OOS" in text
    assert "Buy & hold sur la même période OOS, frais inclus" in text
    assert "même base" in text
    assert "Moyennes pondérées par le nombre de bougies de test" in text
    assert "la position est fermée à la fin de chaque fenêtre de test" in text
    assert "max_drawdown, le ratio vaut |IS| / |OOS|" in text
    assert "⚠" not in text  # aucun kill switch : pas d'avertissement
    assert "ne l'aurait pas déclenché" not in text  # kill switch désactivé dans frictionless()
    assert "inactive" not in text
    frame = result.windows_frame()
    assert len(frame) == 2
    assert frame.loc[0, "best_params"] == "dummy=1"


def test_summary_without_positive_is_mean(bars: pd.DataFrame) -> None:
    result = walk_forward(bars, Recording, train_bars=100, test_bars=50)
    assert math.isnan(result.overfit_ratio)
    assert "n/d" in result.summary()


def test_walk_forward_accepts_unsorted_input(bars: pd.DataFrame) -> None:
    shuffled = bars.sample(frac=1.0, random_state=0)
    result = walk_forward(shuffled, Recording, train_bars=100, test_bars=50)
    assert result.oos_equity.index.equals(bars.index[100:])


# ============================================================================ F51 : même base IS / OOS
def test_in_sample_metric_uses_warmup_history_like_oos() -> None:
    """F51 : avec train_bars == test_bars, la fenêtre d'entraînement k+1 est exactement la
    fenêtre de test k : leurs métriques doivent être identiques (même historique de chauffe,
    mêmes bougies évaluées)."""
    df = staircase(60)
    cfg, risk = frictionless()
    result = walk_forward(
        df, AlwaysLongW, train_bars=20, test_bars=20, config=cfg, risk=risk, metric="total_return"
    )
    assert [(w["test_start"], w["test_end"]) for w in result.windows] == [(20, 40), (40, 60)]
    c = df["close"].to_numpy()
    # IS fenêtre 1 : train [0, 20) sans historique → métrique à partir de la fin de la chauffe
    # (bougie 10) : entrée à open[11] = close[10] = 110, sortie close[19] = 119
    assert result.windows[0]["is_metric"] == pytest.approx(c[19] / c[10] - 1)
    # OOS fenêtre 1 : test [20, 40) avec 10 bougies d'historique : entrée à open[21] = 120
    assert result.windows[0]["oos_metric"] == pytest.approx(c[39] / c[20] - 1)
    # IS fenêtre 2 : train [20, 40) avec historique → identique à l'OOS de la fenêtre 1
    assert result.windows[1]["is_metric"] == pytest.approx(result.windows[0]["oos_metric"])
    assert result.windows[1]["oos_metric"] == pytest.approx(c[59] / c[40] - 1)
    assert result.is_metrics_mean == pytest.approx(((c[19] / c[10] - 1) + (c[39] / c[20] - 1)) / 2)
    # l'équité OOS couvre toujours toutes les bougies de test, plate sur la 1re de chaque fenêtre
    assert result.oos_equity.index.equals(df.index[20:])
    assert result.oos_positions.tolist()[:3] == [0, 1, 1]
    assert result.oos_positions.tolist()[20:23] == [0, 1, 1]


def test_in_sample_sharpe_not_diluted_by_warmup(bars: pd.DataFrame) -> None:
    """F51 : même invariant avec une vraie stratégie et le Sharpe (sensible aux bougies
    plates) ; avant la correction, l'IS était calculé sur la fenêtre entière, chauffe comprise."""
    result = walk_forward(
        bars, SmaCrossTest, grid={"fast": [5], "slow": [20]}, train_bars=75, test_bars=75
    )
    assert result.n_windows == 3
    for k in range(result.n_windows - 1):
        assert result.windows[k + 1]["is_metric"] == pytest.approx(
            result.windows[k]["oos_metric"], abs=1e-12
        )
        assert result.windows[k]["oos_metric"] != 0.0
    # repro de la revue : stratégie identique IS et OOS → ratio raisonnable, pas 1,9
    result = walk_forward(
        staircase(60),
        AlwaysLongW,
        train_bars=20,
        test_bars=20,
        config=frictionless()[0],
        risk=frictionless()[1],
        metric="total_return",
    )
    assert 0.9 < result.overfit_ratio < 1.3


# ============================================================================ F57 : moyennes pondérées
def test_means_weighted_by_test_bars_hand_computed() -> None:
    """13 bougies, train 4, test 5 : fenêtres de test [4, 9) (5 bougies) et [9, 13) (4)."""
    df = staircase(13)
    cfg, risk = frictionless()
    result = walk_forward(
        df, AlwaysLong, train_bars=4, test_bars=5, config=cfg, risk=risk, metric="total_return"
    )
    assert [w["n_test_bars"] for w in result.windows] == [5, 4]
    c = df["close"].to_numpy()
    oos1, oos2 = c[8] / c[4] - 1, c[12] / c[9] - 1
    is1, is2 = c[3] / c[0] - 1, c[8] / c[5] - 1  # train [0, 4) puis [5, 9)
    assert result.windows[0]["oos_metric"] == pytest.approx(oos1)
    assert result.windows[1]["oos_metric"] == pytest.approx(oos2)
    assert result.windows[0]["is_metric"] == pytest.approx(is1)
    assert result.windows[1]["is_metric"] == pytest.approx(is2)
    assert result.oos_metrics_mean == pytest.approx((5 * oos1 + 4 * oos2) / 9)
    assert result.is_metrics_mean == pytest.approx((5 * is1 + 4 * is2) / 9)
    assert result.overfit_ratio == pytest.approx(result.oos_metrics_mean / result.is_metrics_mean)


# ============================================================================ F52 : bornes de fenêtres
def test_boundary_reopens_counted_with_fees() -> None:
    """Toujours long, frais 0,1 % : à la borne entre les deux fenêtres, le trade est fermé
    « end » à close[7] puis rouvert à open[9] : un aller-retour forcé, deux jambes de frais."""
    df = staircase(12)
    _cfg, risk = frictionless()
    cfg = BacktestConfig(initial_cash=CASH, fee_rate=0.001, slippage=0.0)
    result = walk_forward(df, AlwaysLong, train_bars=4, test_bars=4, config=cfg, risk=risk)
    assert result.oos_trades["exit_reason"].tolist() == ["end", "end"]
    assert result.n_boundary_reopens == 1
    # fenêtre 1 : achat de 10 000 / (104 × 1.001) unités à 104, revente à 107 (jambe de sortie)
    qty1 = CASH / (104.0 * 1.001)
    exit_fee = qty1 * 107.0 * 0.001
    # fenêtre 2 : achat de 10 000 / (108 × 1.001) unités à 108 (jambe de ré-entrée)
    qty2 = CASH / (108.0 * 1.001)
    entry_fee = qty2 * 108.0 * 0.001
    assert result.boundary_fees == pytest.approx(exit_fee + entry_fee)
    text = result.summary()
    assert "1 aller-retour(s) forcé(s)" in text
    assert f"≈ {exit_fee + entry_fee:.2f} de frais" in text
    assert "1 trade(s) de plus" in text
    # stratégie plate : rien à compter
    flat = walk_forward(df, Recording, train_bars=4, test_bars=4, config=cfg, risk=risk)
    assert flat.n_boundary_reopens == 0 and flat.boundary_fees == 0.0
    assert "aller-retour" not in flat.summary()
    assert "la position est fermée à la fin de chaque fenêtre" in flat.summary()
    # une seule fenêtre : la note n'a pas lieu d'être
    single = walk_forward(df, AlwaysLong, train_bars=8, test_bars=4, config=cfg, risk=risk)
    assert single.n_windows == 1 and "la position est fermée" not in single.summary()


def test_oos_buy_hold_is_net_of_costs() -> None:
    """D8 : le buy & hold OOS part de la première ouverture tradable (open[train_bars + 1])
    et paie un aller-retour de frais + glissement, comme la stratégie."""
    df = staircase(12)
    _cfg, risk = frictionless()
    cfg = BacktestConfig(initial_cash=CASH, fee_rate=0.001, slippage=0.0005)
    result = walk_forward(df, AlwaysLong, train_bars=4, test_bars=4, config=cfg, risk=risk)
    expected = 111.0 * 0.9995 * 0.999 / (104.0 * 1.0005 * 1.001) - 1.0
    assert result.oos_metrics["buy_hold_return"] == pytest.approx(expected)
    # par fenêtre aussi : fenêtre 2 = open[9] = 108 → close[11] = 111
    assert result.windows[1]["oos_return"] < 111.0 / 108.0 - 1.0  # la stratégie paie ses frais


# ============================================================================ F55 : kill switch après coup
def test_chained_oos_equity_checked_against_max_drawdown() -> None:
    """Escalier descendant 100, 99, 98… ; kill switch à 5 %. Chaque fenêtre perd ~3 % (jamais
    5 % : le moteur ne le déclenche pas), mais la courbe OOS enchaînée perd 6,3 %."""
    df = staircase(12, step=-1.0)
    cfg = BacktestConfig(initial_cash=CASH, fee_rate=0.0, slippage=0.0)
    risk = RiskManager(
        RiskConfig(
            risk_per_trade=1.0,
            max_position_fraction=1.0,
            max_drawdown=0.05,
            daily_loss_limit=None,
            min_trade_notional=0.0,
            default_stop_pct=0.99,
        )
    )
    result = walk_forward(df, AlwaysLong, train_bars=4, test_bars=4, config=cfg, risk=risk)
    c = df["close"].to_numpy()
    w1 = c[7] / c[4]  # 93 / 96 : -3,1 % dans la fenêtre 1
    assert result.windows[0]["oos_return"] == pytest.approx(w1 - 1)
    assert result.n_window_kill_switches == 0  # le moteur repart de zéro à chaque fenêtre
    assert all(w["oos_kill_switch"] is False for w in result.windows)
    assert result.oos_metrics["max_drawdown"] == pytest.approx(w1 * c[11] / c[8] - 1)  # -6,28 %
    assert result.max_drawdown_limit == 0.05
    # première bougie sous -5 % : équité w1 × close[t] / 92 < 0,95 ⇔ close[t] < 90,22 → t = 10
    assert result.oos_kill_switch_time == df.index[10]
    assert result.oos_kill_switch_drawdown == pytest.approx(1 - w1 * c[10] / c[8])
    text = result.summary()
    assert "⚠ En réel, le kill switch (drawdown max 5.00 %) aurait arrêté le robot le" in text
    assert "2024-01-01 10:00" in text
    assert f"baisse de {(1 - w1 * c[10] / c[8]) * 100:.2f} %" in text
    assert "repartent de zéro à chaque fenêtre" in text
    # escalier montant : pas de dépassement, message rassurant
    up = walk_forward(staircase(12), AlwaysLong, train_bars=4, test_bars=4, config=cfg, risk=risk)
    assert up.oos_kill_switch_time is None and up.oos_kill_switch_drawdown == 0.0
    assert "ne l'aurait pas déclenché" in up.summary()
    assert "⚠" not in up.summary()


def test_chained_kill_switch_logged_and_window_kill_switch_counted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger=LOGGER)
    df = staircase(12, step=-1.0)
    cfg = BacktestConfig(initial_cash=CASH, fee_rate=0.0, slippage=0.0)
    risk = RiskManager(
        RiskConfig(
            risk_per_trade=1.0,
            max_position_fraction=1.0,
            max_drawdown=0.02,
            daily_loss_limit=None,
            min_trade_notional=0.0,
            default_stop_pct=0.99,
        )
    )
    result = walk_forward(df, AlwaysLong, train_bars=4, test_bars=4, config=cfg, risk=risk)
    # -2 % atteints dans chaque fenêtre : le kill switch du moteur ferme la position
    assert result.n_window_kill_switches == 2
    assert all(w["oos_kill_switch"] for w in result.windows)
    assert result.oos_trades["exit_reason"].tolist() == ["kill_switch", "kill_switch"]
    assert "Kill switch déclenché à l'intérieur de 2 fenêtre(s)" in result.summary()
    assert result.oos_kill_switch_time is not None
    msgs = [r.getMessage() for r in caplog.records if r.name == LOGGER]
    assert any("kill switch aurait arrêté le robot" in m for m in msgs)


def test_default_risk_config_used_when_risk_is_none() -> None:
    df = staircase(12)
    result = walk_forward(df, Recording, train_bars=4, test_bars=4)
    assert result.max_drawdown_limit == pytest.approx(RiskConfig().max_drawdown)
    assert result.daily_limit_active is True


# ============================================================================ D5 : limite journalière
def test_daily_limit_inactive_on_daily_bars_is_reported() -> None:
    df = make_ohlcv(n=300, seed=5, freq="1D")
    result = walk_forward(df, Recording, train_bars=100, test_bars=50)
    assert result.daily_limit_active is False
    assert "Limite de perte journalière inactive : bougies >= 1 jour" in result.summary()
    hourly = walk_forward(make_ohlcv(n=300, seed=5), Recording, train_bars=100, test_bars=50)
    assert hourly.daily_limit_active is True
    assert "inactive" not in hourly.summary()


# ============================================================================ stratégie réelle (optionnelle)
def test_walk_forward_with_registered_strategy(bars: pd.DataFrame) -> None:
    mod = pytest.importorskip("tradingbot.strategies.sma_cross")
    cls = None
    for obj in vars(mod).values():
        if isinstance(obj, type) and getattr(obj, "name", None) == "sma_cross":
            cls = obj
    if cls is None:
        pytest.skip("stratégie sma_cross introuvable")
    result = walk_forward(bars, cls, train_bars=150, test_bars=50, max_combinations=4)
    assert result.n_windows == 3
    assert len(result.oos_equity) == 150
    assert result.oos_equity.iloc[0] == pytest.approx(BacktestConfig().initial_cash)
