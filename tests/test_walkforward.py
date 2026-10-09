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
from tradingbot.backtest.metrics import METRIC_KEYS
from tradingbot.backtest.walkforward import (
    WINDOW_KEYS,
    WalkForwardResult,
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
    """Pas de frais, pas de glissement, 100 % du capital investi, aucun garde-fou."""
    cfg = BacktestConfig(initial_cash=CASH, fee_rate=0.0, slippage=0.0)
    risk = RiskManager(
        RiskConfig(
            max_position_fraction=1.0,
            max_drawdown=None,
            daily_loss_limit=None,
            min_trade_notional=0.0,
        )
    )
    return cfg, risk


def staircase(n: int = 12, start: float = 100.0) -> pd.DataFrame:
    """Clôtures 100, 101, 102… ; ``open[t] = close[t-1]`` (pas de gap)."""
    close = start + np.arange(n, dtype=float)
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
    # chaque ligne reproduit exactement le backtest de sa combinaison
    for _, row in table.iterrows():
        res = run_backtest(bars, SmaCrossTest(fast=int(row["fast"]), slow=int(row["slow"])))
        assert row["total_return"] == res.metrics["total_return"]
        assert row["sharpe"] == res.metrics["sharpe"]
        assert row["n_trades"] == res.metrics["n_trades"]


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
    assert math.isnan(_overfit_ratio(0.0, 0.5))
    assert math.isnan(_overfit_ratio(-1.0, 0.5))
    assert math.isnan(_overfit_ratio(math.nan, 0.5))
    assert math.isnan(_overfit_ratio(1.0, math.nan))
    assert math.isnan(_overfit_ratio(math.inf, 1.0))


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
