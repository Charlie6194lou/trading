"""Tests des rapports : libellés français, valeurs formatées à la main, fichiers écrits,
graphique optionnel (matplotlib absent → None)."""

from __future__ import annotations

import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from conftest import make_ohlcv

from tradingbot.backtest import metrics as metrics_mod
from tradingbot.backtest.engine import TRADE_COLUMNS, BacktestConfig, BacktestResult, run_backtest
from tradingbot.backtest.report import (
    REPORT_FILES,
    _fmt_timeframe,
    _json_safe,
    format_comparison,
    format_report,
    plot_result,
    save_report,
)

LOGGER = "tradingbot.backtest.report"
CASH = 10_000.0
PPY = 8760.0


# ============================================================================ fabriques
def index(n: int, freq: str = "1h") -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")


def flat(text: str) -> str:
    """Texte sur une ligne (les paragraphes de lecture sont repliés par textwrap)."""
    return " ".join(text.split())


def empty_trades() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=object) for c in TRADE_COLUMNS})


def make_trades(pnls: list[float], start: pd.Timestamp | None = None) -> pd.DataFrame:
    """Un trade par valeur de ``pnl`` (notionnel 1 000 → ``pnl_pct = pnl / 1000``)."""
    n = len(pnls)
    base = start if start is not None else pd.Timestamp("2024-01-01", tz="UTC")
    entries = [base + pd.Timedelta(hours=4 * i) for i in range(n)]
    exits = [e + pd.Timedelta(hours=3) for e in entries]
    return pd.DataFrame(
        {
            "entry_time": pd.DatetimeIndex(entries),
            "exit_time": pd.DatetimeIndex(exits),
            "side": np.ones(n, dtype=np.int64),
            "entry_price": np.full(n, 100.0),
            "exit_price": 100.0 + np.asarray(pnls, dtype=float) / 10.0,
            "size": np.full(n, 10.0),
            "pnl": np.asarray(pnls, dtype=float),
            "pnl_pct": np.asarray(pnls, dtype=float) / 1000.0,
            "fees": np.full(n, 1.0),
            "bars_held": np.full(n, 3, dtype=np.int64),
            "exit_reason": ["signal" if p >= 0 else "stop_loss" for p in pnls],
        }
    )


def make_result(
    equity: list[float],
    trades: pd.DataFrame | None = None,
    name: str = "strat_test",
    params: dict[str, Any] | None = None,
    benchmark: pd.Series | None = None,
    metrics: dict[str, Any] | None = None,
    config: BacktestConfig | None = None,
    ppy: float = PPY,
) -> BacktestResult:
    idx = index(len(equity))
    eq = pd.Series(equity, index=idx, dtype=float, name="equity")
    trades = trades if trades is not None else empty_trades()
    positions = pd.Series(np.zeros(len(eq), dtype=np.int64), index=idx, name="position")
    cfg = config if config is not None else BacktestConfig(initial_cash=equity[0])
    if metrics is None:
        metrics = metrics_mod.compute_metrics(
            eq, trades, ppy, benchmark=benchmark, positions=positions
        )
    return BacktestResult(
        strategy_name=name,
        params=params if params is not None else {"n": 3},
        config=cfg,
        equity=eq,
        returns=eq.pct_change().fillna(0.0).rename("returns"),
        positions=positions,
        trades=trades,
        metrics=metrics,
        signals=pd.DataFrame(index=idx),
        periods_per_year=ppy,
    )


class ScriptedStrategy:
    """Stratégie « duck-typée » : long sur des plages de bougies données."""

    name = "scripted"
    description = "plages longues explicites"
    default_params: dict[str, Any] = {"ranges": ()}

    def __init__(self, **params: Any) -> None:
        self.params = {**self.default_params, **params}

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        sig = np.zeros(len(df), dtype=np.int64)
        for start, end in self.params["ranges"]:
            sig[start:end] = 1
        return pd.DataFrame(
            {"signal": sig, "stop_loss": np.nan, "take_profit": np.nan}, index=df.index
        )

    def warmup_bars(self) -> int:
        return 0


@pytest.fixture
def engine_result() -> BacktestResult:
    df = make_ohlcv(n=120, seed=9)
    strat = ScriptedStrategy(ranges=((5, 20), (30, 45), (50, 60), (70, 100)))
    return run_backtest(df, strat, BacktestConfig(initial_cash=CASH))


@pytest.fixture
def no_matplotlib(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fait échouer ``import matplotlib`` (même s'il est installé)."""
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)


# ============================================================================ format_report
def test_format_report_contains_french_labels_and_strategy(engine_result: BacktestResult) -> None:
    text = format_report(engine_result)
    assert "Rapport de backtest : scripted" in text
    for label in (
        "Stratégie",
        "Paramètres",
        "Période",
        "Bougies",
        "Capital initial",
        "Capital final",
        "Rendement total",
        "CAGR",
        "Volatilité annualisée",
        "Sharpe",
        "Sortino",
        "Drawdown maximal",
        "Nombre de trades",
        "Taux de réussite",
        "Profit factor",
        "Frais totaux",
        "Comparaison avec le buy & hold",
        "Buy & hold",
        "Lecture (pour débutant)",
        "Rappel",
    ):
        assert label in text, label
    assert "120 (pas de 1 h)" in text
    assert "Nombre de trades : 4" in text
    assert " %" in text
    # le buy & hold vient des métriques du moteur
    bh = engine_result.metrics["buy_hold_return"]
    assert f"{bh * 100:+.2f} %" in text
    # 4 trades ≤ 5 : un seul tableau
    assert "Tous les trades (4)" in text
    assert "pires trades" not in text


def test_format_report_hand_computed_values() -> None:
    # 10 000 → 11 230 : +12.30 % ; creux 10 200 après un plus haut à 10 500 : -2.86 %
    result = make_result([10_000.0, 10_500.0, 10_200.0, 11_230.0], params={"fast": 5, "slow": 20})
    text = format_report(result, benchmark_return=0.05)
    assert "Rendement total            : +12.30 %" in text
    assert "Capital final              : 11 230.00" in text
    assert "Capital initial   : 10 000.00" in text
    assert "Drawdown maximal           : -2.86 %" in text
    assert "Durée max du drawdown      : 1 bougies" in text
    assert "Paramètres        : fast=5, slow=20" in text
    assert "Stratégie  : +12.30 %" in text
    assert "Buy & hold : +5.00 %" in text
    assert "Écart      : +7.30 points" in text
    assert "fait mieux que le simple achat-conservation" in text
    reading = flat(text.split("Lecture (pour débutant)")[1])
    assert "elle fait mieux que le marché" in reading
    assert "10 000.00 en 11 230.00 (+12.30 %)" in reading
    assert "contre +5.00 % en achetant simplement" in reading


def test_format_report_worse_than_benchmark_and_sharpe_wording() -> None:
    result = make_result([10_000.0, 9_900.0, 9_800.0, 9_700.0])
    text = format_report(result, benchmark_return=0.10)
    assert "Rendement total            : -3.00 %" in text
    assert "fait moins bien que le simple achat-conservation" in text
    assert "Écart      : -13.00 points" in text
    reading = flat(text.split("Lecture (pour débutant)")[1])
    assert "elle fait moins bien que le marché" in reading
    assert "négatif : la stratégie perd de l'argent" in reading


def test_format_report_without_benchmark() -> None:
    result = make_result([10_000.0, 10_100.0])
    assert "buy_hold_return" not in result.metrics
    text = format_report(result)
    assert "Buy & hold : non disponible" in text
    assert "contre" not in flat(text.split("Lecture (pour débutant)")[1])


def test_format_report_zero_trades() -> None:
    result = make_result([10_000.0] * 5)
    text = format_report(result)
    assert "Nombre de trades : 0" in text
    assert "Aucun trade sur la période." in text
    assert "aucun trade sur la période" in flat(text)  # paragraphe de lecture
    assert "Rendement total            : +0.00 %" in text
    assert "Frais totaux     : 0.00" in text


def test_format_report_worst_and_best_trades_tables() -> None:
    pnls = [-60.0, 5.0, -50.0, 10.0, -40.0, 20.0, -30.0, 30.0, -20.0, 40.0, -10.0, 50.0]
    result = make_result([10_000.0] * 50, trades=make_trades(pnls))
    text = format_report(result)
    assert "Nombre de trades : 12" in text
    worst_start = text.index("Les 5 pires trades")
    best_start = text.index("Les 5 meilleurs trades")
    reading_start = text.index("Lecture (pour débutant)")
    assert worst_start < best_start < reading_start
    worst = text[worst_start:best_start]
    best = text[best_start:reading_start]
    # pires : -60, -50, -40, -30, -20 dans cet ordre ; -10 et les gains absents
    positions = [worst.index(f"{p:.2f} |") for p in (-60.0, -50.0, -40.0, -30.0, -20.0)]
    assert positions == sorted(positions)
    assert "-10.00" not in worst
    assert re.search(r"(?<![-.\d])50\.00 \|", worst) is None  # +50 absent des pires
    # meilleurs : 50, 40, 30, 20, 10 dans cet ordre ; 5 absent
    positions = [best.index(f" {p:.2f} |") for p in (50.0, 40.0, 30.0, 20.0, 10.0)]
    assert positions == sorted(positions)
    assert re.search(r"(?<![-.\d])5\.00 \|", best) is None
    rows = [ln for ln in best.splitlines() if ln.startswith("2024-")]
    assert len(rows) == 5
    assert all("-" not in row.split("|")[5] for row in rows)  # colonne P&L sans perte
    rows = [ln for ln in worst.splitlines() if ln.startswith("2024-")]
    assert len(rows) == 5
    assert all("-" in row.split("|")[5] for row in rows)
    # colonnes et libellés français des raisons de sortie
    assert "Entrée" in worst and "Sortie" in worst and "Raison" in worst
    assert "stop" in worst and "Long" in worst
    assert "signal" in best
    # pnl_pct : -60 / 1000 = -6.00 %
    assert "-6.00 %" in worst


def test_format_report_handles_short_and_inf_profit_factor() -> None:
    trades = make_trades([10.0, 20.0])
    trades.loc[:, "side"] = np.array([-1, 1], dtype=np.int64)
    result = make_result([10_000.0] * 10, trades=trades)
    assert math.isinf(result.metrics["profit_factor"])
    text = format_report(result)
    assert "Profit factor    : ∞" in text
    assert "Short" in text and "Long" in text


def test_fmt_timeframe() -> None:
    assert _fmt_timeframe(index(3, "1h")) == "1 h"
    assert _fmt_timeframe(index(3, "4h")) == "4 h"
    assert _fmt_timeframe(index(3, "1D")) == "1 j"
    assert _fmt_timeframe(index(3, "15min")) == "15 min"
    assert _fmt_timeframe(index(1)) == "n/d"


# ============================================================================ format_comparison
def _metrics(sharpe: float, total: float, trades: int) -> dict[str, Any]:
    return {
        "total_return": total,
        "cagr": total * 2,
        "sharpe": sharpe,
        "max_drawdown": -0.1,
        "n_trades": trades,
        "win_rate": 0.5,
        "buy_hold_return": 0.02,
    }


def test_format_comparison_sorted_by_sharpe_with_buy_hold_row() -> None:
    a = make_result([10_000.0, 10_100.0, 10_200.0], name="alpha", metrics=_metrics(0.5, 0.02, 3))
    b = make_result([10_000.0, 10_100.0, 10_200.0], name="beta", metrics=_metrics(2.0, 0.05, 7))
    c = make_result([10_000.0, 10_100.0, 10_200.0], name="gamma", metrics=_metrics(1.0, -0.01, 1))
    text = format_comparison([a, b, c])
    lines = text.splitlines()
    assert lines[0].startswith("Comparaison des stratégies")
    for col in ("Stratégie", "Rendement", "CAGR", "Sharpe", "Max DD", "Trades", "Réussite"):
        assert col in lines[2]
    assert text.index("beta") < text.index("gamma") < text.index("alpha")
    assert lines[-1].startswith("Buy & hold")
    assert "+2.00 %" in lines[-1]
    assert "+5.00 %" in text and "-1.00 %" in text
    assert "| 7 |" in text.replace("  ", " ") or " 7 |" in text
    assert "50.00 %" in text


def test_format_comparison_with_benchmark_series() -> None:
    a = make_result(
        [10_000.0, 10_100.0, 10_200.0], name="alpha", metrics=_metrics(0.5, 0.02, 3), ppy=365.0
    )
    bench = pd.Series([100.0, 110.0, 121.0], index=index(3, "1D"))
    text = format_comparison([a], benchmark=bench)
    last = text.splitlines()[-1]
    assert last.startswith("Buy & hold")
    assert "+21.00 %" in last
    # rendements 0, 10 %, 10 % : moyenne 1/15, écart-type (ddof=1) sqrt(1/300)
    expected = (1 / 15) / math.sqrt(1 / 300) * math.sqrt(365.0)
    assert expected == pytest.approx(22.06, abs=0.005)
    assert f"{expected:.2f}" in last
    assert "+0.00 %" in last  # pas de drawdown


def test_format_comparison_benchmark_overflow_falls_back() -> None:
    # +21 % en 2 bougies horaires : CAGR hors de portée d'un float → ligne de repli
    a = make_result([10_000.0, 10_100.0, 10_200.0], name="alpha", metrics=_metrics(0.5, 0.02, 3))
    bench = pd.Series([100.0, 110.0, 121.0], index=index(3))
    text = format_comparison([a], benchmark=bench)
    last = text.splitlines()[-1]
    assert last.startswith("Buy & hold")
    assert "+2.00 %" in last  # buy_hold_return des métriques
    assert "–" in last  # Sharpe et drawdown indisponibles dans la ligne de repli


def test_format_comparison_duplicate_names_show_params() -> None:
    a = make_result(
        [10_000.0, 10_100.0], name="sma", params={"n": 3}, metrics=_metrics(1.0, 0.0, 1)
    )
    b = make_result(
        [10_000.0, 10_100.0], name="sma", params={"n": 5}, metrics=_metrics(2.0, 0.0, 1)
    )
    text = format_comparison([a, b])
    assert "sma (n=3)" in text and "sma (n=5)" in text
    assert text.index("n=5") < text.index("n=3")


def test_format_comparison_empty() -> None:
    assert format_comparison([]) == "Aucun résultat à comparer."


# ============================================================================ save_report
def test_save_report_writes_files_with_right_content(tmp_path: Path, no_matplotlib: None) -> None:
    pnls = [-60.0, 5.0, -50.0, 10.0, -40.0, 20.0, -30.0, 30.0, -20.0, 40.0, -10.0, 50.0]
    result = make_result(
        [10_000.0, 10_500.0, 10_200.0, 11_230.0], trades=make_trades(pnls), params={"n": 3}
    )
    out = tmp_path / "rapport"
    paths = save_report(result, out)
    assert [p.name for p in paths] == list(REPORT_FILES)
    assert all(p.exists() for p in paths)

    equity = pd.read_csv(out / "equity.csv", parse_dates=["timestamp"])
    assert list(equity.columns) == ["timestamp", "equity", "returns", "position", "drawdown"]
    assert len(equity) == 4
    assert equity["equity"].tolist() == [10_000.0, 10_500.0, 10_200.0, 11_230.0]
    assert equity.loc[2, "drawdown"] == pytest.approx(10_200.0 / 10_500.0 - 1.0)
    assert equity.loc[1, "returns"] == pytest.approx(0.05)
    assert pd.Timestamp(equity.loc[0, "timestamp"]) == result.equity.index[0]

    trades = pd.read_csv(out / "trades.csv")
    assert len(trades) == 12
    assert trades["pnl"].tolist() == pnls
    assert set(TRADE_COLUMNS) <= set(trades.columns)

    summary = (out / "summary.txt").read_text(encoding="utf-8")
    assert summary.startswith("Rapport de backtest : strat_test")
    assert "+12.30 %" in summary

    with (out / "metrics.json").open(encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["strategy"] == "strat_test"
    assert data["params"] == {"n": 3}
    assert data["config"]["initial_cash"] == 10_000.0
    assert data["n_bars"] == 4
    assert data["start"] == "2024-01-01T00:00:00+00:00"
    assert data["metrics"]["total_return"] == pytest.approx(0.123)
    assert data["metrics"]["n_trades"] == 12
    assert set(metrics_mod.METRIC_KEYS) - {"buy_hold_return"} <= set(data["metrics"])


def test_save_report_zero_trades_and_json_without_nan(tmp_path: Path, no_matplotlib: None) -> None:
    result = make_result([10_000.0] * 5)
    paths = save_report(result, tmp_path)
    assert len(paths) == 4
    trades = pd.read_csv(tmp_path / "trades.csv")
    assert len(trades) == 0
    assert list(trades.columns) == list(TRADE_COLUMNS)
    raw = (tmp_path / "metrics.json").read_text(encoding="utf-8")
    assert "NaN" not in raw and "Infinity" not in raw
    data = json.loads(raw)
    assert data["metrics"]["n_trades"] == 0


def test_save_report_infinite_profit_factor_becomes_null(
    tmp_path: Path, no_matplotlib: None
) -> None:
    result = make_result([10_000.0] * 10, trades=make_trades([10.0, 20.0]))
    save_report(result, tmp_path)
    raw = (tmp_path / "metrics.json").read_text(encoding="utf-8")
    assert "Infinity" not in raw
    data = json.loads(raw)
    assert data["metrics"]["profit_factor"] is None
    assert data["metrics"]["win_rate"] == 1.0


def test_save_report_from_engine_result(tmp_path: Path, engine_result: BacktestResult) -> None:
    paths = save_report(engine_result, tmp_path / "a" / "b")
    names = [p.name for p in paths]
    assert names[:4] == list(REPORT_FILES)
    assert set(names) <= {*REPORT_FILES, "equity.png"}
    data = json.loads((tmp_path / "a" / "b" / "metrics.json").read_text(encoding="utf-8"))
    assert data["metrics"] == pytest.approx(
        {
            k: (None if isinstance(v, float) and not math.isfinite(v) else v)
            for k, v in engine_result.metrics.items()
        }
    )
    trades = pd.read_csv(tmp_path / "a" / "b" / "trades.csv")
    assert len(trades) == len(engine_result.trades) == 4


def test_json_safe() -> None:
    ts = pd.Timestamp("2024-01-01", tz="UTC")
    data = {
        "a": np.int64(3),
        "b": np.float64("nan"),
        "c": ts,
        "d": (1, 2),
        "e": float("inf"),
        "f": np.array([1.5, 2.5]),
        "g": BacktestConfig(),
        "h": True,
        3: "clé entière",
    }
    out = _json_safe(data)
    assert out["a"] == 3 and isinstance(out["a"], int)
    assert out["b"] is None and out["e"] is None
    assert out["c"] == "2024-01-01T00:00:00+00:00"
    assert out["d"] == [1, 2] and out["f"] == [1.5, 2.5]
    assert out["g"]["fee_rate"] == 0.001 and out["g"]["stop_loss_pct"] is None
    assert out["h"] is True
    assert out["3"] == "clé entière"
    json.dumps(out, allow_nan=False)


# ============================================================================ plot_result
def test_plot_result_returns_none_without_matplotlib(
    tmp_path: Path, no_matplotlib: None, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger=LOGGER)
    result = make_result([10_000.0, 10_100.0, 10_050.0])
    target = tmp_path / "equity.png"
    assert plot_result(result, target) is None
    assert not target.exists()
    assert any("matplotlib" in r.getMessage() for r in caplog.records if r.name == LOGGER)


def test_save_report_without_matplotlib_has_no_png(tmp_path: Path, no_matplotlib: None) -> None:
    result = make_result([10_000.0, 10_100.0, 10_050.0])
    paths = save_report(result, tmp_path)
    assert [p.name for p in paths] == list(REPORT_FILES)
    assert not (tmp_path / "equity.png").exists()


def test_plot_result_with_matplotlib(tmp_path: Path, engine_result: BacktestResult) -> None:
    pytest.importorskip("matplotlib")
    target = tmp_path / "sous" / "dossier" / "equity.png"
    out = plot_result(engine_result, target)
    assert out == target
    assert target.exists() and target.stat().st_size > 0
    # sans trade non plus, le graphique doit s'écrire
    assert plot_result(make_result([10_000.0] * 5), tmp_path / "flat.png") is not None
