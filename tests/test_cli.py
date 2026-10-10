"""Tests de la ligne de commande (``tradingbot.cli.main``). Aucun accès réseau : l'exchange
ccxt est remplacé par un faux objet servant des bougies et un ticker, et le broker réel par
un faux ``CcxtBroker`` (jamais construit pour de vrai)."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from conftest import make_ohlcv

from tradingbot import cli
from tradingbot.cli import DEMO_WARNING, main
from tradingbot.data.loader import load_csv, save_csv
from tradingbot.execution.paper import PaperBroker
from tradingbot.strategies import STRATEGIES

REPO_ROOT = Path(__file__).resolve().parents[1]


# ============================================================================ faux objets
class FakeExchange:
    """Exchange ccxt minimal : bougies d'un DataFrame + ticker, sans réseau."""

    rateLimit = 0
    timeframes = {"1m": "1m", "15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}

    def __init__(self, df: pd.DataFrame, price: float | None = None) -> None:
        self.df = df
        self.price = price
        self.ohlcv_calls: list[tuple[Any, ...]] = []
        self.ticker_calls = 0

    def parse8601(self, text: str) -> None:  # le chargeur retombe alors sur pandas
        return None

    def milliseconds(self) -> int:
        return int(time.time() * 1000)

    def fetch_ohlcv(
        self, symbol: str, timeframe: str, since: int | None = None, limit: int | None = None
    ) -> list[list[float]]:
        self.ohlcv_calls.append((symbol, timeframe, since, limit))
        rows = [
            [
                int(ts.timestamp() * 1000),
                float(r.open),
                float(r.high),
                float(r.low),
                float(r.close),
                float(r.volume),
            ]
            for ts, r in zip(self.df.index, self.df.itertuples(index=False), strict=True)
        ]
        if since is not None:
            rows = [r for r in rows if r[0] >= since]
        if limit is not None:
            rows = rows[-limit:] if since is None else rows[:limit]
        return rows

    def fetch_ticker(self, symbol: str) -> dict[str, float]:
        self.ticker_calls += 1
        last = self.price if self.price is not None else float(self.df["close"].iloc[-1])
        return {"symbol": symbol, "last": last}


class FakeCcxtBroker(PaperBroker):
    """Remplaçant de ``CcxtBroker`` : enregistre les arguments, trade sur un compte fictif."""

    created: list[dict[str, Any]] = []
    exchange_for_prices: FakeExchange | None = None

    def __init__(
        self,
        exchange_id: str,
        api_key: str,
        api_secret: str,
        sandbox: bool = False,
        confirm_live: bool = False,
        exchange: Any | None = None,
    ) -> None:
        FakeCcxtBroker.created.append(
            {
                "exchange_id": exchange_id,
                "api_key": api_key,
                "api_secret": api_secret,
                "sandbox": sandbox,
                "confirm_live": confirm_live,
                "exchange": exchange,
            }
        )
        feed = FakeCcxtBroker.exchange_for_prices
        super().__init__(
            initial_cash=1000.0,
            price_feed=(lambda sym: feed.fetch_ticker(sym)["last"]) if feed is not None else None,
        )


def rising_ohlcv(n: int = 700, start: str = "2024-01-01") -> pd.DataFrame:
    """Hausse régulière avec un léger zigzag : ``sma_cross`` est long sur la dernière bougie."""
    t = np.arange(n, dtype=float)
    close = 100.0 + 0.2 * t + 0.3 * np.sin(t / 3.0)
    open_ = np.concatenate([[100.0], close[:-1]])
    high = np.maximum(open_, close) + 0.1
    low = np.minimum(open_, close) - 0.1
    index = pd.date_range(start=start, periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 1000.0},
        index=index,
    )


# ============================================================================ fixtures
@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Répertoire courant temporaire (pas de config.toml / .env parasite) et journalisation
    du paquet remise dans son état initial après chaque test."""
    monkeypatch.chdir(tmp_path)
    pkg = logging.getLogger("tradingbot")
    level, handlers = pkg.level, list(pkg.handlers)
    FakeCcxtBroker.created.clear()
    FakeCcxtBroker.exchange_for_prices = None
    yield
    pkg.setLevel(level)
    for handler in list(pkg.handlers):
        if handler not in handlers:
            pkg.removeHandler(handler)


@pytest.fixture
def csv_path(tmp_path: Path) -> Path:
    path = tmp_path / "rand.csv"
    save_csv(make_ohlcv(n=600), path)
    return path


@pytest.fixture
def trend_csv(tmp_path: Path) -> Path:
    path = tmp_path / "trend.csv"
    save_csv(make_ohlcv(n=800, seed=7, drift=0.002, vol=0.005), path)
    return path


@pytest.fixture
def fake_exchange(monkeypatch: pytest.MonkeyPatch) -> FakeExchange:
    exchange = FakeExchange(rising_ohlcv())
    monkeypatch.setattr(cli, "build_public_exchange", lambda exchange_id: exchange)
    return exchange


def _run(capsys: pytest.CaptureFixture[str], argv: list[str]) -> tuple[int, str, str]:
    code = main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# ============================================================================ généralités
def test_no_command_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    code, _out, err = _run(capsys, [])
    assert code == 2
    assert "usage" in err.lower()


def test_unknown_command_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    code, _out, _err = _run(capsys, ["bogus"])
    assert code == 2


def test_help_exits_0(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _err = _run(capsys, ["--help"])
    assert code == 0
    assert "list-strategies" in out and "paper" in out and "live" in out


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _err = _run(capsys, ["--version"])
    assert code == 0
    assert "tradingbot 0.1.0" in out


def test_python_m_entry_point() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "tradingbot", "--version"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert proc.returncode == 0
    assert "tradingbot 0.1.0" in proc.stdout


def test_safe_name() -> None:
    assert cli._safe_name("BTC/USDT") == "BTC-USDT"
    assert cli._safe_name("BTC/USDT:USDT") == "BTC-USDT-USDT"
    assert cli._safe_name("  ") == "unknown"


def test_build_public_exchange_unknown_id() -> None:
    with pytest.raises(cli.CliError, match="Exchange inconnu"):
        cli.build_public_exchange("exchange_qui_n_existe_pas")


def test_build_public_exchange_known_id_builds_without_network() -> None:
    exchange = cli.build_public_exchange("binance")
    assert hasattr(exchange, "fetch_ohlcv")
    assert exchange.enableRateLimit is True


def test_ticker_price_fallbacks() -> None:
    class Ticker:
        def __init__(self, payload: Any) -> None:
            self.payload = payload

        def fetch_ticker(self, symbol: str) -> Any:
            return self.payload

    assert cli._ticker_price(Ticker({"last": 12.5}), "X/Y") == 12.5
    assert cli._ticker_price(Ticker({"last": None, "close": 7.0}), "X/Y") == 7.0
    from tradingbot.execution.broker import BrokerError

    with pytest.raises(BrokerError, match="Prix indisponible"):
        cli._ticker_price(Ticker({"bid": 1.0}), "X/Y")

    class Broken:
        def fetch_ticker(self, symbol: str) -> Any:
            raise RuntimeError("réseau coupé")

    with pytest.raises(BrokerError, match="réseau coupé"):
        cli._ticker_price(Broken(), "X/Y")


def test_log_level_option_sets_package_logger(capsys: pytest.CaptureFixture[str]) -> None:
    code, _out, _err = _run(capsys, ["list-strategies", "--log-level", "debug"])
    assert code == 0
    assert logging.getLogger("tradingbot").level == logging.DEBUG


# ============================================================================ list-strategies
def test_list_strategies_prints_every_strategy(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _err = _run(capsys, ["list-strategies"])
    assert code == 0
    assert len(STRATEGIES) >= 5
    for name, cls in STRATEGIES.items():
        assert name in out
        assert cls.description.split(" ")[0] in out
    assert "sma_cross" in out and "fast=20" in out


# ============================================================================ demo
def test_demo_runs_and_prints_comparison_and_warning(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _err = _run(capsys, ["demo", "--bars", "600"])
    assert code == 0
    assert "600 bougies horaires synthétiques" in out
    assert "Comparaison des stratégies" in out
    assert "Buy & hold" in out
    for name in STRATEGIES:
        assert name in out
    assert DEMO_WARNING in out
    assert "SYNTHÉTIQUES" in DEMO_WARNING and "ne prouve RIEN" in DEMO_WARNING


def test_demo_is_reproducible(capsys: pytest.CaptureFixture[str]) -> None:
    _code, out1, _ = _run(capsys, ["demo", "--bars", "600"])
    _code, out2, _ = _run(capsys, ["demo", "--bars", "600"])
    assert out1 == out2
    _code, out3, _ = _run(capsys, ["demo", "--bars", "600", "--seed", "7"])
    assert out3 != out1


def test_demo_rejects_too_few_bars(capsys: pytest.CaptureFixture[str]) -> None:
    code, _out, err = _run(capsys, ["demo", "--bars", "100"])
    assert code == 1
    assert "--bars" in err and "300" in err


def test_demo_report_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out_dir = tmp_path / "demo_rep"
    code, out, _err = _run(capsys, ["demo", "--bars", "600", "--report-dir", str(out_dir)])
    assert code == 0
    assert (out_dir / "comparison.txt").is_file()
    assert "Comparaison des stratégies" in (out_dir / "comparison.txt").read_text(encoding="utf-8")
    for name in STRATEGIES:
        assert (out_dir / name / "summary.txt").is_file()
        assert (out_dir / name / "metrics.json").is_file()
    assert "Fichiers écrits" in out


# ============================================================================ backtest
def test_backtest_csv_prints_report(csv_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _err = _run(capsys, ["backtest", "--strategy", "sma_cross", "--data", str(csv_path)])
    assert code == 0
    assert "600 bougies lues" in out
    assert "Rapport de backtest : sma_cross" in out
    assert "Sharpe" in out and "Drawdown maximal" in out
    assert "Lecture (pour débutant)" in out
    assert "Capital initial      : 1 000.00" in out  # défaut de config.example.toml


def test_backtest_report_dir_writes_files(
    csv_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = tmp_path / "rapport"
    code, out, _err = _run(
        capsys,
        [
            "backtest",
            "--strategy",
            "sma_cross",
            "--data",
            str(csv_path),
            "--report-dir",
            str(out_dir),
        ],
    )
    assert code == 0
    for name in ("equity.csv", "trades.csv", "summary.txt", "metrics.json"):
        assert (out_dir / name).is_file(), name
        assert str(out_dir / name) in out
    payload = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert payload["strategy"] == "sma_cross"
    assert payload["n_bars"] == 600
    equity = pd.read_csv(out_dir / "equity.csv")
    assert len(equity) == 600
    assert list(equity.columns) == ["timestamp", "equity", "returns", "position", "drawdown"]


def test_backtest_param_overrides(trend_csv: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _err = _run(
        capsys,
        [
            "backtest",
            "--strategy",
            "sma_cross",
            "--data",
            str(trend_csv),
            "--param",
            "fast=5",
            "--param",
            "slow=15",
            "--param",
            "atr_mult=2.5",
        ],
    )
    assert code == 0
    assert "fast=5, slow=15, atr_window=14, atr_mult=2.5" in out


def test_backtest_param_without_equal_sign(
    csv_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _out, err = _run(
        capsys, ["backtest", "--strategy", "sma_cross", "--data", str(csv_path), "--param", "fast"]
    )
    assert code == 1
    assert "format attendu nom=valeur" in err


def test_backtest_unknown_param(csv_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, _out, err = _run(
        capsys,
        ["backtest", "--strategy", "sma_cross", "--data", str(csv_path), "--param", "inconnu=3"],
    )
    assert code == 1
    assert "Paramètres inconnus" in err and "inconnu" in err


def test_backtest_invalid_param_value(csv_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, _out, err = _run(
        capsys,
        ["backtest", "--strategy", "sma_cross", "--data", str(csv_path), "--param", "fast=100"],
    )
    assert code == 1
    assert "fast" in err and "slow" in err


def test_backtest_unknown_strategy_exits_1(
    csv_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _out, err = _run(capsys, ["backtest", "--strategy", "nope", "--data", str(csv_path)])
    assert code == 1
    assert "Stratégie inconnue" in err
    assert "sma_cross" in err  # la liste des stratégies disponibles est proposée


def test_backtest_missing_csv_exits_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, _out, err = _run(
        capsys, ["backtest", "--strategy", "sma_cross", "--data", str(tmp_path / "absent.csv")]
    )
    assert code == 1
    assert "introuvable" in err


def test_backtest_cost_overrides(csv_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _err = _run(
        capsys,
        [
            "backtest",
            "--strategy",
            "sma_cross",
            "--data",
            str(csv_path),
            "--initial-cash",
            "5000",
            "--fee",
            "0",
            "--slippage",
            "0.002",
        ],
    )
    assert code == 0
    assert "Capital initial      : 5 000.00" in out
    assert "frais 0.00 %, glissement 0.20 %" in out


def test_backtest_strategy_and_params_from_config(
    csv_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        '[strategy]\nname = "sma_cross"\n[strategy.params]\nfast = 7\nslow = 21\n'
        "[risk]\ninitial_cash = 2500.0\n[costs]\nfee_rate = 0.002\n",
        encoding="utf-8",
    )
    code, out, _err = _run(capsys, ["backtest", "--data", str(csv_path), "--config", str(config)])
    assert code == 0
    assert "Rapport de backtest : sma_cross" in out
    assert "fast=7, slow=21" in out
    assert "Capital initial      : 2 500.00" in out
    assert "frais 0.20 %" in out
    # --param prime sur [strategy.params]
    code, out, _err = _run(
        capsys, ["backtest", "--data", str(csv_path), "--config", str(config), "--param", "fast=9"]
    )
    assert code == 0
    assert "fast=9, slow=21" in out


def test_backtest_config_params_ignored_for_another_strategy(
    csv_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        '[strategy]\nname = "sma_cross"\n[strategy.params]\nfast = 7\n', encoding="utf-8"
    )
    code, out, _err = _run(
        capsys,
        [
            "backtest",
            "--data",
            str(csv_path),
            "--config",
            str(config),
            "--strategy",
            "donchian_breakout",
        ],
    )
    assert code == 0
    assert "Rapport de backtest : donchian_breakout" in out
    assert "entry_window=20, exit_window=10" in out


def test_backtest_from_exchange_uses_cache(
    fake_exchange: FakeExchange, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cache_dir = tmp_path / "cache"
    argv = [
        "backtest",
        "--strategy",
        "sma_cross",
        "--exchange",
        "binance",
        "--symbol",
        "BTC/USDT",
        "--timeframe",
        "1h",
        "--cache-dir",
        str(cache_dir),
    ]
    code, out, _err = _run(capsys, argv)
    assert code == 0
    assert "Données : BTC/USDT 1h sur binance" in out
    assert "700 bougies" in out
    assert "téléchargées et mises en cache" in out
    assert (cache_dir / "binance_BTC-USDT_1h.csv").is_file()
    assert len(fake_exchange.ohlcv_calls) == 1
    code, out, _err = _run(capsys, argv)
    assert code == 0
    assert len(fake_exchange.ohlcv_calls) == 1  # deuxième fois : cache, aucun appel
    assert "cache réutilisé tel quel, aucun téléchargement" in out  # F32 : dit clairement
    time.sleep(0.01)  # mtime distinct pour le téléchargement forcé
    code, out, _err = _run(capsys, [*argv, "--refresh"])
    assert code == 0
    assert len(fake_exchange.ohlcv_calls) == 2
    assert "cache complété / mis à jour" in out


def test_backtest_exchange_defaults_from_config(
    fake_exchange: FakeExchange, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        '[general]\nexchange = "kraken"\nsymbol = "ETH/USD"\ntimeframe = "4h"\n', encoding="utf-8"
    )
    code, out, _err = _run(
        capsys,
        [
            "backtest",
            "--strategy",
            "sma_cross",
            "--config",
            str(config),
            "--cache-dir",
            str(tmp_path / "c"),
        ],
    )
    assert code == 0
    assert "Données : ETH/USD 4h sur kraken" in out
    assert fake_exchange.ohlcv_calls[0][:2] == ("ETH/USD", "4h")


def test_backtest_exchange_with_since_filters(
    fake_exchange: FakeExchange, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _err = _run(
        capsys,
        [
            "backtest",
            "--strategy",
            "sma_cross",
            "--exchange",
            "binance",
            "--symbol",
            "BTC/USDT",
            "--timeframe",
            "1h",
            "--since",
            "2024-01-10",
            "--cache-dir",
            str(tmp_path / "c"),
        ],
    )
    assert code == 0
    # 700 bougies depuis le 1er janvier : à partir du 10 janvier il en reste 700 - 9*24 = 484
    assert "484 bougies" in out


# ============================================================================ download
def test_download_writes_cache_and_copy(
    fake_exchange: FakeExchange, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cache_dir = tmp_path / "cache"
    out_csv = tmp_path / "copie.csv"
    code, out, _err = _run(
        capsys,
        [
            "download",
            "--exchange",
            "binance",
            "--symbol",
            "BTC/USDT",
            "--timeframe",
            "1h",
            "--cache-dir",
            str(cache_dir),
            "--out",
            str(out_csv),
        ],
    )
    assert code == 0
    assert "700 bougies" in out
    cache_file = cache_dir / "binance_BTC-USDT_1h.csv"
    assert cache_file.is_file() and str(cache_file) in out
    assert out_csv.is_file()
    loaded = load_csv(out_csv)
    assert len(loaded) == 700
    np.testing.assert_allclose(loaded["close"].to_numpy(), fake_exchange.df["close"].to_numpy())
    assert loaded.index.equals(fake_exchange.df.index)


def test_download_empty_result_exits_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    empty = FakeExchange(rising_ohlcv(5).iloc[0:0])
    monkeypatch.setattr(cli, "build_public_exchange", lambda exchange_id: empty)
    code, _out, err = _run(
        capsys,
        [
            "download",
            "--exchange",
            "binance",
            "--symbol",
            "XXX/YYY",
            "--timeframe",
            "1h",
            "--cache-dir",
            str(tmp_path),
        ],
    )
    assert code == 1
    assert "Aucune bougie" in err


# ============================================================================ compare
def test_compare_csv(csv_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _err = _run(capsys, ["compare", "--data", str(csv_path)])
    assert code == 0
    assert "Comparaison des stratégies" in out
    for name in STRATEGIES:
        assert name in out
    assert "Buy & hold" in out
    assert "Lecture :" in out


def test_compare_subset_and_report_dir(
    csv_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = tmp_path / "cmp"
    code, out, _err = _run(
        capsys,
        [
            "compare",
            "--data",
            str(csv_path),
            "--strategies",
            "sma_cross,donchian_breakout",
            "--report-dir",
            str(out_dir),
        ],
    )
    assert code == 0
    assert "sma_cross" in out and "donchian_breakout" in out
    assert "ema_trend" not in out.split("Comparaison des stratégies")[1].split("Lecture")[0]
    assert (out_dir / "comparison.txt").is_file()
    assert (out_dir / "sma_cross" / "summary.txt").is_file()
    assert (out_dir / "donchian_breakout" / "equity.csv").is_file()
    assert not (out_dir / "ema_trend").exists()


def test_compare_unknown_strategy_in_subset(
    csv_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _out, err = _run(
        capsys, ["compare", "--data", str(csv_path), "--strategies", "sma_cross,zzz"]
    )
    assert code == 1
    assert "Stratégie inconnue" in err


# ============================================================================ optimize
def test_optimize_small_windows(
    csv_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = tmp_path / "wf"
    code, out, _err = _run(
        capsys,
        [
            "optimize",
            "--strategy",
            "sma_cross",
            "--data",
            str(csv_path),
            "--train-bars",
            "200",
            "--test-bars",
            "100",
            "--report-dir",
            str(out_dir),
        ],
    )
    assert code == 0
    assert "Walk-forward : sma_cross" in out
    assert "Fenêtres : 4 (entraînement 200 bougies, test 100 bougies" in out
    assert "Ratio OOS / IS" in out
    assert "Détail des fenêtres" in out
    for name in ("walkforward.txt", "windows.csv", "oos_equity.csv", "oos_trades.csv"):
        assert (out_dir / name).is_file(), name
    windows = pd.read_csv(out_dir / "windows.csv")
    assert len(windows) == 4
    assert list(windows["test_start"]) == [200, 300, 400, 500]
    equity = pd.read_csv(out_dir / "oos_equity.csv")
    assert len(equity) == 400
    assert equity["equity"].iloc[0] == pytest.approx(1000.0)


def test_optimize_default_windows(csv_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # 600 bougies : default_windows -> test 600 // 7 = 85, train 600 - 4*85 = 260
    code, out, _err = _run(capsys, ["optimize", "--strategy", "sma_cross", "--data", str(csv_path)])
    assert code == 0
    assert "Fenêtres : entraînement 260 bougies, test 85 bougies" in out
    assert "Fenêtres : 4 (entraînement 260 bougies, test 85 bougies" in out


def test_optimize_metric_and_anchored(csv_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _err = _run(
        capsys,
        [
            "optimize",
            "--strategy",
            "sma_cross",
            "--data",
            str(csv_path),
            "--train-bars",
            "300",
            "--test-bars",
            "150",
            "--metric",
            "total_return",
            "--anchored",
        ],
    )
    assert code == 0
    assert "métrique : total_return" in out
    assert "ancré" in out


def test_optimize_windows_too_large_exits_1(
    csv_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _out, err = _run(
        capsys,
        [
            "optimize",
            "--strategy",
            "sma_cross",
            "--data",
            str(csv_path),
            "--train-bars",
            "500",
            "--test-bars",
            "200",
        ],
    )
    assert code == 1
    assert "Pas assez de bougies" in err


# ============================================================================ paper
def _paper_argv(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "paper",
        "--strategy",
        "sma_cross",
        "--exchange",
        "binance",
        "--symbol",
        "BTC/USDT",
        "--timeframe",
        "1h",
        "--state-dir",
        str(tmp_path / "state"),
        "--once",
        *extra,
    ]


def test_paper_once_opens_position_and_prints_state(
    fake_exchange: FakeExchange, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _err = _run(capsys, _paper_argv(tmp_path))
    assert code == 0
    assert "ARGENT FICTIF" in out
    assert "Exchange" in out and "binance" in out and "BTC/USDT" in out
    state = json.loads(out[out.index("{") :])
    assert state["action"] == "open"
    assert state["signal"] == 1 and state["position"] == 1
    assert state["amount"] > 0
    assert state["timestamp"] == fake_exchange.df.index[-1].isoformat()
    assert state["order"] == "paper-1"
    for key in ("timestamp", "signal", "action", "position", "equity", "reason"):
        assert key in state
    # sma_cross porte son stop ATR sur chaque bougie de la position : un robot démarré en
    # plein milieu d'une tendance entre donc AVEC le stop de la stratégie, et la taille est
    # basée sur le risque (1 % / distance au stop, plafonnée à 95 % de l'équité) ; prix
    # d'exécution = ticker × (1 + glissement 0,05 %).
    last = fake_exchange.fetch_ticker("BTC/USDT")["last"]
    assert state["stop_loss"] is not None and 0 < state["stop_loss"] < last
    expected_amount = min(0.95 * 1000.0 / last, 0.01 * 1000.0 / (last - state["stop_loss"]))
    assert state["amount"] == pytest.approx(expected_amount, rel=2e-3)
    assert state["amount"] * last <= 0.95 * 1000.0 * (1 + 1e-9)
    assert state["entry_price"] == pytest.approx(last * 1.0005)
    assert state["equity"] < 1000.0  # frais et glissement payés

    broker_file = tmp_path / "state" / "paper" / "binance_BTC-USDT_1h" / "account.json"
    assert broker_file.is_file()
    saved = json.loads(broker_file.read_text(encoding="utf-8"))
    assert len(saved["orders"]) == 1 and saved["orders"][0]["side"] == "buy"
    trader_dir = tmp_path / "state" / "paper" / "binance_BTC-USDT_1h"
    assert (trader_dir / "last_bar.json").is_file()
    assert (trader_dir / "position.json").is_file()
    assert (trader_dir / "risk.json").is_file()
    assert json.loads((trader_dir / "position.json").read_text(encoding="utf-8"))["side"] == 1
    # l'exchange n'a été interrogé que pour les bougies (lookback + 1) et le ticker
    assert fake_exchange.ohlcv_calls == [("BTC/USDT", "1h", None, 501)]
    assert fake_exchange.ticker_calls >= 1


def test_paper_once_twice_is_idempotent(
    fake_exchange: FakeExchange, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(capsys, _paper_argv(tmp_path))[0] == 0
    code, out, _err = _run(capsys, _paper_argv(tmp_path))
    assert code == 0
    state = json.loads(out[out.index("{") :])
    assert state["action"] == "skip"
    assert state["reason"] == "bougie déjà traitée"
    assert state["position"] == 1
    saved = json.loads(
        (tmp_path / "state" / "paper" / "binance_BTC-USDT_1h" / "account.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(saved["orders"]) == 1  # aucun ordre en double


def test_paper_once_ema_trend_has_stop_and_risk_based_size(
    fake_exchange: FakeExchange, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # ema_trend fournit un stop suiveur à CHAQUE bougie en position : l'entrée a un stop et
    # la taille vaut équité × risk_per_trade / |prix - stop| (1 % du capital risqué).
    from tradingbot.strategies import get_strategy

    # (atr_mult=10 : stop assez loin pour que la règle du 1 % soit plus contraignante que le
    # plafond de 95 % du capital)
    code, out, _err = _run(
        capsys, _paper_argv(tmp_path, "--strategy", "ema_trend", "--param", "atr_mult=10")
    )
    assert code == 0
    state = json.loads(out[out.index("{") :])
    assert state["action"] == "open" and state["position"] == 1
    window = fake_exchange.df.iloc[-501:]  # lookback 500 + 1 bougie demandées
    signals = get_strategy("ema_trend", atr_mult=10).generate_signals(window)
    expected_stop = float(signals["stop_loss"].iloc[-1])
    last = fake_exchange.fetch_ticker("BTC/USDT")["last"]
    assert state["stop_loss"] == pytest.approx(expected_stop)
    assert expected_stop < last
    risk_size = 1000.0 * 0.01 / (last - expected_stop)
    assert risk_size < 0.95 * 1000.0 / last  # la règle du risque est bien celle qui s'applique
    assert state["amount"] == pytest.approx(risk_size)
    assert state["amount"] * last < 0.95 * 1000.0


def test_paper_once_stop_hit_on_restart(
    fake_exchange: FakeExchange, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = _paper_argv(tmp_path, "--strategy", "ema_trend")
    assert _run(capsys, argv)[0] == 0
    fake_exchange.price = 1.0  # le prix s'effondre sous le stop suiveur
    code, out, _err = _run(capsys, argv)
    assert code == 0
    state = json.loads(out[out.index("{") :])
    assert state["action"] == "stop_loss"
    assert "Stop touché" in state["reason"]
    assert state["position"] == 0
    assert state["order"] == "paper-2"
    assert state["equity"] < 100.0  # vendu à 1.0 : presque tout est perdu, comme prévu


def test_paper_initial_cash_and_costs_overrides(
    fake_exchange: FakeExchange, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # 5 000 et non 250 : avec le stop ATR de sma_cross (~90 USDT sous le prix) et 1 % de
    # risque, une entrée sur 250 de capital ferait ~6,6 USDT de notionnel, sous le minimum
    # de 10 USDT, donc aucun trade.
    code, out, _err = _run(
        capsys, _paper_argv(tmp_path, "--initial-cash", "5000", "--fee", "0", "--slippage", "0")
    )
    assert code == 0
    assert "cash 5 000.00" in out
    state = json.loads(out[out.index("{") :])
    assert state["entry_price"] == pytest.approx(fake_exchange.fetch_ticker("BTC/USDT")["last"])


def test_paper_loop_with_max_iterations(
    fake_exchange: FakeExchange, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = [a for a in _paper_argv(tmp_path) if a != "--once"]
    code, out, _err = _run(capsys, [*argv, "--max-iterations", "2", "--poll", "0.01"])
    assert code == 0
    assert "Boucle démarrée" in out
    assert len(fake_exchange.ohlcv_calls) == 2
    assert (tmp_path / "state" / "paper" / "binance_BTC-USDT_1h" / "account.json").is_file()


def test_paper_short_signal_is_ignored_without_allow_short(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    falling = rising_ohlcv()
    falling = pd.DataFrame(
        {
            "open": falling["open"].to_numpy()[::-1],
            "high": falling["high"].to_numpy()[::-1],
            "low": falling["low"].to_numpy()[::-1],
            "close": falling["close"].to_numpy()[::-1],
            "volume": 1000.0,
        },
        index=falling.index,
    )
    exchange = FakeExchange(falling)
    monkeypatch.setattr(cli, "build_public_exchange", lambda exchange_id: exchange)
    code, out, _err = _run(capsys, _paper_argv(tmp_path))
    assert code == 0
    state = json.loads(out[out.index("{") :])
    assert state["signal"] == 0 and state["action"] == "hold" and state["position"] == 0
    assert "short ignoré" in state["reason"]


def test_paper_lookback_is_raised_above_warmup(
    fake_exchange: FakeExchange, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _out, _err = _run(capsys, _paper_argv(tmp_path, "--lookback", "10"))
    assert code == 0
    # chauffe sma_cross = 55 bougies, marge 50 -> 105 bougies demandées (+1 non clôturée)
    assert fake_exchange.ohlcv_calls == [("BTC/USDT", "1h", None, 106)]


def test_paper_data_source_error_is_reported_not_raised(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    class Down(FakeExchange):
        def fetch_ohlcv(self, *args: Any, **kwargs: Any) -> list[list[float]]:
            raise RuntimeError("API indisponible")

    monkeypatch.setattr(cli, "build_public_exchange", lambda exchange_id: Down(rising_ohlcv()))
    code, out, err = _run(capsys, _paper_argv(tmp_path))
    # F44 : l'état est affiché (pas d'exception), mais le code de sortie dit l'échec à un
    # script / cron qui enchaîne des `--once`
    assert code == 1
    state = json.loads(out[out.index("{") :])
    assert state["action"] == "error"
    assert "API indisponible" in state["reason"]
    assert "code de sortie 1" in err


# ============================================================================ live
def _live_config(tmp_path: Path, mode: str = "live") -> Path:
    config = tmp_path / "config.toml"
    config.write_text(
        f'[general]\nmode = "{mode}"\nexchange = "binance"\nsymbol = "BTC/USDT"\n'
        'timeframe = "1h"\nstate_dir = "etat"\n[strategy]\nname = "sma_cross"\n',
        encoding="utf-8",
    )
    return config


def _set_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRADINGBOT_API_KEY", "cle-de-test")
    monkeypatch.setenv("TRADINGBOT_API_SECRET", "secret-de-test")


def _no_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRADINGBOT_API_KEY", raising=False)
    monkeypatch.delenv("TRADINGBOT_API_SECRET", raising=False)


def _forbid_real_broker(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("CcxtBroker ne doit pas être construit")

    monkeypatch.setattr(cli, "CcxtBroker", boom)


def test_live_without_flag_exits_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _no_keys(monkeypatch)
    _forbid_real_broker(monkeypatch)
    code, out, err = _run(capsys, ["live"])
    assert code == 2
    assert "REFUS" in err
    assert "--i-understand-the-risks" in err
    assert 'mode = "paper"' in err
    assert "TRADINGBOT_API_KEY" in err
    assert "Aucun ordre" in err
    assert out == ""


def test_live_with_flag_but_paper_mode_exits_2(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _set_keys(monkeypatch)
    _forbid_real_broker(monkeypatch)
    config = _live_config(tmp_path, mode="paper")
    code, _out, err = _run(
        capsys, ["live", "--config", str(config), "--i-understand-the-risks", "--yes"]
    )
    assert code == 2
    assert 'mode = "paper"' in err and 'mode = "live"' in err
    assert "--i-understand-the-risks" not in err  # cette condition-là est remplie
    assert "TRADINGBOT_API_KEY" not in err


def test_live_without_keys_exits_2(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _no_keys(monkeypatch)
    _forbid_real_broker(monkeypatch)
    config = _live_config(tmp_path)
    code, _out, err = _run(
        capsys, ["live", "--config", str(config), "--i-understand-the-risks", "--yes"]
    )
    assert code == 2
    assert "TRADINGBOT_API_KEY" in err and "JAMAIS le retrait" in err
    assert 'écrivez mode = "live"' not in err  # le mode est bon : pas de refus à ce titre
    assert "--i-understand-the-risks" not in err


def test_live_recap_then_abort_without_oui(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _set_keys(monkeypatch)
    _forbid_real_broker(monkeypatch)
    config = _live_config(tmp_path)
    monkeypatch.setattr("builtins.input", lambda prompt="": "non")
    code, out, _err = _run(capsys, ["live", "--config", str(config), "--i-understand-the-risks"])
    assert code == 2
    assert "ARGENT RÉEL" in out
    assert "binance" in out and "BTC/USDT" in out and "1h" in out and "sma_cross" in out
    assert "Kill switch (drawdown)" in out and "20.00 %" in out
    assert "Limite de perte journalière" in out and "5.00 %" in out
    assert "Risque par trade" in out and "1.00 %" in out
    assert "Annulé" in out


def test_live_recap_then_abort_on_eof(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _set_keys(monkeypatch)
    _forbid_real_broker(monkeypatch)
    config = _live_config(tmp_path)

    def no_tty(prompt: str = "") -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", no_tty)
    code, out, _err = _run(capsys, ["live", "--config", str(config), "--i-understand-the-risks"])
    assert code == 2
    assert "Annulé" in out


def test_live_keys_from_dotenv_next_to_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # .env à côté de config.toml ; os.environ remplacé par une copie pour ne rien laisser
    monkeypatch.setattr(
        os, "environ", {k: v for k, v in os.environ.items() if not k.startswith("TRADINGBOT_")}
    )
    _forbid_real_broker(monkeypatch)
    config = _live_config(tmp_path)
    (tmp_path / ".env").write_text(
        "TRADINGBOT_API_KEY=abc\nTRADINGBOT_API_SECRET=def\n", encoding="utf-8"
    )
    monkeypatch.setattr("builtins.input", lambda prompt="": "non")
    code, out, err = _run(capsys, ["live", "--config", str(config), "--i-understand-the-risks"])
    assert code == 2
    assert "REFUS" not in err
    assert "ARGENT RÉEL" in out and "Annulé" in out
    assert "abc" not in out and "def" not in out  # les secrets ne sont jamais affichés


def test_live_yes_with_fake_broker_runs_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _set_keys(monkeypatch)
    exchange = FakeExchange(rising_ohlcv())
    monkeypatch.setattr(cli, "build_public_exchange", lambda exchange_id: exchange)
    FakeCcxtBroker.exchange_for_prices = exchange
    monkeypatch.setattr(cli, "CcxtBroker", FakeCcxtBroker)
    config = _live_config(tmp_path)
    code, out, _err = _run(
        capsys, ["live", "--config", str(config), "--i-understand-the-risks", "--yes", "--once"]
    )
    assert code == 0, out
    assert len(FakeCcxtBroker.created) == 1
    created = FakeCcxtBroker.created[0]
    assert created["exchange_id"] == "binance"
    assert created["api_key"] == "cle-de-test" and created["api_secret"] == "secret-de-test"
    assert created["confirm_live"] is True
    assert created["sandbox"] is False
    assert "ARGENT RÉEL" in out and "Connecté à binance" in out
    assert "cle-de-test" not in out and "secret-de-test" not in out
    state = json.loads(out[out.index("{") :])
    assert state["action"] == "open" and state["position"] == 1
    trader_dir = tmp_path / "etat" / "live" / "binance_BTC-USDT_1h"
    assert (trader_dir / "last_bar.json").is_file()


def test_live_oui_typed_proceeds_and_sandbox_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _set_keys(monkeypatch)
    exchange = FakeExchange(rising_ohlcv())
    monkeypatch.setattr(cli, "build_public_exchange", lambda exchange_id: exchange)
    FakeCcxtBroker.exchange_for_prices = exchange
    monkeypatch.setattr(cli, "CcxtBroker", FakeCcxtBroker)
    monkeypatch.setattr("builtins.input", lambda prompt="": "OUI")
    config = _live_config(tmp_path)
    code, out, _err = _run(
        capsys, ["live", "--config", str(config), "--i-understand-the-risks", "--sandbox", "--once"]
    )
    assert code == 0, out
    assert FakeCcxtBroker.created[0]["sandbox"] is True
    assert "SANDBOX" in out
    assert (tmp_path / "etat" / "sandbox" / "binance_BTC-USDT_1h" / "last_bar.json").is_file()


def test_live_broker_error_at_connection_exits_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from tradingbot.execution.broker import BrokerError

    _set_keys(monkeypatch)

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise BrokerError("clés API refusées par l'exchange")

    monkeypatch.setattr(cli, "CcxtBroker", refuse)
    config = _live_config(tmp_path)
    code, _out, err = _run(
        capsys, ["live", "--config", str(config), "--i-understand-the-risks", "--yes"]
    )
    assert code == 1
    assert "clés API refusées" in err


# ============================================================================ journalisation
def _record(msg: str, level: int = logging.WARNING) -> logging.LogRecord:
    return logging.LogRecord("tradingbot.x", level, __file__, 1, msg, None, None)


def test_dedup_filter_shows_each_message_once_when_enabled() -> None:
    flt = cli._DedupFilter()
    flt.reset(enabled=True)
    assert flt.filter(_record("matplotlib absent")) is True
    assert flt.filter(_record("matplotlib absent")) is False
    assert flt.filter(_record("autre message")) is True
    assert flt.filter(_record("matplotlib absent", logging.ERROR)) is True  # niveau différent
    flt.reset(enabled=False)
    assert flt.filter(_record("matplotlib absent")) is True
    assert flt.filter(_record("matplotlib absent")) is True


def test_configure_logging_single_handler_and_dedup_toggle() -> None:
    cli._configure_logging(None, "warning", dedup=True)
    pkg = logging.getLogger("tradingbot")
    handlers = [h for h in pkg.handlers if getattr(h, "_tradingbot_cli", False)]
    assert len(handlers) == 1
    flt = next(f for f in handlers[0].filters if isinstance(f, cli._DedupFilter))
    assert flt.enabled is True
    assert pkg.level == logging.WARNING
    # une commande paper/live : niveau explicite, pas de dédoublonnage, même handler
    cli._configure_logging("info", "warning", dedup=False)
    assert flt.enabled is False
    assert pkg.level == logging.INFO
    assert [h for h in pkg.handlers if getattr(h, "_tradingbot_cli", False)] == handlers


def test_offline_commands_dedup_logs_and_paper_live_do_not() -> None:
    parser = cli.build_parser()
    for command in ("list-strategies", "demo", "backtest", "compare", "optimize"):
        assert parser.parse_args([command]).dedup_logs is True, command
    for command in ("download", "paper", "live"):
        assert getattr(parser.parse_args([command]), "dedup_logs", False) is False, command


# ============================================================================ F32 : cache et --since
def test_download_with_since_before_cache_reports_cache_reuse(
    fake_exchange: FakeExchange, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cache_dir = tmp_path / "cache"
    base = ["download", "--exchange", "binance", "--symbol", "BTC/USDT", "--timeframe", "1h"]
    code, out, _err = _run(capsys, [*base, "--cache-dir", str(cache_dir)])
    assert code == 0
    assert "téléchargement..." in out and "téléchargées et mises en cache" in out
    # --since antérieur au cache : le chargeur demande le complément ; l'exchange (faux) n'a
    # rien d'antérieur à janvier 2024, le cache reste donc inchangé, et la sortie le DIT au
    # lieu d'annoncer un téléchargement « depuis 2023-12-01 »
    code, out, err = _run(capsys, [*base, "--cache-dir", str(cache_dir), "--since", "2023-12-01"])
    assert code == 0
    assert "cache existant, seul le complément manquant sera téléchargé" in out
    assert "cache réutilisé tel quel, aucun téléchargement" in out
    assert "Téléchargement de" not in out
    assert len(fake_exchange.ohlcv_calls) == 2  # un appel pour le complément (vide)
    assert "inchangé" in err  # avertissement du chargeur : rien de plus ancien disponible
    assert "--refresh" in out


# ============================================================================ F29 : chat Telegram
def test_build_notifier_never_logs_the_telegram_chat_id(caplog: pytest.LogCaptureFixture) -> None:
    from tradingbot.config import NotificationSettings, Settings

    settings = Settings(
        notifications=NotificationSettings(telegram=True),
        telegram_token="123:ABC",
        telegram_chat_id="987654321",
    )
    caplog.set_level(logging.DEBUG, logger="tradingbot")
    notifier = cli._build_notifier(settings)
    assert type(notifier).__name__ == "MultiNotifier"
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "Telegram activées" in text
    assert "987654321" not in text and "123:ABC" not in text


# ============================================================================ F36 : capital affiché
def test_paper_recap_shows_effective_initial_cash(
    fake_exchange: FakeExchange, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "config.toml"
    config.write_text("[risk]\ninitial_cash = 1000.0\n", encoding="utf-8")
    code, out, _err = _run(
        capsys, _paper_argv(tmp_path, "--config", str(config), "--initial-cash", "5000")
    )
    assert code == 0
    recap = out[: out.index("{")]
    assert "Capital de référence" in recap
    assert "5 000.00" in recap.split("Capital de référence")[1].splitlines()[0]
    assert "1 000.00" not in recap  # plus deux capitaux contradictoires à l'écran
    assert "cash 5 000.00" in recap


# ============================================================================ récapitulatif
def test_paper_recap_shows_stop_rule_reentry_and_daily_limit(
    fake_exchange: FakeExchange, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _err = _run(capsys, _paper_argv(tmp_path))
    assert code == 0
    recap = out[: out.index("{")]
    assert "Stop de protection" in recap and "3 × ATR(14)" in recap and "5.00 % du prix" in recap
    assert "Ré-entrée après stop" in recap and "(wait)" in recap
    assert "Limite de perte journalière : 5.00 %" in recap
    assert "INACTIVE" not in recap  # bougies d'une heure : la limite s'applique
    assert "verrou" in recap
    assert "Position existante          : aucune (BTC absent du compte)" in recap


def test_paper_recap_daily_limit_inactive_on_daily_candles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    daily = rising_ohlcv(400)
    # 400 bougies journalières qui se terminent hier (le chargeur ne demande à l'exchange que
    # les bougies récentes : « maintenant - (lookback + 1) jours »)
    start = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=400)
    daily.index = pd.date_range(start=start, periods=len(daily), freq="1D", tz="UTC")
    exchange = FakeExchange(daily)
    monkeypatch.setattr(cli, "build_public_exchange", lambda exchange_id: exchange)
    argv = [a if a != "1h" else "1d" for a in _paper_argv(tmp_path)]
    code, out, _err = _run(capsys, argv)
    assert code == 0
    recap = out[: out.index("{")]
    assert "Limite de perte journalière : 5.00 % — INACTIVE sur ce timeframe" in recap
    assert "bougies >= 1 jour" in recap


# ============================================================================ F10 : short en live
def test_live_refuses_allow_short_from_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _set_keys(monkeypatch)
    _forbid_real_broker(monkeypatch)
    config = _live_config(tmp_path)
    config.write_text(config.read_text(encoding="utf-8") + "[risk]\nallow_short = true\n")
    code, out, err = _run(
        capsys, ["live", "--config", str(config), "--i-understand-the-risks", "--yes"]
    )
    assert code == 2
    assert "REFUS" in err and "vente à découvert" in err and "allow_short = false" in err
    assert "comptant" in err
    assert out == ""  # aucune connexion, aucun récapitulatif


def test_live_refuses_allow_short_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _set_keys(monkeypatch)
    _forbid_real_broker(monkeypatch)
    config = _live_config(tmp_path)
    code, _out, err = _run(
        capsys,
        ["live", "--config", str(config), "--i-understand-the-risks", "--yes", "--allow-short"],
    )
    assert code == 2
    assert "--allow-short" in err and "vente à découvert" in err
    assert 'écrivez mode = "live"' not in err  # les autres verrous sont levés


def test_paper_allow_short_is_a_simulation_with_a_warning(
    fake_exchange: FakeExchange,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="tradingbot.cli")
    code, out, _err = _run(capsys, _paper_argv(tmp_path, "--allow-short"))
    assert code == 0
    recap = out[: out.index("{")]
    assert "Vente à découvert" in recap and "simulation seulement" in recap
    assert any("simulation" in r.getMessage() and "live" in r.getMessage() for r in caplog.records)


# ============================================================================ F7 : verrou
def _paper_state_dir(tmp_path: Path) -> Path:
    return tmp_path / "state" / "paper" / "binance_BTC-USDT_1h"


def test_paper_refuses_when_state_dir_is_locked_by_another_robot(
    fake_exchange: FakeExchange, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from tradingbot.live.runner import StateDirLock

    other = StateDirLock(_paper_state_dir(tmp_path))
    other.acquire()
    try:
        code, out, err = _run(capsys, _paper_argv(tmp_path))
        assert code == 2
        assert "REFUS" in err and "déjà utilisé par un autre robot" in err
        assert str(os.getpid()) in err  # le PID du détenteur aide à le retrouver
        assert "{" not in out  # aucune itération n'a eu lieu
        assert not (_paper_state_dir(tmp_path) / "position.json").exists()
    finally:
        other.release()
    # l'autre robot s'arrête : la commande passe
    assert _run(capsys, _paper_argv(tmp_path))[0] == 0


def test_paper_once_releases_the_lock_before_returning(
    fake_exchange: FakeExchange, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from tradingbot.live.runner import StateDirLock

    assert _run(capsys, _paper_argv(tmp_path))[0] == 0
    lock = StateDirLock(_paper_state_dir(tmp_path))
    lock.acquire()  # ne lève pas : le verrou a été rendu par --once
    assert lock.held
    lock.release()


# ============================================================================ F45 : Ctrl-C → 130
def test_paper_loop_interrupted_exits_130_and_releases_lock(
    fake_exchange: FakeExchange,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from tradingbot.live.runner import LiveTrader, StateDirLock

    def ctrl_c(seconds: float) -> None:
        raise KeyboardInterrupt

    class InterruptedTrader(LiveTrader):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["sleep"] = ctrl_c  # Ctrl-C pendant l'attente entre deux interrogations
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(cli, "LiveTrader", InterruptedTrader)
    argv = [a for a in _paper_argv(tmp_path) if a != "--once"]
    code, out, err = _run(capsys, [*argv, "--max-iterations", "3"])
    assert code == 130
    assert "Boucle démarrée" in out
    assert "Arrêt demandé (Ctrl-C)" in err
    assert len(fake_exchange.ohlcv_calls) == 1  # une itération, puis l'interruption
    # la position fictive est mémorisée et le verrou rendu
    assert json.loads((_paper_state_dir(tmp_path) / "position.json").read_text())["side"] == 1
    lock = StateDirLock(_paper_state_dir(tmp_path))
    lock.acquire()
    lock.release()


# ============================================================================ --adopt-position
def _seed_paper_account_with_btc(tmp_path: Path, amount: float = 0.5) -> Path:
    """Compte fictif qui détient déjà du BTC alors que le robot n'a aucune position mémorisée."""
    account = _paper_state_dir(tmp_path) / "account.json"
    account.parent.mkdir(parents=True)
    broker = PaperBroker(initial_cash=1000.0, state_path=account)
    broker.set_price("BTC/USDT", 100.0)
    broker.market_order("BTC/USDT", "buy", amount)
    broker.save()
    return account


def test_paper_existing_balance_not_adopted_without_flag(
    fake_exchange: FakeExchange, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _seed_paper_account_with_btc(tmp_path)
    code, out, _err = _run(capsys, _paper_argv(tmp_path))
    assert code == 0
    recap = out[: out.index("{")]
    assert "Position existante          : 0.5 BTC NON gérés par le robot" in recap
    assert "--adopt-position" in recap
    state = json.loads(out[out.index("{") :])
    assert state["signal"] == 1 and state["action"] == "refused" and state["position"] == 0
    assert "position existante non gérée" in state["reason"]
    saved = json.loads((_paper_state_dir(tmp_path) / "account.json").read_text(encoding="utf-8"))
    assert len(saved["orders"]) == 1  # rien vendu, rien acheté


def test_paper_adopt_position_flag_hands_the_balance_to_the_robot(
    fake_exchange: FakeExchange, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _seed_paper_account_with_btc(tmp_path)
    code, out, _err = _run(capsys, _paper_argv(tmp_path, "--adopt-position"))
    assert code == 0
    recap = out[: out.index("{")]
    assert "Position existante          : 0.5 BTC CONFIÉS au robot (--adopt-position)" in recap
    state = json.loads(out[out.index("{") :])
    assert state["signal"] == 1 and state["position"] == 1
    assert state["action"] == "hold" and state["amount"] == pytest.approx(0.5)
    assert state["stop_loss"] is not None  # le stop de la stratégie protège la position adoptée
    record = json.loads((_paper_state_dir(tmp_path) / "position.json").read_text(encoding="utf-8"))
    assert record["side"] == 1 and record["amount"] == pytest.approx(0.5)
    # au redémarrage, c'est désormais la position DU robot (position.json), plus un solde étranger
    code, out, _err = _run(capsys, _paper_argv(tmp_path))
    assert code == 0
    assert "position du robot reprise de position.json" in out[: out.index("{")]


def test_adopt_position_flag_is_accepted_by_paper_and_live_parsers() -> None:
    parser = cli.build_parser()
    for command in ("paper", "live"):
        args = parser.parse_args([command, "--adopt-position"])
        assert args.adopt_position is True, command
        assert parser.parse_args([command]).adopt_position is False, command
