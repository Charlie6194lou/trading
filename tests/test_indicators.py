"""Tests de `tradingbot.indicators` : valeurs calculées à la main, implémentations de
référence scalaires indépendantes (boucles) et absence totale de look-ahead.

Aucun accès réseau, aucune dépendance aux autres modules du projet.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from conftest import make_ohlcv

from tradingbot import indicators as ind

# --- helpers ------------------------------------------------------------------------


def _series(values: list[float]) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=len(values), freq="1h", tz="UTC")
    return pd.Series(values, index=idx, dtype="float64")


def _bars(rows: list[tuple[float, float, float]]) -> pd.DataFrame:
    """Construit un DataFrame OHLCV à partir de tuples (high, low, close)."""
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="1h", tz="UTC")
    high = [r[0] for r in rows]
    low = [r[1] for r in rows]
    close = [r[2] for r in rows]
    return pd.DataFrame(
        {
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": [1.0] * len(rows),
        },
        index=idx,
        dtype="float64",
    )


def _ref_wilder(values: list[float], window: int) -> list[float]:
    """Lissage de Wilder de référence (boucle scalaire) sur des valeurs sans NaN."""
    out = [math.nan] * len(values)
    if len(values) < window:
        return out
    acc = sum(values[:window]) / window
    out[window - 1] = acc
    for i in range(window, len(values)):
        acc = (acc * (window - 1) + values[i]) / window
        out[i] = acc
    return out


def _ref_rsi(closes: list[float], window: int) -> list[float]:
    """RSI de Wilder de référence (boucle scalaire)."""
    n = len(closes)
    out = [math.nan] * n
    if n <= window:
        return out
    deltas = [closes[i] - closes[i - 1] for i in range(1, n)]
    gains = [max(d, 0.0) for d in deltas]
    losses = [max(-d, 0.0) for d in deltas]
    ag = sum(gains[:window]) / window
    al = sum(losses[:window]) / window

    def _rsi(g: float, lo: float) -> float:
        return 50.0 if g + lo == 0 else 100.0 * g / (g + lo)

    out[window] = _rsi(ag, al)
    for i in range(window, len(deltas)):
        ag = (ag * (window - 1) + gains[i]) / window
        al = (al * (window - 1) + losses[i]) / window
        out[i + 1] = _rsi(ag, al)
    return out


def _ref_true_range(df: pd.DataFrame) -> list[float]:
    h, lo, c = df["high"].tolist(), df["low"].tolist(), df["close"].tolist()
    tr = [math.nan]
    for i in range(1, len(df)):
        tr.append(max(h[i] - lo[i], abs(h[i] - c[i - 1]), abs(lo[i] - c[i - 1])))
    return tr


def _ref_atr(df: pd.DataFrame, window: int) -> list[float]:
    tr = _ref_true_range(df)
    return [math.nan] + _ref_wilder(tr[1:], window)


def _ref_adx(df: pd.DataFrame, window: int) -> tuple[list[float], list[float], list[float]]:
    """ADX de Wilder de référence (boucle scalaire) : (adx, plus_di, minus_di)."""
    h, lo = df["high"].tolist(), df["low"].tolist()
    n = len(df)
    tr = _ref_true_range(df)[1:]
    plus_dm: list[float] = []
    minus_dm: list[float] = []
    for i in range(1, n):
        up = h[i] - h[i - 1]
        down = lo[i - 1] - lo[i]
        plus_dm.append(up if (up > down and up > 0) else 0.0)
        minus_dm.append(down if (down > up and down > 0) else 0.0)
    s_tr = _ref_wilder(tr, window)
    s_p = _ref_wilder(plus_dm, window)
    s_m = _ref_wilder(minus_dm, window)
    plus_di = [math.nan] * n
    minus_di = [math.nan] * n
    dx: list[float] = []
    for j in range(len(tr)):
        if math.isnan(s_tr[j]):
            continue
        p = 0.0 if s_tr[j] == 0 else 100.0 * s_p[j] / s_tr[j]
        m = 0.0 if s_tr[j] == 0 else 100.0 * s_m[j] / s_tr[j]
        plus_di[j + 1] = p
        minus_di[j + 1] = m
        dx.append(0.0 if p + m == 0 else 100.0 * abs(p - m) / (p + m))
    first_dx = n - len(dx)
    adx_vals = _ref_wilder(dx, window)
    adx = [math.nan] * first_dx + adx_vals
    return adx, plus_di, minus_di


# --- sma / ema ----------------------------------------------------------------------


class TestSma:
    def test_hand_computed(self) -> None:
        s = _series([1, 2, 3, 4, 5, 6])
        out = ind.sma(s, 3)
        expected = [np.nan, np.nan, 2.0, 3.0, 4.0, 5.0]
        np.testing.assert_allclose(out.to_numpy(), expected, equal_nan=True)
        assert out.index.equals(s.index)
        assert out.name == "sma_3"

    def test_window_one_is_identity(self) -> None:
        s = _series([3.0, 1.5, 2.5])
        np.testing.assert_allclose(ind.sma(s, 1).to_numpy(), s.to_numpy())

    def test_short_series_all_nan(self) -> None:
        s = _series([1.0, 2.0])
        assert ind.sma(s, 5).isna().all()
        assert len(ind.sma(s, 5)) == 2

    def test_invalid_window(self) -> None:
        s = _series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            ind.sma(s, 0)
        with pytest.raises(TypeError):
            ind.sma(s, 2.5)  # type: ignore[arg-type]

    def test_does_not_mutate_input(self) -> None:
        s = _series([1, 2, 3, 4])
        before = s.copy()
        ind.sma(s, 2)
        pd.testing.assert_series_equal(s, before)


class TestEma:
    def test_hand_computed_span_3(self) -> None:
        # alpha = 2 / (3 + 1) = 0.5
        s = _series([10, 11, 12, 11, 13])
        out = ind.ema(s, 3)
        expected = [10.0, 10.5, 11.25, 11.125, 12.0625]
        np.testing.assert_allclose(out.to_numpy(), expected)
        assert out.name == "ema_3"

    @pytest.mark.parametrize("span", [2, 5, 20])
    def test_matches_manual_recursion(self, span: int) -> None:
        close = make_ohlcv(120, seed=1)["close"]
        alpha = 2.0 / (span + 1)
        values = close.tolist()
        manual = [values[0]]
        for x in values[1:]:
            manual.append(alpha * x + (1 - alpha) * manual[-1])
        np.testing.assert_allclose(ind.ema(close, span).to_numpy(), manual, rtol=1e-12)

    def test_leading_nan_preserved(self) -> None:
        s = _series([np.nan, np.nan, 4.0, 6.0])
        out = ind.ema(s, 3)
        np.testing.assert_allclose(out.to_numpy(), [np.nan, np.nan, 4.0, 5.0], equal_nan=True)


# --- returns / volatility -------------------------------------------------------------


class TestReturns:
    def test_simple(self) -> None:
        s = _series([100, 110, 99])
        np.testing.assert_allclose(ind.returns(s).to_numpy(), [np.nan, 0.1, -0.1], equal_nan=True)

    def test_log(self) -> None:
        s = _series([100, 110, 99])
        expected = [np.nan, math.log(1.1), math.log(0.9)]
        np.testing.assert_allclose(ind.returns(s, log=True).to_numpy(), expected, equal_nan=True)

    def test_log_non_positive_price_is_nan(self) -> None:
        s = _series([100, 0, 50])
        out = ind.returns(s, log=True)
        assert out.isna().all()


class TestRollingVolatility:
    def test_hand_computed_two_returns(self) -> None:
        s = _series([100, 110, 99, 108.9])
        # rendements log : ln(1.1), ln(0.9), ln(1.1) ; écart-type (ddof=1) de deux
        # valeurs = |x1 - x2| / sqrt(2)
        sd = abs(math.log(1.1) - math.log(0.9)) / math.sqrt(2)
        out = ind.rolling_volatility(s, 2)
        np.testing.assert_allclose(out.to_numpy(), [np.nan, np.nan, sd, sd], equal_nan=True)

    def test_annualised(self) -> None:
        s = _series([100, 110, 99, 108.9])
        sd = abs(math.log(1.1) - math.log(0.9)) / math.sqrt(2)
        out = ind.rolling_volatility(s, 2, periods_per_year=8760)
        np.testing.assert_allclose(out.iloc[2], sd * math.sqrt(8760))
        plain = ind.rolling_volatility(s, 2)
        np.testing.assert_allclose((out / plain).dropna().to_numpy(), math.sqrt(8760), rtol=1e-12)

    def test_warmup_length(self) -> None:
        close = make_ohlcv(50)["close"]
        out = ind.rolling_volatility(close, 20)
        assert out.iloc[:20].isna().all()
        assert out.iloc[20:].notna().all()

    def test_constant_price_has_zero_vol(self) -> None:
        s = _series([5.0] * 10)
        out = ind.rolling_volatility(s, 3)
        np.testing.assert_allclose(out.dropna().to_numpy(), 0.0)

    def test_invalid_periods_per_year(self) -> None:
        with pytest.raises(ValueError):
            ind.rolling_volatility(_series([1.0, 2.0, 3.0]), 2, periods_per_year=0)


# --- Wilder smoothing / RSI -----------------------------------------------------------


class TestWilderSmooth:
    def test_hand_computed_with_leading_nan(self) -> None:
        s = _series([np.nan, 10.0, 4.0, 7.0, 1.0])
        out = ind.wilder_smooth(s, 3)
        # amorçage = (10 + 4 + 7) / 3 = 7 ; puis 7 + (1 - 7) / 3 = 5
        np.testing.assert_allclose(
            out.to_numpy(), [np.nan, np.nan, np.nan, 7.0, 5.0], equal_nan=True
        )

    def test_too_short_is_all_nan(self) -> None:
        out = ind.wilder_smooth(_series([1.0, 2.0]), 3)
        assert out.isna().all() and len(out) == 2

    def test_matches_reference_loop(self) -> None:
        close = make_ohlcv(100, seed=5)["close"]
        ref = _ref_wilder(close.tolist(), 7)
        np.testing.assert_allclose(
            ind.wilder_smooth(close, 7).to_numpy(), ref, rtol=1e-12, equal_nan=True
        )


class TestRsi:
    def test_hand_built_window_3(self) -> None:
        # deltas : +1, -0.5, +1, +0.5, -1
        s = _series([10, 11, 10.5, 11.5, 12, 11])
        out = ind.rsi(s, 3)
        # t=3 : gain_moy = (1+0+1)/3 = 2/3 ; perte_moy = (0+0.5+0)/3 = 1/6 -> 100*(2/3)/(5/6) = 80
        # t=4 : gain_moy = (2/3*2 + 0.5)/3 = 11/18 ; perte_moy = (1/6*2 + 0)/3 = 1/9 = 2/18
        #       -> 100 * 11/13
        # t=5 : gain_moy = (11/18*2 + 0)/3 = 11/27 ; perte_moy = (1/9*2 + 1)/3 = 11/27 -> 50
        expected = [np.nan, np.nan, np.nan, 80.0, 100.0 * 11 / 13, 50.0]
        np.testing.assert_allclose(out.to_numpy(), expected, equal_nan=True)
        assert out.name == "rsi_3"

    def test_textbook_14_period_example(self) -> None:
        # Exemple classique (StockCharts / Wilder) sur 33 clôtures. Les valeurs publiées
        # (70.53, 66.32, ...) sont obtenues à partir de clôtures non arrondies ; avec les
        # clôtures arrondies ci-dessous la récurrence exacte de Wilder donne 70.46, 66.25...
        closes = [
            44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89,
            46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64, 46.21, 46.25,
            45.71, 46.45, 45.78, 45.35, 44.03, 44.18, 44.22, 44.57, 43.42, 42.66, 43.13,
        ]  # fmt: skip
        s = _series(closes)
        out = ind.rsi(s, 14)
        assert out.iloc[:14].isna().all()
        assert out.iloc[14:].notna().all()
        ref = _ref_rsi(closes, 14)
        np.testing.assert_allclose(out.to_numpy(), ref, rtol=1e-12, equal_nan=True)
        assert out.iloc[14] == pytest.approx(70.46, abs=0.01)
        assert out.iloc[15] == pytest.approx(66.25, abs=0.01)
        assert out.iloc[-1] == pytest.approx(37.79, abs=0.01)
        assert out.min() == pytest.approx(33.09, abs=0.01)

    def test_bounds_on_random_data(self) -> None:
        close = make_ohlcv(500, seed=11)["close"]
        out = ind.rsi(close, 14)
        valid = out.dropna()
        assert len(valid) == 500 - 14
        assert (valid >= 0).all() and (valid <= 100).all()
        np.testing.assert_allclose(
            out.to_numpy(), _ref_rsi(close.tolist(), 14), rtol=1e-10, equal_nan=True
        )

    def test_monotonic_series_hits_bounds(self) -> None:
        up = _series(list(range(1, 30)))
        down = _series(list(range(30, 1, -1)))
        np.testing.assert_allclose(ind.rsi(up, 5).dropna().to_numpy(), 100.0)
        np.testing.assert_allclose(ind.rsi(down, 5).dropna().to_numpy(), 0.0)

    def test_flat_market_is_50(self) -> None:
        s = _series([5.0] * 12)
        out = ind.rsi(s, 4)
        assert out.iloc[:4].isna().all()
        np.testing.assert_allclose(out.iloc[4:].to_numpy(), 50.0)

    def test_short_series_all_nan(self) -> None:
        out = ind.rsi(_series([1.0, 2.0, 3.0]), 14)
        assert out.isna().all() and len(out) == 3


# --- ATR ------------------------------------------------------------------------------


class TestAtr:
    def test_true_range_hand_computed(self) -> None:
        df = _bars([(12, 10, 11), (13, 11, 12), (12, 9, 10), (15, 13, 14), (14, 12, 13)])
        tr = ind.true_range(df)
        # t=1 : max(2, |13-11|, |11-11|) = 2 ; t=2 : max(3, |12-12|, |9-12|) = 3
        # t=3 : max(2, |15-10|, |13-10|) = 5 (gap haussier) ; t=4 : max(2, 0, 2) = 2
        np.testing.assert_allclose(tr.to_numpy(), [np.nan, 2, 3, 5, 2], equal_nan=True)

    def test_hand_computed_window_3(self) -> None:
        df = _bars([(12, 10, 11), (13, 11, 12), (12, 9, 10), (15, 13, 14), (14, 12, 13)])
        out = ind.atr(df, 3)
        # ATR[3] = (2 + 3 + 5) / 3 = 10/3 ; ATR[4] = (10/3 * 2 + 2) / 3 = 26/9
        expected = [np.nan, np.nan, np.nan, 10 / 3, 26 / 9]
        np.testing.assert_allclose(out.to_numpy(), expected, equal_nan=True)
        assert out.name == "atr_3"

    def test_matches_reference_loop(self) -> None:
        df = make_ohlcv(200, seed=9)
        out = ind.atr(df, 14)
        np.testing.assert_allclose(out.to_numpy(), _ref_atr(df, 14), rtol=1e-10, equal_nan=True)
        assert out.iloc[:14].isna().all()
        assert (out.dropna() > 0).all()

    def test_missing_column(self) -> None:
        df = _bars([(12, 10, 11), (13, 11, 12)]).drop(columns=["low"])
        with pytest.raises(ValueError, match="low"):
            ind.atr(df, 2)


# --- MACD -----------------------------------------------------------------------------


class TestMacd:
    def test_hand_computed_first_values(self) -> None:
        s = _series([1, 2, 3])
        out = ind.macd(s, fast=2, slow=3, signal=2)
        # ema2 (alpha 2/3) : 1, 5/3, 23/9 ; ema3 (alpha 1/2) : 1, 1.5, 2.25
        macd_line = [0.0, 5 / 3 - 1.5, 23 / 9 - 2.25]
        # signal = ema2 de la ligne macd
        sig = [macd_line[0]]
        for x in macd_line[1:]:
            sig.append(2 / 3 * x + 1 / 3 * sig[-1])
        assert list(out.columns) == ["macd", "signal", "hist"]
        np.testing.assert_allclose(out["macd"].to_numpy(), macd_line)
        np.testing.assert_allclose(out["signal"].to_numpy(), sig)
        np.testing.assert_allclose(out["hist"].to_numpy(), np.array(macd_line) - np.array(sig))

    def test_consistent_with_ema(self) -> None:
        close = make_ohlcv(150, seed=2)["close"]
        out = ind.macd(close)
        expected_macd = ind.ema(close, 12) - ind.ema(close, 26)
        np.testing.assert_allclose(out["macd"].to_numpy(), expected_macd.to_numpy())
        np.testing.assert_allclose(out["signal"].to_numpy(), ind.ema(expected_macd, 9).to_numpy())
        np.testing.assert_allclose(out["hist"].to_numpy(), (out["macd"] - out["signal"]).to_numpy())
        assert out.index.equals(close.index)

    def test_fast_must_be_below_slow(self) -> None:
        with pytest.raises(ValueError):
            ind.macd(_series([1.0, 2.0, 3.0]), fast=5, slow=5)


# --- Bollinger --------------------------------------------------------------------------


class TestBollinger:
    def test_hand_computed_window_3(self) -> None:
        s = _series([1, 2, 3, 4, 5])
        out = ind.bollinger(s, 3, 2.0)
        assert list(out.columns) == ["middle", "upper", "lower", "pct_b", "bandwidth"]
        sd = math.sqrt(2 / 3)  # écart-type (population) de [1, 2, 3]
        np.testing.assert_allclose(
            out["middle"].to_numpy(), [np.nan, np.nan, 2.0, 3.0, 4.0], equal_nan=True
        )
        assert out["upper"].iloc[2] == pytest.approx(2 + 2 * sd)
        assert out["lower"].iloc[2] == pytest.approx(2 - 2 * sd)
        # pct_b = (3 - (2 - 2sd)) / (4sd) = 0.5 + 1 / (4sd)
        assert out["pct_b"].iloc[2] == pytest.approx(0.5 + 1 / (4 * sd))
        # bandwidth = 4sd / 2 = 2sd
        assert out["bandwidth"].iloc[2] == pytest.approx(2 * sd)
        assert out.iloc[:2].isna().all().all()

    def test_middle_equals_sma(self) -> None:
        close = make_ohlcv(100, seed=4)["close"]
        out = ind.bollinger(close, 20)
        np.testing.assert_allclose(
            out["middle"].to_numpy(), ind.sma(close, 20).to_numpy(), equal_nan=True
        )
        np.testing.assert_allclose(
            out["upper"].to_numpy() - out["middle"].to_numpy(),
            out["middle"].to_numpy() - out["lower"].to_numpy(),
            equal_nan=True,
        )

    def test_pct_b_in_unit_interval_when_inside_bands(self) -> None:
        close = make_ohlcv(400, seed=8)["close"]
        out = ind.bollinger(close, 20, 2.0).dropna()
        price = close.loc[out.index]
        inside = (price >= out["lower"]) & (price <= out["upper"])
        assert inside.any()
        assert (out.loc[inside, "pct_b"] >= 0).all()
        assert (out.loc[inside, "pct_b"] <= 1).all()
        above = price > out["upper"]
        below = price < out["lower"]
        assert (out.loc[above, "pct_b"] > 1).all()
        assert (out.loc[below, "pct_b"] < 0).all()
        assert (out["bandwidth"] >= 0).all()

    def test_flat_price_pct_b_is_half(self) -> None:
        out = ind.bollinger(_series([7.0] * 8), 3)
        np.testing.assert_allclose(out["pct_b"].dropna().to_numpy(), 0.5)
        np.testing.assert_allclose(out["bandwidth"].dropna().to_numpy(), 0.0)


# --- Donchian ---------------------------------------------------------------------------


class TestDonchian:
    def test_excludes_current_bar(self) -> None:
        df = _bars([(1, 0, 0.5), (2, 1, 1.5), (3, 2, 2.5), (4, 3, 3.5), (5, 4, 4.5)])
        out = ind.donchian(df, 3)
        assert list(out.columns) == ["upper", "lower", "middle"]
        # upper[3] = max(high[0..2]) = 3 (et non 4 : la bougie courante est exclue)
        np.testing.assert_allclose(
            out["upper"].to_numpy(), [np.nan, np.nan, np.nan, 3.0, 4.0], equal_nan=True
        )
        np.testing.assert_allclose(
            out["lower"].to_numpy(), [np.nan, np.nan, np.nan, 0.0, 1.0], equal_nan=True
        )
        np.testing.assert_allclose(
            out["middle"].to_numpy(), [np.nan, np.nan, np.nan, 1.5, 2.5], equal_nan=True
        )
        # une cassure est détectable : la clôture courante dépasse le canal passé
        assert (df["close"].iloc[3:] > out["upper"].iloc[3:]).all()

    def test_equals_shifted_rolling(self) -> None:
        df = make_ohlcv(120, seed=6)
        out = ind.donchian(df, 20)
        np.testing.assert_allclose(
            out["upper"].to_numpy(),
            df["high"].rolling(20).max().shift(1).to_numpy(),
            equal_nan=True,
        )
        assert out["upper"].iloc[:20].isna().all()
        assert out["upper"].iloc[20:].notna().all()
        valid = out.dropna()
        assert (valid["upper"] >= valid["lower"]).all()

    def test_missing_column(self) -> None:
        with pytest.raises(ValueError, match="high"):
            ind.donchian(pd.DataFrame({"low": [1.0, 2.0]}), 2)


# --- ADX --------------------------------------------------------------------------------


class TestAdx:
    def test_hand_computed_window_2(self) -> None:
        df = _bars([(10, 9, 9.5), (12, 10, 11), (13, 11, 12), (12, 10, 11), (14, 12, 13)])
        out = ind.adx(df, 2)
        assert list(out.columns) == ["adx", "plus_di", "minus_di"]
        # +DM : 2, 1, 0, 2 ; -DM : 0, 0, 1, 0 ; TR : 2.5, 2, 2, 3 (bougies 1..4)
        # lissage Wilder (fenêtre 2) : TR -> 2.25, 2.125, 2.5625 ; +DM -> 1.5, 0.75, 1.375 ;
        # -DM -> 0, 0.5, 0.25
        plus_di = [np.nan, np.nan, 100 * 1.5 / 2.25, 100 * 0.75 / 2.125, 100 * 1.375 / 2.5625]
        minus_di = [np.nan, np.nan, 0.0, 100 * 0.5 / 2.125, 100 * 0.25 / 2.5625]
        # DX : 100, 20, 900/13 ; ADX : amorçage (100 + 20)/2 = 60, puis (60 + 900/13)/2
        adx_vals = [np.nan, np.nan, np.nan, 60.0, (60 + 900 / 13) / 2]
        np.testing.assert_allclose(out["plus_di"].to_numpy(), plus_di, equal_nan=True)
        np.testing.assert_allclose(out["minus_di"].to_numpy(), minus_di, equal_nan=True)
        np.testing.assert_allclose(out["adx"].to_numpy(), adx_vals, equal_nan=True)

    def test_bounds_and_reference_loop(self) -> None:
        df = make_ohlcv(300, seed=13)
        out = ind.adx(df, 14)
        valid = out.dropna()
        assert len(valid) == 300 - (2 * 14 - 1)
        for col in ("adx", "plus_di", "minus_di"):
            assert (valid[col] >= 0).all() and (valid[col] <= 100).all()
        ref_adx, ref_plus, ref_minus = _ref_adx(df, 14)
        np.testing.assert_allclose(out["adx"].to_numpy(), ref_adx, rtol=1e-9, equal_nan=True)
        np.testing.assert_allclose(out["plus_di"].to_numpy(), ref_plus, rtol=1e-9, equal_nan=True)
        np.testing.assert_allclose(out["minus_di"].to_numpy(), ref_minus, rtol=1e-9, equal_nan=True)

    def test_warmup_lengths(self) -> None:
        df = make_ohlcv(60, seed=3)
        out = ind.adx(df, 14)
        assert out["plus_di"].iloc[:14].isna().all() and out["plus_di"].iloc[14:].notna().all()
        assert out["adx"].iloc[:27].isna().all() and out["adx"].iloc[27:].notna().all()

    def test_strong_uptrend(self) -> None:
        df = make_ohlcv(200, seed=7, drift=0.01, vol=0.001)
        out = ind.adx(df, 14).dropna()
        assert (out["plus_di"] > out["minus_di"]).all()
        assert out["adx"].iloc[-1] > 50

    def test_flat_market_is_zero(self) -> None:
        df = _bars([(5, 5, 5)] * 12)
        out = ind.adx(df, 3).dropna()
        assert len(out) > 0
        np.testing.assert_allclose(out.to_numpy(), 0.0)


# --- crossover / crossunder -------------------------------------------------------------


class TestCross:
    def test_exact_bar_positions_vs_scalar(self) -> None:
        a = _series([1, 2, 3, 2, 1, 2.5, 2.0, 1.0])
        over = ind.crossover(a, 2)
        under = ind.crossunder(a, 2)
        assert over.dtype == bool and under.dtype == bool
        assert list(np.flatnonzero(over.to_numpy())) == [2, 5]
        assert list(np.flatnonzero(under.to_numpy())) == [4, 7]
        assert over.index.equals(a.index)

    def test_exact_bar_positions_vs_series(self) -> None:
        a = _series([1, 3, 2, 4, 1])
        b = _series([2, 2, 3, 3, 3])
        assert ind.crossover(a, b).tolist() == [False, True, False, True, False]
        assert ind.crossunder(a, b).tolist() == [False, False, True, False, True]

    def test_touch_then_cross(self) -> None:
        # a == b à t=1 n'est pas un croisement ; le dépassement à t=2 l'est.
        a = _series([1, 2, 3])
        assert ind.crossover(a, 2).tolist() == [False, False, True]
        b = _series([3, 2, 1])
        assert ind.crossunder(b, 2).tolist() == [False, False, True]

    def test_nan_is_false(self) -> None:
        a = _series([np.nan, 3, np.nan, 1, 3])
        assert ind.crossover(a, 2).tolist() == [False, False, False, False, True]
        assert ind.crossunder(a, 2).tolist() == [False, False, False, False, False]
        # t=2 : a > b mais b[t-1] est NaN -> pas de croisement ; t=4 : vrai croisement
        b = _series([np.nan, np.nan, 4, 4, 4])
        expected = [False, False, False, False, True]
        assert ind.crossover(_series([1, 5, 5, 3, 5]), b).tolist() == expected

    def test_on_moving_averages(self) -> None:
        close = make_ohlcv(300, seed=21)["close"]
        fast, slow = ind.sma(close, 5), ind.sma(close, 20)
        over, under = ind.crossover(fast, slow), ind.crossunder(fast, slow)
        assert over.any() and under.any()
        assert not (over & under).any()
        # vérification scalaire bougie par bougie
        f, s = fast.to_numpy(), slow.to_numpy()
        for t in range(1, len(close)):
            exp_over = bool(f[t] > s[t] and f[t - 1] <= s[t - 1])
            exp_under = bool(f[t] < s[t] and f[t - 1] >= s[t - 1])
            assert over.iloc[t] == exp_over
            assert under.iloc[t] == exp_under
        assert not over.iloc[0] and not under.iloc[0]


# --- causalité (aucun look-ahead) ---------------------------------------------------


Indicator = Callable[[pd.DataFrame], pd.Series | pd.DataFrame]

CAUSAL_CASES: dict[str, Indicator] = {
    "sma": lambda df: ind.sma(df["close"], 10),
    "ema": lambda df: ind.ema(df["close"], 10),
    "rsi": lambda df: ind.rsi(df["close"], 14),
    "atr": lambda df: ind.atr(df, 14),
    "true_range": lambda df: ind.true_range(df),
    "wilder_smooth": lambda df: ind.wilder_smooth(df["close"], 7),
    "macd": lambda df: ind.macd(df["close"]),
    "bollinger": lambda df: ind.bollinger(df["close"], 20),
    "donchian": lambda df: ind.donchian(df, 20),
    "adx": lambda df: ind.adx(df, 14),
    "rolling_volatility": lambda df: ind.rolling_volatility(df["close"], 20, 8760),
    "returns": lambda df: ind.returns(df["close"]),
    "log_returns": lambda df: ind.returns(df["close"], log=True),
    "crossover": lambda df: ind.crossover(ind.sma(df["close"], 5), ind.sma(df["close"], 20)),
    "crossunder": lambda df: ind.crossunder(ind.sma(df["close"], 5), ind.sma(df["close"], 20)),
}


@pytest.mark.parametrize("name", sorted(CAUSAL_CASES))
@pytest.mark.parametrize("k", [3, 17, 50, 123, 299])
def test_no_lookahead(name: str, k: int) -> None:
    """Calculer sur df.iloc[:k] doit donner exactement les k premières valeurs du calcul
    sur la série complète : la valeur en t ne dépend que des données <= t."""
    df = make_ohlcv(300, seed=99)
    func = CAUSAL_CASES[name]
    full = func(df)
    partial = func(df.iloc[:k])
    assert len(partial) == k
    assert partial.index.equals(df.index[:k])
    full_values = full.iloc[:k].to_numpy(dtype="float64")
    partial_values = partial.to_numpy(dtype="float64")
    np.testing.assert_allclose(partial_values, full_values, rtol=1e-12, atol=0, equal_nan=True)


@pytest.mark.parametrize("name", sorted(CAUSAL_CASES))
def test_future_change_does_not_alter_past(name: str) -> None:
    """Modifier les bougies après t ne change aucune valeur <= t."""
    df = make_ohlcv(200, seed=5)
    altered = df.copy()
    t = 150
    altered.loc[altered.index[t + 1] :, ["open", "high", "low", "close"]] *= 3.0
    func = CAUSAL_CASES[name]
    a = func(df).iloc[: t + 1].to_numpy(dtype="float64")
    b = func(altered).iloc[: t + 1].to_numpy(dtype="float64")
    np.testing.assert_allclose(a, b, rtol=0, atol=0, equal_nan=True)


def test_inputs_are_not_mutated() -> None:
    df = make_ohlcv(80, seed=1)
    before = df.copy()
    for func in CAUSAL_CASES.values():
        func(df)
    pd.testing.assert_frame_equal(df, before)
