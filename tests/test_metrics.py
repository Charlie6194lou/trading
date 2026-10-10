"""Tests des métriques : valeurs calculées à la main sur de toutes petites séries."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from tradingbot.backtest.metrics import (
    METRIC_KEYS,
    annual_volatility,
    annualize_return,
    cagr,
    calmar_ratio,
    compute_metrics,
    drawdown_series,
    exposure,
    max_drawdown,
    max_drawdown_duration,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
    years_elapsed,
)


def series(values: list[float], freq: str | pd.Timedelta = "1D") -> pd.Series:
    index = pd.date_range("2024-01-01", periods=len(values), freq=freq, tz="UTC")
    return pd.Series(values, index=index, dtype=float)


#: 4 intervalles de 91 j 6 h = 365 jours exactement : 4 bougies par an « calendaires »
QUARTER = pd.Timedelta(days=91, hours=6)


def trades(
    pnl: list[float], pnl_pct: list[float] | None = None, bars: list[int] | None = None
) -> pd.DataFrame:
    n = len(pnl)
    return pd.DataFrame(
        {
            "pnl": pnl,
            "pnl_pct": pnl_pct if pnl_pct is not None else [p / 1000 for p in pnl],
            "bars_held": bars if bars is not None else [1] * n,
        }
    )


# ============================================================================ sharpe / sortino
def test_sharpe_hand_computed() -> None:
    # Rendements 1 %, 2 %, 3 % : moyenne 2 %, écart-type (ddof=1) 1 % → 2 × sqrt(ppy).
    assert sharpe_ratio([0.01, 0.02, 0.03], periods_per_year=4) == pytest.approx(4.0)
    assert sharpe_ratio([0.01, 0.02, 0.03], periods_per_year=252) == pytest.approx(
        2 * math.sqrt(252)
    )
    # Avec un taux sans risque annuel de 4 % sur 4 périodes : excès = r - 1 % → moyenne 1 %.
    assert sharpe_ratio([0.01, 0.02, 0.03], periods_per_year=4, risk_free=0.04) == pytest.approx(
        2.0
    )
    # Série symétrique : moyenne nulle → 0.
    assert sharpe_ratio([0.1, -0.1, 0.1, -0.1], 252) == 0.0


def test_sharpe_zero_when_no_vol_or_too_short() -> None:
    assert sharpe_ratio([0.01, 0.01, 0.01], 252) == 0.0
    assert sharpe_ratio([0.0, 0.0], 252) == 0.0
    assert sharpe_ratio([0.05], 252) == 0.0
    assert sharpe_ratio([], 252) == 0.0
    assert sharpe_ratio(pd.Series([0.01, np.nan, 0.03]), 4) == pytest.approx(
        sharpe_ratio([0.01, 0.03], 4)
    )


def test_sortino_hand_computed() -> None:
    # Rendements 1 %, -2 %, 3 %, 0 : moyenne 0,5 % ; déviation baissière
    # sqrt(mean([0, 0.0004, 0, 0])) = sqrt(0.0001) = 1 % → 0,5 × sqrt(ppy).
    assert sortino_ratio([0.01, -0.02, 0.03, 0.0], periods_per_year=4) == pytest.approx(1.0)
    assert sortino_ratio([0.01, -0.02, 0.03, 0.0], periods_per_year=16) == pytest.approx(2.0)
    # Aucune bougie perdante → déviation nulle → 0 (convention).
    assert sortino_ratio([0.01, 0.02], 4) == 0.0
    assert sortino_ratio([0.5], 4) == 0.0


def test_annual_volatility_hand_computed() -> None:
    assert annual_volatility([0.01, 0.02, 0.03], periods_per_year=4) == pytest.approx(0.02)
    assert annual_volatility([0.01], 4) == 0.0
    assert annual_volatility([0.01, 0.01], 4) == 0.0


# ============================================================================ drawdown
def test_drawdown_series_and_max_drawdown_hand_computed() -> None:
    eq = series([100, 120, 90, 110, 130, 100])
    dd = drawdown_series(eq)
    np.testing.assert_allclose(dd.to_numpy(), [0, 0, -0.25, -1 / 12, 0, -3 / 13])
    assert dd.index.equals(eq.index)
    assert max_drawdown(eq) == pytest.approx(-0.25)
    assert max_drawdown_duration(eq) == 2  # bougies 2 et 3 sous le plus haut 120


def test_max_drawdown_duration_longest_stretch() -> None:
    # Sous le plus haut (10) de l'indice 1 à 5 inclus (5 bougies), retour au plus haut à 6.
    eq = series([10, 9, 8, 9, 9.5, 9.9, 10, 10.5, 10])
    assert max_drawdown_duration(eq) == 5
    assert max_drawdown(eq) == pytest.approx(-0.2)
    # Toujours au plus haut : 0.
    assert max_drawdown_duration(series([1, 2, 3])) == 0
    # Jamais de nouveau plus haut : toute la série sauf la première bougie.
    assert max_drawdown_duration(series([3, 2, 1, 1])) == 3


def test_drawdown_constant_or_empty() -> None:
    assert max_drawdown(series([5, 5, 5])) == 0.0
    assert max_drawdown_duration(series([5, 5, 5])) == 0
    assert max_drawdown(pd.Series(dtype=float)) == 0.0
    assert max_drawdown_duration(pd.Series(dtype=float)) == 0
    assert len(drawdown_series(pd.Series(dtype=float))) == 0


# ============================================================================ cagr / calmar
def test_cagr_hand_computed_from_bar_count() -> None:
    """Sans index daté, le nombre d'années vaut ``(n - 1) / periods_per_year``."""
    # 100 → 121 en 2 périodes à 1 période / an : (1,21)^(1/2) - 1 = 10 %.
    assert cagr(np.array([100.0, 110.0, 121.0]), periods_per_year=1) == pytest.approx(0.10)
    # 100 → 121 en 2 périodes à 2 périodes / an : 1 an → 21 %.
    assert cagr(np.array([100.0, 110.0, 121.0]), periods_per_year=2) == pytest.approx(0.21)
    # Series sans DatetimeIndex : même repli.
    assert cagr(pd.Series([100.0, 90.0, 81.0]), periods_per_year=1) == pytest.approx(-0.10)
    # 100 → 200 en 365 bougies journalières → 100 %.
    eq = np.array([100.0] + [150.0] * 364 + [200.0])
    assert cagr(eq, periods_per_year=365) == pytest.approx(1.0)


def test_cagr_hand_computed_from_elapsed_time() -> None:
    """Avec un DatetimeIndex, le nombre d'années est le temps écoulé / 365 jours (F58)."""
    # 366 clôtures journalières = 365 jours écoulés = 1 an : 100 → 200 → +100 %.
    eq = series([100.0] + [150.0] * 364 + [200.0])
    assert cagr(eq, periods_per_year=365) == pytest.approx(1.0)
    # periods_per_year n'intervient plus quand l'index est daté.
    assert cagr(eq, periods_per_year=252) == pytest.approx(1.0)
    # 5 points espacés d'un trimestre : 100 → 121 en 1 an → 21 %.
    assert cagr(series([100, 105, 110, 115, 121], freq=QUARTER), 4) == pytest.approx(0.21)
    # 731 clôtures = 730 jours écoulés = 2 ans : 100 → 81 → -10 % par an.
    eq = series([100.0] + [90.0] * 729 + [81.0])
    assert cagr(eq, periods_per_year=365) == pytest.approx(-0.10)


def test_cagr_with_gaps_uses_elapsed_time_not_bar_count() -> None:
    """F58 : 50 bougies horaires, deux mois de trou, 50 bougies : 100 → 110."""
    idx = pd.DatetimeIndex(
        list(pd.date_range("2024-01-01", periods=50, freq="1h", tz="UTC"))
        + list(pd.date_range("2024-03-01", periods=50, freq="1h", tz="UTC"))
    )
    eq = pd.Series(np.linspace(100.0, 110.0, 100), index=idx)
    years = (idx[-1] - idx[0]) / pd.Timedelta(days=365)
    assert years_elapsed(eq, 8760) == pytest.approx(years)
    assert cagr(eq, 8760) == pytest.approx(1.1 ** (1 / years) - 1)  # ≈ +75 %
    # Sur le nombre de bougies (99 h), le même gain aurait donné un chiffre absurde.
    assert cagr(eq.to_numpy(), 8760) > 1000


def test_years_elapsed_fallbacks() -> None:
    assert years_elapsed(series([1.0, 2.0, 3.0]), 365) == pytest.approx(2 / 365)
    assert years_elapsed(np.array([1.0, 2.0, 3.0]), 4) == pytest.approx(0.5)
    assert years_elapsed(np.array([1.0]), 4) == 0.0
    assert years_elapsed(np.array([1.0, 2.0]), 0) == 0.0
    # Index daté mais horodatages identiques : repli sur le nombre de bougies.
    same = pd.Series([1.0, 2.0], index=pd.DatetimeIndex(["2024-01-01", "2024-01-01"], tz="UTC"))
    assert years_elapsed(same, 2) == pytest.approx(0.5)


def test_cagr_degenerate_cases() -> None:
    assert cagr(series([100]), 365) == 0.0
    assert cagr(series([]), 365) == 0.0
    assert cagr(series([0, 100]), 365) == 0.0
    assert cagr(series([100, 0]), 365) == -1.0
    assert cagr(series([100, 100, 100]), 365) == 0.0


def test_cagr_never_raises_on_huge_growth() -> None:
    """F40 / F53 : ``float ** float`` levait OverflowError ; on renvoie ``inf``."""
    assert cagr(pd.Series([100.0, 130.0]), 525_600) == math.inf
    assert cagr(np.array([100.0, 200.0]), 525_600) == math.inf
    assert cagr(np.array([100.0, 300.0]), 8_760) == math.inf
    eq = series([100.0, 130.0], freq="1min")
    assert cagr(eq, 525_600) == math.inf
    # +10 % sur 3 bougies horaires (2 intervalles) : 1.1 ** 4380 ≈ 2e181, représentable.
    assert cagr(np.array([100.0, 105.0, 110.0]), 8_760) == pytest.approx(1.1**4380 - 1)
    # Sur un seul intervalle, 1.1 ** 8760 dépasse la plage des float → inf, pas d'exception.
    assert cagr(np.array([100.0, 110.0]), 8_760) == math.inf
    # Le Calmar d'un CAGR infini vaut 0 (convention « valeur neutre »).
    assert calmar_ratio(math.inf, -0.1) == 0.0


def test_annualize_return_hand_computed() -> None:
    assert annualize_return(0.21, 1.0) == pytest.approx(0.21)
    assert annualize_return(0.21, 2.0) == pytest.approx(0.10)
    assert annualize_return(1.0, 0.5) == pytest.approx(3.0)  # ×2 en 6 mois = ×4 par an
    assert annualize_return(-0.19, 2.0) == pytest.approx(-0.10)
    assert annualize_return(0.3, 2 / 525_600) == math.inf
    assert annualize_return(-1.0, 1.0) == -1.0
    assert annualize_return(-1.5, 1.0) == -1.0
    assert annualize_return(0.5, 0.0) == 0.0
    assert annualize_return(math.nan, 1.0) == 0.0


def test_compute_metrics_never_raises_on_short_high_gain_series() -> None:
    """F40 : run_backtest / walk_forward plantaient via compute_metrics."""
    eq = series([100.0, 130.0], freq="1min")
    m = compute_metrics(eq, None, periods_per_year=525_600, benchmark=eq)
    assert m["cagr"] == math.inf
    assert m["calmar"] == 0.0
    assert m["total_return"] == pytest.approx(0.30)
    assert m["buy_hold_return"] == pytest.approx(0.30)
    json.dumps(m)  # inf reste sérialisable (le rapport l'écrit null)


def test_calmar_ratio() -> None:
    assert calmar_ratio(0.30, -0.15) == pytest.approx(2.0)
    assert calmar_ratio(0.30, 0.0) == 0.0
    assert calmar_ratio(-0.10, -0.20) == pytest.approx(-0.5)


# ============================================================================ trades
def test_profit_factor_and_win_rate_hand_computed() -> None:
    tr = trades([10, -5, 20, -5])
    assert profit_factor(tr) == pytest.approx(3.0)  # 30 / 10
    assert win_rate(tr) == pytest.approx(0.5)
    assert profit_factor(trades([10, 5])) == float("inf")
    assert win_rate(trades([10, 5])) == 1.0
    assert profit_factor(trades([-10, -5])) == 0.0
    assert win_rate(trades([-10, -5])) == 0.0
    assert profit_factor(trades([0.0, 0.0])) == 0.0  # ni gain ni perte
    assert win_rate(trades([0.0, 10.0])) == 0.5  # un trade nul n'est pas gagnant


def test_profit_factor_and_win_rate_empty() -> None:
    assert profit_factor(pd.DataFrame()) == 0.0
    assert profit_factor(None) == 0.0
    assert win_rate(pd.DataFrame()) == 0.0
    assert win_rate(pd.DataFrame({"pnl": []})) == 0.0


def test_exposure() -> None:
    assert exposure(pd.Series([0, 1, 1, 0])) == pytest.approx(0.5)
    assert exposure(np.array([-1, 0, 0, 0, 1])) == pytest.approx(0.4)
    assert exposure([0, 0]) == 0.0
    assert exposure([]) == 0.0
    assert exposure(None) == 0.0


# ============================================================================ compute_metrics
def test_compute_metrics_hand_computed() -> None:
    # Équité : 100 → 110 → 99 → 108,9 → 120. Rendements (avec le 0 initial) :
    # [0, 0.1, -0.1, 0.1, 0.1019...]. Plus haut 110 à l'indice 1, creux 99 → -10 %.
    # Index trimestriel : 4 intervalles = 365 jours = 1 an (cohérent avec ppy = 4).
    eq = series([100, 110, 99, 108.9, 120], freq=QUARTER)
    tr = trades(pnl=[10, -11, 9.9, 11.1], pnl_pct=[0.1, -0.1, 0.1, 0.1], bars=[1, 1, 1, 1])
    bench = series([50, 55, 60, 58, 75], freq=QUARTER)
    m = compute_metrics(eq, tr, periods_per_year=4, benchmark=bench)

    assert set(m) == set(METRIC_KEYS)
    assert m["total_return"] == pytest.approx(0.20)
    assert m["final_equity"] == 120.0
    assert m["max_drawdown"] == pytest.approx(-0.10)
    assert m["max_drawdown_duration"] == 2  # indices 2 et 3 sous 110
    assert m["cagr"] == pytest.approx(1.2 ** (4 / 4) - 1)  # 4 périodes = 1 an
    rets = np.array([0.0, 0.1, -0.1, 0.1, 120 / 108.9 - 1])
    assert m["sharpe"] == pytest.approx(rets.mean() / rets.std(ddof=1) * 2)
    assert m["annual_volatility"] == pytest.approx(rets.std(ddof=1) * 2)
    downside = math.sqrt(np.mean(np.minimum(rets, 0) ** 2))
    assert m["sortino"] == pytest.approx(rets.mean() / downside * 2)
    assert m["calmar"] == pytest.approx(m["cagr"] / 0.10)
    assert m["n_trades"] == 4
    assert m["win_rate"] == pytest.approx(0.75)
    assert m["profit_factor"] == pytest.approx(31.0 / 11.0)
    assert m["avg_trade_pct"] == pytest.approx(0.05)
    assert m["avg_win_pct"] == pytest.approx(0.1)
    assert m["avg_loss_pct"] == pytest.approx(-0.1)
    assert m["best_trade_pct"] == pytest.approx(0.1)
    assert m["worst_trade_pct"] == pytest.approx(-0.1)
    assert m["avg_bars_held"] == 1.0
    assert m["exposure"] == pytest.approx(4 / 5)  # 4 bougies tenues sur 5 (via bars_held)
    assert m["buy_hold_return"] == pytest.approx(0.5)


def test_compute_metrics_positions_argument_overrides_exposure() -> None:
    eq = series([100, 100, 100, 100])
    tr = trades([1.0], bars=[4])
    assert compute_metrics(eq, tr, 365)["exposure"] == 1.0
    m = compute_metrics(eq, tr, 365, positions=pd.Series([0, 1, 0, 0]))
    assert m["exposure"] == pytest.approx(0.25)


def test_compute_metrics_empty_trades_and_constant_equity() -> None:
    eq = series([100, 100, 100, 100])
    for empty in (pd.DataFrame(), None, pd.DataFrame(columns=["pnl", "pnl_pct", "bars_held"])):
        m = compute_metrics(eq, empty, periods_per_year=365)
        assert m["n_trades"] == 0
        assert m["total_return"] == 0.0
        assert m["cagr"] == 0.0
        assert m["sharpe"] == 0.0
        assert m["sortino"] == 0.0
        assert m["annual_volatility"] == 0.0
        assert m["max_drawdown"] == 0.0
        assert m["max_drawdown_duration"] == 0
        assert m["calmar"] == 0.0
        assert m["win_rate"] == 0.0
        assert m["profit_factor"] == 0.0
        assert m["avg_trade_pct"] == 0.0
        assert m["exposure"] == 0.0
        assert m["final_equity"] == 100.0
        assert "buy_hold_return" not in m


def test_compute_metrics_empty_equity() -> None:
    m = compute_metrics(
        pd.Series(dtype=float), pd.DataFrame(), 365, benchmark=pd.Series(dtype=float)
    )
    assert m["final_equity"] == 0.0
    assert m["total_return"] == 0.0
    assert m["buy_hold_return"] == 0.0
    assert m["n_trades"] == 0


def test_compute_metrics_returns_plain_python_types() -> None:
    eq = series([100, 101, 99, 103])
    tr = trades([1.0, -2.0, 4.0])
    m = compute_metrics(eq, tr, 365, benchmark=eq)
    for key, value in m.items():
        assert type(value) in (float, int), f"{key} : {type(value)}"
    json.dumps(m)
    assert type(m["n_trades"]) is int
    assert type(m["max_drawdown_duration"]) is int


def test_compute_metrics_profit_factor_inf_is_serialisable() -> None:
    m = compute_metrics(series([100, 110]), trades([5.0, 5.0]), 365)
    assert m["profit_factor"] == float("inf")
    assert "Infinity" in json.dumps(m)


def test_compute_metrics_tolerates_nan_in_trades() -> None:
    tr = pd.DataFrame(
        {"pnl": [1.0, np.nan, -1.0], "pnl_pct": [0.01, 0.02, np.nan], "bars_held": [1, 2, 3]}
    )
    m = compute_metrics(series([100, 101, 100]), tr, 365)
    assert m["n_trades"] == 2
    assert m["win_rate"] == pytest.approx(0.5)
    assert m["avg_trade_pct"] == pytest.approx(0.01)  # seul trade avec pnl et pnl_pct finis
    assert m["avg_bars_held"] == pytest.approx(2.0)


def test_compute_metrics_accepts_numpy_equity() -> None:
    m = compute_metrics(np.array([100.0, 120.0, 90.0]), None, 365)  # type: ignore[arg-type]
    assert m["max_drawdown"] == pytest.approx(-0.25)
    assert m["total_return"] == pytest.approx(-0.10)
