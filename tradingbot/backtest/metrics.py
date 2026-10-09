"""Métriques de performance d'une courbe d'équité et d'une liste de trades.

Toutes les fonctions sont tolérantes : série vide, équité constante ou DataFrame de
trades vide renvoient des valeurs neutres (``0.0``) au lieu de lever une exception.
:func:`compute_metrics` renvoie uniquement des ``float`` / ``int`` Python (sérialisable
en JSON).

Conventions :

- ``equity`` : ``pd.Series`` de la valeur du portefeuille à chaque bougie.
- ``returns`` : rendements par bougie (``equity.pct_change().fillna(0)``).
- ``periods_per_year`` : nombre de bougies par an (8760 pour du 1h en crypto, 365 en 1d).
- ``risk_free`` : taux sans risque **annuel** (fraction), converti par bougie en
  ``risk_free / periods_per_year``.
- ``trades`` : ``pd.DataFrame`` avec au moins ``pnl`` (gain net en devise) ; ``pnl_pct``
  et ``bars_held`` sont utilisés s'ils existent.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "annual_volatility",
    "cagr",
    "calmar_ratio",
    "compute_metrics",
    "drawdown_series",
    "exposure",
    "max_drawdown",
    "max_drawdown_duration",
    "profit_factor",
    "sharpe_ratio",
    "sortino_ratio",
    "win_rate",
]

METRIC_KEYS: tuple[str, ...] = (
    "total_return",
    "cagr",
    "annual_volatility",
    "sharpe",
    "sortino",
    "max_drawdown",
    "max_drawdown_duration",
    "calmar",
    "n_trades",
    "win_rate",
    "profit_factor",
    "avg_trade_pct",
    "avg_win_pct",
    "avg_loss_pct",
    "best_trade_pct",
    "worst_trade_pct",
    "avg_bars_held",
    "exposure",
    "buy_hold_return",
    "final_equity",
)


# --------------------------------------------------------------------------- utilitaires
def _as_float_array(values: pd.Series | np.ndarray | list[float] | None) -> np.ndarray:
    """Tableau float64 1D, sans NaN ni infinis."""
    if values is None:
        return np.empty(0, dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def _safe(x: float) -> float:
    """float Python ; 0.0 si NaN."""
    x = float(x)
    return 0.0 if math.isnan(x) else x


def _trade_column(trades: pd.DataFrame | None, column: str) -> np.ndarray:
    """Colonne d'un DataFrame de trades sous forme de tableau float (vide si absente)."""
    if trades is None or len(trades) == 0 or column not in trades.columns:
        return np.empty(0, dtype=np.float64)
    return _as_float_array(trades[column])


def _trade_pnl_pairs(trades: pd.DataFrame | None) -> tuple[np.ndarray, np.ndarray]:
    """(pnl, pnl_pct) alignés, restreints aux trades où les deux valeurs sont finies.

    Si ``pnl_pct`` est absent, ``pnl`` sert de substitut pour le signe uniquement.
    """
    if trades is None or len(trades) == 0 or "pnl" not in trades.columns:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty
    pnl = np.asarray(trades["pnl"], dtype=np.float64).ravel()
    if "pnl_pct" in trades.columns:
        pct = np.asarray(trades["pnl_pct"], dtype=np.float64).ravel()
    else:
        pct = np.full(pnl.shape, np.nan)
    mask = np.isfinite(pnl) & np.isfinite(pct)
    return pnl[mask], pct[mask]


def _returns_from_equity(equity: pd.Series) -> np.ndarray:
    """Rendements par bougie (``pct_change().fillna(0)``), infinis remplacés par 0."""
    eq = np.asarray(equity, dtype=np.float64).ravel()
    if eq.size < 2:
        return np.zeros(eq.size, dtype=np.float64)
    prev = eq[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.where(prev != 0, eq[1:] / prev - 1.0, 0.0)
    rets = np.where(np.isfinite(rets), rets, 0.0)
    return np.concatenate([[0.0], rets])


# --------------------------------------------------------------------------- drawdown
def drawdown_series(equity: pd.Series) -> pd.Series:
    """Drawdown à chaque bougie : ``equity / plus_haut_courant - 1`` (≤ 0)."""
    if len(equity) == 0:
        return pd.Series(dtype=np.float64, index=equity.index, name="drawdown")
    eq = pd.Series(np.asarray(equity, dtype=np.float64), index=equity.index)
    peak = eq.cummax()
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = eq / peak - 1.0
    dd = dd.where(peak > 0, 0.0).fillna(0.0)
    dd.name = "drawdown"
    return dd


def max_drawdown(equity: pd.Series) -> float:
    """Pire drawdown, en fraction **négative** (ex. ``-0.23``) ; ``0.0`` si aucun."""
    if len(equity) == 0:
        return 0.0
    return _safe(min(float(drawdown_series(equity).min()), 0.0))


def max_drawdown_duration(equity: pd.Series) -> int:
    """Plus longue période (en bougies) passée sous le plus haut précédent."""
    if len(equity) == 0:
        return 0
    below = (drawdown_series(equity).to_numpy() < 0).astype(np.int64)
    longest = current = 0
    for flag in below:
        current = current + 1 if flag else 0
        if current > longest:
            longest = current
    return int(longest)


# --------------------------------------------------------------------------- ratios
def annual_volatility(returns: pd.Series | np.ndarray, periods_per_year: float) -> float:
    """Écart-type des rendements (ddof=1) annualisé ; ``0.0`` si moins de 2 points."""
    r = _as_float_array(returns)
    if r.size < 2 or periods_per_year <= 0:
        return 0.0
    return _safe(float(np.std(r, ddof=1)) * math.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series | np.ndarray, periods_per_year: float, risk_free: float = 0.0
) -> float:
    """Ratio de Sharpe annualisé : ``mean(excès) / std(ddof=1) * sqrt(periods_per_year)``.

    ``0.0`` si la volatilité est nulle ou s'il y a moins de 2 points.
    """
    r = _as_float_array(returns)
    if r.size < 2 or periods_per_year <= 0:
        return 0.0
    excess = r - risk_free / periods_per_year
    std = float(np.std(excess, ddof=1))
    if std <= 0 or not math.isfinite(std):
        return 0.0
    return _safe(float(np.mean(excess)) / std * math.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series | np.ndarray, periods_per_year: float, risk_free: float = 0.0
) -> float:
    """Ratio de Sortino annualisé : ``mean(excès) / déviation_baissière * sqrt(ppy)``.

    La déviation baissière est ``sqrt(mean(min(excès, 0)²))``. ``0.0`` si elle est nulle
    (aucune bougie perdante) ou s'il y a moins de 2 points.
    """
    r = _as_float_array(returns)
    if r.size < 2 or periods_per_year <= 0:
        return 0.0
    excess = r - risk_free / periods_per_year
    downside = float(np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2)))
    if downside <= 0 or not math.isfinite(downside):
        return 0.0
    return _safe(float(np.mean(excess)) / downside * math.sqrt(periods_per_year))


def cagr(equity: pd.Series | np.ndarray, periods_per_year: float) -> float:
    """Taux de croissance annualisé entre la première et la dernière équité.

    Le nombre d'années vaut ``(len(equity) - 1) / periods_per_year``. ``0.0`` si moins de
    2 points ou équité initiale nulle ; ``-1.0`` si l'équité finale est ≤ 0.
    """
    eq = np.asarray(equity, dtype=np.float64).ravel()
    n_periods = eq.size - 1
    if n_periods <= 0 or periods_per_year <= 0:
        return 0.0
    first, last = float(eq[0]), float(eq[-1])
    if not math.isfinite(first) or not math.isfinite(last) or first <= 0:
        return 0.0
    if last <= 0:
        return -1.0
    years = n_periods / periods_per_year
    return _safe((last / first) ** (1.0 / years) - 1.0)


def calmar_ratio(cagr_value: float, max_dd: float) -> float:
    """CAGR divisé par le drawdown maximal (en valeur absolue) ; ``0.0`` sans drawdown."""
    if max_dd >= 0 or not math.isfinite(max_dd) or not math.isfinite(cagr_value):
        return 0.0
    return _safe(cagr_value / abs(max_dd))


# --------------------------------------------------------------------------- trades
def profit_factor(trades: pd.DataFrame | None) -> float:
    """Somme des gains / somme des pertes (valeurs absolues).

    ``0.0`` sans trade ; ``inf`` s'il y a des gains et aucune perte.
    """
    pnl = _trade_column(trades, "pnl")
    if pnl.size == 0:
        return 0.0
    wins = float(pnl[pnl > 0].sum())
    losses = float(-pnl[pnl < 0].sum())
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return wins / losses


def win_rate(trades: pd.DataFrame | None) -> float:
    """Fraction des trades clôturés avec un gain net strictement positif."""
    pnl = _trade_column(trades, "pnl")
    if pnl.size == 0:
        return 0.0
    return float(np.count_nonzero(pnl > 0)) / float(pnl.size)


def exposure(positions: pd.Series | np.ndarray | None) -> float:
    """Fraction des bougies pendant lesquelles une position (≠ 0) est tenue."""
    if positions is None or len(positions) == 0:
        return 0.0
    pos = np.asarray(positions, dtype=np.float64).ravel()
    return float(np.count_nonzero(pos != 0)) / float(pos.size)


# --------------------------------------------------------------------------- synthèse
def compute_metrics(
    equity: pd.Series,
    trades: pd.DataFrame | None,
    periods_per_year: float,
    risk_free: float = 0.0,
    benchmark: pd.Series | None = None,
    positions: pd.Series | np.ndarray | None = None,
) -> dict[str, float | int]:
    """Calcule toutes les métriques d'un backtest.

    :param equity: valeur du portefeuille à chaque bougie.
    :param trades: une ligne par trade (colonnes ``pnl``, ``pnl_pct``, ``bars_held``…) ;
        peut être vide ou ``None``.
    :param periods_per_year: bougies par an (8760 en 1h, 365 en 1d).
    :param risk_free: taux sans risque annuel (fraction).
    :param benchmark: série de prix (ex. ``close``) ; ajoute ``buy_hold_return``.
    :param positions: position tenue à chaque bougie (-1/0/1) ; si fournie, ``exposure``
        est la fraction de bougies où elle est non nulle. Sinon, ``exposure`` est estimée
        à partir de ``bars_held`` des trades.
    :returns: dict de ``float``/``int`` Python, clés ``METRIC_KEYS`` (``buy_hold_return``
        seulement si ``benchmark`` est fourni). Ne lève jamais pour des entrées vides ou
        une équité constante.
    """
    eq = np.asarray(equity, dtype=np.float64).ravel() if equity is not None else np.empty(0)
    n_bars = int(eq.size)
    rets = _returns_from_equity(eq)

    first = float(eq[0]) if n_bars else 0.0
    last = float(eq[-1]) if n_bars else 0.0
    total_return = _safe(last / first - 1.0) if n_bars and first > 0 else 0.0

    equity_series = equity if isinstance(equity, pd.Series) else pd.Series(eq)
    mdd = max_drawdown(equity_series)
    growth = cagr(eq, periods_per_year)

    n_trades = int(_trade_column(trades, "pnl").size)
    pnl, pnl_pct = _trade_pnl_pairs(trades)
    bars_held = _trade_column(trades, "bars_held")

    if pnl_pct.size:
        wins_pct = pnl_pct[pnl > 0]
        losses_pct = pnl_pct[pnl < 0]
        avg_trade_pct = _safe(float(pnl_pct.mean()))
        avg_win_pct = _safe(float(wins_pct.mean())) if wins_pct.size else 0.0
        avg_loss_pct = _safe(float(losses_pct.mean())) if losses_pct.size else 0.0
        best_trade_pct = _safe(float(pnl_pct.max()))
        worst_trade_pct = _safe(float(pnl_pct.min()))
    else:
        avg_trade_pct = avg_win_pct = avg_loss_pct = best_trade_pct = worst_trade_pct = 0.0

    if positions is not None:
        expo = exposure(positions)
    elif bars_held.size and n_bars:
        expo = min(1.0, float(bars_held.sum()) / n_bars)
    else:
        expo = 0.0

    metrics: dict[str, float | int] = {
        "total_return": total_return,
        "cagr": growth,
        "annual_volatility": annual_volatility(rets, periods_per_year),
        "sharpe": sharpe_ratio(rets, periods_per_year, risk_free),
        "sortino": sortino_ratio(rets, periods_per_year, risk_free),
        "max_drawdown": mdd,
        "max_drawdown_duration": max_drawdown_duration(equity_series),
        "calmar": calmar_ratio(growth, mdd),
        "n_trades": n_trades,
        "win_rate": win_rate(trades),
        "profit_factor": profit_factor(trades),
        "avg_trade_pct": avg_trade_pct,
        "avg_win_pct": avg_win_pct,
        "avg_loss_pct": avg_loss_pct,
        "best_trade_pct": best_trade_pct,
        "worst_trade_pct": worst_trade_pct,
        "avg_bars_held": _safe(float(bars_held.mean())) if bars_held.size else 0.0,
        "exposure": expo,
    }
    if benchmark is not None:
        bench = _as_float_array(benchmark)
        metrics["buy_hold_return"] = (
            _safe(float(bench[-1] / bench[0] - 1.0)) if bench.size and bench[0] > 0 else 0.0
        )
    metrics["final_equity"] = _safe(last)
    return metrics
