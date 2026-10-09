"""Rapports de backtest en français : texte aligné, comparaison, fichiers et graphique.

- :func:`format_report` : rapport complet d'un :class:`~tradingbot.backtest.engine.BacktestResult`
  (en-tête, métriques, comparaison avec le buy & hold, pires/meilleurs trades, paragraphe
  de lecture pour débutant) ;
- :func:`format_comparison` : tableau une ligne par stratégie, trié par Sharpe, plus la
  ligne buy & hold ;
- :func:`save_report` : écrit ``equity.csv``, ``trades.csv``, ``summary.txt``,
  ``metrics.json`` (et ``equity.png`` si matplotlib est installé) ;
- :func:`plot_result` : courbe d'équité + drawdown + marqueurs de trades ; matplotlib est
  importé paresseusement et son absence n'est jamais une erreur (``None``).
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import math
import textwrap
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from tradingbot.backtest import metrics as metrics_mod

if TYPE_CHECKING:
    from tradingbot.backtest.engine import BacktestResult

logger = logging.getLogger(__name__)

__all__ = [
    "format_comparison",
    "format_report",
    "plot_result",
    "save_report",
]

#: nombre de trades affichés dans les tableaux « pires » et « meilleurs »
N_EXTREME_TRADES = 5
#: largeur des paragraphes de lecture
TEXT_WIDTH = 96
#: fichiers toujours produits par :func:`save_report`
REPORT_FILES: tuple[str, ...] = ("equity.csv", "trades.csv", "summary.txt", "metrics.json")

EXIT_REASON_LABELS: dict[str, str] = {
    "signal": "signal",
    "stop_loss": "stop",
    "take_profit": "objectif",
    "end": "fin",
    "kill_switch": "kill switch",
    "daily_limit": "limite journ.",
}

# Couleurs (palette de référence, voir skill dataviz) : série 1 bleu, statut vert / rouge.
_COLOR_EQUITY = "#2a78d6"
_COLOR_GOOD = "#008300"
_COLOR_BAD = "#e34948"
_COLOR_NEUTRAL = "#52514e"


# --------------------------------------------------------------------------- formatage
def _fmt_pct(x: Any, signed: bool = True, digits: int = 2) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "n/d"
    if math.isnan(x):
        return "n/d"
    if math.isinf(x):
        return "∞" if x > 0 else "-∞"
    return f"{x * 100:+.{digits}f} %" if signed else f"{x * 100:.{digits}f} %"


def _fmt_num(x: Any, digits: int = 2) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "n/d"
    if math.isnan(x):
        return "n/d"
    if math.isinf(x):
        return "∞" if x > 0 else "-∞"
    return f"{x:.{digits}f}"


def _fmt_money(x: Any) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "n/d"
    if not math.isfinite(x):
        return "n/d"
    return f"{x:,.2f}".replace(",", " ")


def _fmt_ts(ts: Any) -> str:
    try:
        return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return str(ts)


def _fmt_params(params: dict[str, Any]) -> str:
    if not params:
        return "(défaut)"
    return ", ".join(f"{k}={v}" for k, v in params.items())


def _fmt_timeframe(index: pd.Index) -> str:
    """Pas de temps des bougies d'après l'écart médian (ex. « 1 h », « 1 j », « 15 min »)."""
    if len(index) < 2:
        return "n/d"
    try:
        deltas = pd.Series(pd.DatetimeIndex(index)).diff().dropna()
    except (TypeError, ValueError):
        return "n/d"
    deltas = deltas[deltas > pd.Timedelta(0)]
    if deltas.empty:
        return "n/d"
    seconds = float(deltas.median().total_seconds())
    if seconds % 86400 == 0:
        return f"{int(seconds // 86400)} j"
    if seconds % 3600 == 0:
        return f"{int(seconds // 3600)} h"
    if seconds % 60 == 0:
        return f"{int(seconds // 60)} min"
    return f"{seconds:g} s"


def _aligned(rows: list[tuple[str, str]]) -> list[str]:
    """Lignes ``label : valeur`` avec les deux-points alignés."""
    if not rows:
        return []
    width = max(len(label) for label, _ in rows)
    return [f"{label.ljust(width)} : {value}" for label, value in rows]


def _wrap(text: str, bullet: str = "- ") -> str:
    return textwrap.fill(
        text, width=TEXT_WIDTH, initial_indent=bullet, subsequent_indent=" " * len(bullet)
    )


def _table(headers: list[str], rows: list[list[str]], align_right: set[int]) -> list[str]:
    """Tableau texte à colonnes alignées (séparateur ``|``)."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(cells: list[str]) -> str:
        parts = []
        for i, cell in enumerate(cells):
            parts.append(cell.rjust(widths[i]) if i in align_right else cell.ljust(widths[i]))
        return " | ".join(parts).rstrip()

    lines = [fmt(headers), "-+-".join("-" * w for w in widths)]
    lines.extend(fmt(row) for row in rows)
    return lines


def _side_label(side: Any) -> str:
    try:
        return "Long" if int(side) > 0 else "Short"
    except (TypeError, ValueError):
        return str(side)


def _trade_rows(trades: pd.DataFrame) -> list[list[str]]:
    rows: list[list[str]] = []
    for _, t in trades.iterrows():
        reason = str(t.get("exit_reason", ""))
        rows.append(
            [
                _fmt_ts(t.get("entry_time")),
                _fmt_ts(t.get("exit_time")),
                _side_label(t.get("side", 1)),
                _fmt_num(t.get("entry_price"), 4),
                _fmt_num(t.get("exit_price"), 4),
                _fmt_money(t.get("pnl")),
                _fmt_pct(t.get("pnl_pct")),
                str(int(t.get("bars_held", 0))) if not pd.isna(t.get("bars_held", 0)) else "n/d",
                EXIT_REASON_LABELS.get(reason, reason),
            ]
        )
    return rows


def _trades_table(trades: pd.DataFrame) -> list[str]:
    headers = [
        "Entrée",
        "Sortie",
        "Sens",
        "Prix entrée",
        "Prix sortie",
        "P&L",
        "P&L %",
        "Bougies",
        "Raison",
    ]
    return _table(headers, _trade_rows(trades), align_right={3, 4, 5, 6, 7})


def _sort_key(trades: pd.DataFrame) -> str:
    return "pnl" if "pnl" in trades.columns else "pnl_pct"


def _benchmark_return(result: BacktestResult, benchmark_return: float | None) -> float | None:
    if benchmark_return is not None:
        value = float(benchmark_return)
        return None if math.isnan(value) else value
    value = result.metrics.get("buy_hold_return")
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) else value


def _total_fees(trades: pd.DataFrame) -> float:
    if trades is None or len(trades) == 0 or "fees" not in trades.columns:
        return 0.0
    fees = np.asarray(trades["fees"], dtype=np.float64)
    return float(np.nansum(fees))


# --------------------------------------------------------------------------- lecture
def _reading(result: BacktestResult, bench: float | None) -> list[str]:
    """Paragraphe « Lecture » en langage simple, pour un débutant."""
    m = result.metrics
    cash = float(result.config.initial_cash)
    final = float(m.get("final_equity", cash))
    total = float(m.get("total_return", 0.0))
    sharpe = float(m.get("sharpe", 0.0))
    mdd = float(m.get("max_drawdown", 0.0))
    mdd_dur = int(m.get("max_drawdown_duration", 0))
    n_trades = int(m.get("n_trades", 0))
    win = float(m.get("win_rate", 0.0))
    pf = float(m.get("profit_factor", 0.0))
    fees = _total_fees(result.trades)
    lines: list[str] = []

    txt = (
        f"Rendement : sur la période, la stratégie aurait transformé {_fmt_money(cash)} en "
        f"{_fmt_money(final)} ({_fmt_pct(total)})"
    )
    if bench is not None:
        verdict = "mieux" if total > bench else ("moins bien" if total < bench else "pareil")
        txt += (
            f", contre {_fmt_pct(bench)} en achetant simplement au début et en conservant "
            f"(buy & hold) : elle fait {verdict} que le marché"
        )
    lines.append(_wrap(txt + "."))

    if sharpe < 0:
        s_txt = "négatif : la stratégie perd de l'argent en moyenne"
    elif sharpe < 1:
        s_txt = "entre 0 et 1 : gain faible par rapport aux secousses subies, résultat médiocre"
    elif sharpe < 2:
        s_txt = "entre 1 et 2 : bon équilibre entre gain et risque"
    else:
        s_txt = (
            "supérieur à 2 : excellent sur le papier, souvent trop beau pour être vrai "
            "(sur-apprentissage ? vérifiez en walk-forward)"
        )
    lines.append(
        _wrap(
            f"Sharpe {_fmt_num(sharpe)} : le ratio de Sharpe mesure le gain obtenu par unité "
            f"de risque pris (volatilité). Ici il est {s_txt}."
        )
    )

    lines.append(
        _wrap(
            f"Drawdown maximal {_fmt_pct(mdd)} : à un moment, le capital est descendu de "
            f"{_fmt_pct(abs(mdd), signed=False)} sous son plus haut précédent et la plus longue "
            f"période passée sous un plus haut a duré {mdd_dur} bougies. Demandez-vous si vous "
            "auriez tenu psychologiquement pendant cette baisse."
        )
    )

    if n_trades == 0:
        lines.append(
            _wrap(
                "Trades : aucun trade sur la période. La stratégie n'a jamais pris position : "
                "vérifiez les paramètres, la période de chauffe ou la durée des données."
            )
        )
    else:
        t_txt = (
            f"Trades : {n_trades} trades, {_fmt_pct(win, signed=False)} de gagnants, profit "
            f"factor {_fmt_num(pf)} (somme des gains divisée par la somme des pertes ; au-dessus "
            "de 1 la stratégie gagne plus qu'elle ne perd)."
        )
        if n_trades < 30:
            t_txt += (
                " Avec moins de 30 trades, le résultat doit beaucoup au hasard et n'est pas "
                "significatif statistiquement."
            )
        lines.append(_wrap(t_txt))
        lines.append(
            _wrap(
                f"Frais : {_fmt_money(fees)} payés en frais de transaction, soit "
                f"{_fmt_pct(fees / cash if cash else 0.0, signed=False)} du capital initial "
                "(le glissement est déjà inclus dans les prix d'exécution). Les coûts sont la "
                "première cause d'échec des stratégies qui tradent souvent."
            )
        )
    lines.append(
        _wrap(
            "Rappel : les performances passées ne garantissent pas les performances futures. "
            "Validez toujours une stratégie en walk-forward puis en paper trading avant "
            "d'engager de l'argent réel, et ne risquez jamais ce que vous ne pouvez pas perdre."
        )
    )
    return lines


# --------------------------------------------------------------------------- API publique
def format_report(result: BacktestResult, benchmark_return: float | None = None) -> str:
    """Rapport texte complet (français, colonnes alignées) d'un backtest.

    :param result: résultat de :func:`~tradingbot.backtest.engine.run_backtest`.
    :param benchmark_return: rendement du buy & hold sur la même période (fraction) ; par
        défaut ``result.metrics["buy_hold_return"]`` s'il existe.
    """
    m = result.metrics
    eq = result.equity
    trades = result.trades if result.trades is not None else pd.DataFrame()
    n_bars = len(eq)
    cfg = result.config
    bench = _benchmark_return(result, benchmark_return)

    title = f"Rapport de backtest : {result.strategy_name}"
    lines: list[str] = [title, "=" * len(title), ""]

    if n_bars:
        periode = f"{_fmt_ts(eq.index[0])} → {_fmt_ts(eq.index[-1])}"
    else:
        periode = "aucune bougie"
    header_rows = [
        ("Stratégie", str(result.strategy_name)),
        ("Paramètres", _fmt_params(result.params)),
        ("Période", periode),
        ("Bougies", f"{n_bars} (pas de {_fmt_timeframe(eq.index)})"),
        ("Capital initial", _fmt_money(cfg.initial_cash)),
        (
            "Coûts par ordre",
            f"frais {_fmt_pct(cfg.fee_rate, signed=False)}, glissement "
            f"{_fmt_pct(cfg.slippage, signed=False)}",
        ),
    ]
    stops = []
    if cfg.stop_loss_pct:
        stops.append(f"stop {_fmt_pct(cfg.stop_loss_pct, signed=False)}")
    if cfg.take_profit_pct:
        stops.append(f"objectif {_fmt_pct(cfg.take_profit_pct, signed=False)}")
    if cfg.trailing_stop_pct:
        stops.append(f"stop suiveur {_fmt_pct(cfg.trailing_stop_pct, signed=False)}")
    if stops:
        header_rows.append(("Stops par défaut", ", ".join(stops)))
    header_rows.append(("Vente à découvert", "autorisée" if cfg.allow_short else "désactivée"))
    lines.extend(_aligned(header_rows))

    lines.extend(["", "Performance", "-----------"])
    perf_rows = [
        ("Capital final", _fmt_money(m.get("final_equity", cfg.initial_cash))),
        ("Rendement total", _fmt_pct(m.get("total_return", 0.0))),
        ("CAGR (rendement annualisé)", _fmt_pct(m.get("cagr", 0.0))),
        ("Volatilité annualisée", _fmt_pct(m.get("annual_volatility", 0.0), signed=False)),
        ("Sharpe", _fmt_num(m.get("sharpe", 0.0))),
        ("Sortino", _fmt_num(m.get("sortino", 0.0))),
        ("Calmar", _fmt_num(m.get("calmar", 0.0))),
        ("Drawdown maximal", _fmt_pct(m.get("max_drawdown", 0.0))),
        ("Durée max du drawdown", f"{int(m.get('max_drawdown_duration', 0))} bougies"),
        ("Exposition", _fmt_pct(m.get("exposure", 0.0), signed=False)),
    ]
    lines.extend(_aligned(perf_rows))

    lines.extend(["", "Trades", "------"])
    trade_rows = [
        ("Nombre de trades", str(int(m.get("n_trades", 0)))),
        ("Taux de réussite", _fmt_pct(m.get("win_rate", 0.0), signed=False)),
        ("Profit factor", _fmt_num(m.get("profit_factor", 0.0))),
        ("Trade moyen", _fmt_pct(m.get("avg_trade_pct", 0.0))),
        ("Gain moyen", _fmt_pct(m.get("avg_win_pct", 0.0))),
        ("Perte moyenne", _fmt_pct(m.get("avg_loss_pct", 0.0))),
        ("Meilleur trade", _fmt_pct(m.get("best_trade_pct", 0.0))),
        ("Pire trade", _fmt_pct(m.get("worst_trade_pct", 0.0))),
        ("Durée moyenne", f"{_fmt_num(m.get('avg_bars_held', 0.0), 1)} bougies"),
        ("Frais totaux", _fmt_money(_total_fees(trades))),
    ]
    lines.extend(_aligned(trade_rows))

    lines.extend(["", "Comparaison avec le buy & hold", "------------------------------"])
    total = float(m.get("total_return", 0.0))
    if bench is None:
        lines.append("Buy & hold : non disponible (aucune série de prix de référence).")
    else:
        diff = total - bench
        verdict = (
            "La stratégie fait mieux que le simple achat-conservation."
            if diff > 0
            else (
                "La stratégie fait moins bien que le simple achat-conservation."
                if diff < 0
                else "La stratégie fait jeu égal avec le simple achat-conservation."
            )
        )
        lines.extend(
            _aligned(
                [
                    ("Stratégie", _fmt_pct(total)),
                    ("Buy & hold", _fmt_pct(bench)),
                    ("Écart", f"{diff * 100:+.2f} points"),
                ]
            )
        )
        lines.append(verdict)

    n_trades = len(trades)
    if n_trades == 0:
        lines.extend(["", "Trades détaillés", "----------------", "Aucun trade sur la période."])
    elif n_trades <= N_EXTREME_TRADES:
        lines.extend(["", f"Tous les trades ({n_trades})", "-" * (20 + len(str(n_trades)))])
        lines.extend(_trades_table(trades.sort_values(_sort_key(trades), kind="stable")))
    else:
        key = _sort_key(trades)
        worst = trades.sort_values(key, ascending=True, kind="stable").head(N_EXTREME_TRADES)
        best = trades.sort_values(key, ascending=False, kind="stable").head(N_EXTREME_TRADES)
        lines.extend(["", f"Les {N_EXTREME_TRADES} pires trades", "-" * 18])
        lines.extend(_trades_table(worst))
        lines.extend(["", f"Les {N_EXTREME_TRADES} meilleurs trades", "-" * 22])
        lines.extend(_trades_table(best))

    lines.extend(["", "Lecture (pour débutant)", "-----------------------"])
    lines.extend(_reading(result, bench))
    return "\n".join(lines)


def _comparison_rows(results: list[BacktestResult]) -> list[tuple[float, list[str]]]:
    names = [str(r.strategy_name) for r in results]
    duplicated = len(set(names)) != len(names)
    rows: list[tuple[float, list[str]]] = []
    for r in results:
        m = r.metrics
        label = str(r.strategy_name)
        if duplicated and r.params:
            label = f"{label} ({_fmt_params(r.params)})"
        sharpe = float(m.get("sharpe", math.nan))
        rows.append(
            (
                sharpe,
                [
                    label,
                    _fmt_pct(m.get("total_return", 0.0)),
                    _fmt_pct(m.get("cagr", 0.0)),
                    _fmt_num(sharpe),
                    _fmt_pct(m.get("max_drawdown", 0.0)),
                    str(int(m.get("n_trades", 0))),
                    _fmt_pct(m.get("win_rate", 0.0), signed=False),
                ],
            )
        )
    # tri décroissant par Sharpe, NaN en dernier, stable
    rows.sort(key=lambda item: (math.isnan(item[0]), -item[0] if not math.isnan(item[0]) else 0))
    return rows


def _buy_hold_row(results: list[BacktestResult], benchmark: pd.Series | None) -> list[str]:
    ref = results[0]
    n_bars = len(ref.equity)
    ppy = float(getattr(ref, "periods_per_year", 365.0))
    if benchmark is not None and len(benchmark) >= 2:
        prices = np.asarray(benchmark, dtype=np.float64).ravel()
        prices = prices[np.isfinite(prices)]
        if prices.size >= 2 and prices[0] > 0:
            bh_equity = pd.Series(float(ref.config.initial_cash) * prices / prices[0])
            try:
                bm = metrics_mod.compute_metrics(bh_equity, None, ppy)
            except OverflowError:  # CAGR astronomique sur très peu de bougies
                logger.debug("Métriques buy & hold non calculables (dépassement numérique)")
            else:
                return [
                    "Buy & hold",
                    _fmt_pct(bm["total_return"]),
                    _fmt_pct(bm["cagr"]),
                    _fmt_num(bm["sharpe"]),
                    _fmt_pct(bm["max_drawdown"]),
                    "1",
                    "–",
                ]
    bh = ref.metrics.get("buy_hold_return")
    if bh is None:
        return ["Buy & hold", "n/d", "n/d", "–", "–", "1", "–"]
    bh = float(bh)
    if n_bars > 1 and bh > -1.0 and ppy > 0:
        try:
            bh_cagr = (1.0 + bh) ** (ppy / (n_bars - 1)) - 1.0
        except OverflowError:
            bh_cagr = math.inf
    else:
        bh_cagr = -1.0 if bh <= -1.0 else 0.0
    return ["Buy & hold", _fmt_pct(bh), _fmt_pct(bh_cagr), "–", "–", "1", "–"]


def format_comparison(results: list[BacktestResult], benchmark: pd.Series | None = None) -> str:
    """Tableau comparatif : une ligne par stratégie (triées par Sharpe décroissant) plus
    la ligne « Buy & hold ».

    :param results: résultats à comparer (idéalement sur les mêmes bougies).
    :param benchmark: série de prix (``close``) de la période ; si fournie, la ligne buy &
        hold a aussi un Sharpe et un drawdown ; sinon seul le rendement (tiré de
        ``metrics["buy_hold_return"]``) et le CAGR correspondant sont affichés.
    """
    results = list(results)
    if not results:
        return "Aucun résultat à comparer."
    headers = ["Stratégie", "Rendement", "CAGR", "Sharpe", "Max DD", "Trades", "Réussite"]
    ranked = [row for _, row in _comparison_rows(results)]
    all_rows = [*ranked, _buy_hold_row(results, benchmark)]
    table = _table(headers, all_rows, align_right={1, 2, 3, 4, 5, 6})
    # la ligne buy & hold est séparée des stratégies par un trait aux mêmes largeurs
    widths = [len(h) for h in headers]
    for row in all_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    table.insert(len(table) - 1, "-+-".join("-" * w for w in widths))
    title = "Comparaison des stratégies (triées par Sharpe décroissant)"
    return "\n".join([title, "=" * len(title), *table])


def _json_safe(obj: Any) -> Any:
    """Convertit récursivement en types JSON stricts (NaN/inf → ``null``)."""
    if obj is None or isinstance(obj, bool | str):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.generic):
        return _json_safe(obj.item())
    if isinstance(obj, pd.Timestamp | _dt.datetime | _dt.date):
        return obj.isoformat()
    if isinstance(obj, pd.Timedelta | _dt.timedelta):
        return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return _json_safe(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple | set | frozenset | np.ndarray | pd.Series | pd.Index):
        return [_json_safe(v) for v in list(obj)]
    return str(obj)


def _metrics_payload(result: BacktestResult) -> dict[str, Any]:
    eq = result.equity
    return _json_safe(
        {
            "strategy": result.strategy_name,
            "params": dict(result.params or {}),
            "config": result.config,
            "start": eq.index[0] if len(eq) else None,
            "end": eq.index[-1] if len(eq) else None,
            "n_bars": int(len(eq)),
            "periods_per_year": float(getattr(result, "periods_per_year", 365.0)),
            "metrics": dict(result.metrics),
        }
    )


def save_report(result: BacktestResult, out_dir: str | Path) -> list[Path]:
    """Écrit le rapport dans ``out_dir`` et renvoie les chemins créés.

    Fichiers : ``equity.csv`` (timestamp, equity, returns, position, drawdown),
    ``trades.csv`` (une ligne par trade), ``summary.txt`` (:func:`format_report`),
    ``metrics.json`` (stratégie, paramètres, configuration, période et métriques ; NaN et
    infinis écrits ``null``) et ``equity.png`` si matplotlib est disponible.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    equity = result.equity.rename("equity")
    frame = pd.concat(
        [
            equity,
            result.returns.rename("returns"),
            result.positions.rename("position"),
            metrics_mod.drawdown_series(equity).rename("drawdown"),
        ],
        axis=1,
    )
    frame.index.name = "timestamp"
    equity_path = out / "equity.csv"
    frame.to_csv(equity_path, index=True)
    paths.append(equity_path)

    trades_path = out / "trades.csv"
    trades = result.trades if result.trades is not None else pd.DataFrame()
    trades.to_csv(trades_path, index=False)
    paths.append(trades_path)

    summary_path = out / "summary.txt"
    summary_path.write_text(format_report(result) + "\n", encoding="utf-8")
    paths.append(summary_path)

    metrics_path = out / "metrics.json"
    metrics_path.write_text(
        json.dumps(_metrics_payload(result), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    paths.append(metrics_path)

    try:
        png = plot_result(result, out / "equity.png")
    except Exception as exc:  # noqa: BLE001 - un graphique raté ne doit pas perdre le rapport
        logger.warning("Graphique non généré : %s", exc)
        png = None
    if png is not None:
        paths.append(Path(png))
    logger.info("Rapport écrit dans %s (%d fichiers)", out, len(paths))
    return paths


def plot_result(result: BacktestResult, path: str | Path) -> Path | None:
    """Trace l'équité (avec marqueurs d'entrée/sortie) et le drawdown dans ``path``.

    matplotlib est importé paresseusement : s'il n'est pas installé, un avertissement est
    journalisé et la fonction renvoie ``None`` sans lever d'exception.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib n'est pas installé : graphique ignoré (pip install matplotlib)")
        return None

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    eq = result.equity
    dd = metrics_mod.drawdown_series(eq) * 100.0
    m = result.metrics

    fig, (ax_eq, ax_dd) = plt.subplots(
        2, 1, sharex=True, figsize=(12, 7), gridspec_kw={"height_ratios": [3, 1]}
    )
    ax_eq.plot(eq.index, eq.to_numpy(), color=_COLOR_EQUITY, linewidth=1.6, label="Équité")
    ax_eq.axhline(
        float(result.config.initial_cash),
        color=_COLOR_NEUTRAL,
        linestyle="--",
        linewidth=0.8,
        label="Capital initial",
    )
    trades = result.trades
    if trades is not None and len(trades):
        entries = pd.DatetimeIndex(pd.to_datetime(trades["entry_time"], utc=True))
        exits = pd.DatetimeIndex(pd.to_datetime(trades["exit_time"], utc=True))
        eq_entries = eq.reindex(entries).to_numpy()
        eq_exits = eq.reindex(exits).to_numpy()
        sides = np.asarray(trades["side"], dtype=np.int64)
        longs = sides > 0
        if longs.any():
            ax_eq.scatter(
                entries[longs],
                eq_entries[longs],
                marker="^",
                s=36,
                color=_COLOR_GOOD,
                zorder=3,
                label="Entrée long",
            )
        if (~longs).any():
            ax_eq.scatter(
                entries[~longs],
                eq_entries[~longs],
                marker="v",
                s=36,
                color=_COLOR_BAD,
                zorder=3,
                label="Entrée short",
            )
        ax_eq.scatter(
            exits, eq_exits, marker="x", s=30, color=_COLOR_NEUTRAL, zorder=3, label="Sortie"
        )
    ax_eq.set_title(
        f"{result.strategy_name} — rendement {_fmt_pct(m.get('total_return', 0.0))}, "
        f"Sharpe {_fmt_num(m.get('sharpe', 0.0))}, drawdown max "
        f"{_fmt_pct(m.get('max_drawdown', 0.0))}"
    )
    ax_eq.set_ylabel("Équité")
    ax_eq.grid(True, alpha=0.25)
    ax_eq.legend(loc="best", frameon=False)

    ax_dd.fill_between(dd.index, dd.to_numpy(), 0.0, color=_COLOR_BAD, alpha=0.35)
    ax_dd.plot(dd.index, dd.to_numpy(), color=_COLOR_BAD, linewidth=0.8)
    ax_dd.set_ylabel("Drawdown (%)")
    ax_dd.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(target, dpi=110)
    plt.close(fig)
    logger.info("Graphique enregistré : %s", target)
    return target
