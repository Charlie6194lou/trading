"""Recherche sur grille et validation *walk-forward*.

Pourquoi ? Optimiser les paramètres d'une stratégie sur tout l'historique puis juger
le résultat sur ce même historique revient à « réviser avec les réponses de l'examen » :
on surestime presque toujours la performance (sur-apprentissage). Le walk-forward
découpe l'historique en fenêtres successives :

1. on choisit les paramètres sur une fenêtre d'**entraînement** (*in-sample*, IS) ;
2. on les applique, sans les retoucher, sur la fenêtre de **test** qui suit
   (*out-of-sample*, OOS) ;
3. on avance de ``test_bars`` bougies et on recommence.

La courbe OOS obtenue en enchaînant les fenêtres de test est une estimation honnête de
ce que la stratégie aurait donné « en vrai ». Le :attr:`WalkForwardResult.overfit_ratio`
(métrique OOS moyenne / métrique IS moyenne) résume l'écart : proche de 1 = robuste,
très inférieur à 1 = sur-apprentissage.

Conventions :

- toutes les bornes de fenêtres sont des **positions** (``iloc``) ; fin exclusive ;
- ``train_bars`` et ``test_bars`` sont obligatoires (voir :func:`default_windows`) ;
- la métrique d'optimisation doit être « plus grand = mieux » (``sharpe``, ``sortino``,
  ``calmar``, ``total_return``, ``cagr``, ``profit_factor``, ``max_drawdown`` qui est négatif…).
"""

from __future__ import annotations

import itertools
import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from tradingbot.backtest import metrics as metrics_mod
from tradingbot.backtest.engine import BacktestConfig, run_backtest

if TYPE_CHECKING:
    from tradingbot.risk import RiskManager
    from tradingbot.strategies.base import Strategy

try:  # le chargeur de données peut ne pas être disponible (tests isolés)
    from tradingbot.data.loader import ensure_ohlcv as _ensure_ohlcv
except ImportError:  # pragma: no cover - dépend de l'état du dépôt
    _ensure_ohlcv = None

logger = logging.getLogger(__name__)

__all__ = [
    "WalkForwardResult",
    "default_windows",
    "grid_search",
    "walk_forward",
]

#: clés du dictionnaire décrivant chaque fenêtre (contrat ``docs/ARCHITECTURE.md``)
WINDOW_KEYS: tuple[str, ...] = (
    "train_start",
    "train_end",
    "test_start",
    "test_end",
    "best_params",
    "is_metric",
    "oos_metric",
)
_OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


# --------------------------------------------------------------------------- utilitaires
def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Bougies au format canonique (index UTC trié unique, colonnes OHLCV float64)."""
    if _ensure_ohlcv is not None:
        return _ensure_ohlcv(df)
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    missing = [c for c in _OHLCV_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"Colonnes OHLCV manquantes : {missing}")
    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("L'index des bougies doit être un DatetimeIndex (UTC)")
    idx = out.index.tz_localize("UTC") if out.index.tz is None else out.index.tz_convert("UTC")
    out.index = idx
    out = out.loc[:, list(_OHLCV_COLUMNS)].astype(np.float64)
    if not out.index.is_unique:
        out = out.loc[~out.index.duplicated(keep="last")]
    if not out.index.is_monotonic_increasing:
        out = out.sort_index()
    out.index.name = "timestamp"
    return out


def _as_values(name: str, value: Any) -> list[Any]:
    """Valeurs candidates d'un paramètre : un scalaire devient ``[scalaire]``."""
    if isinstance(value, str | bytes) or not isinstance(value, Iterable):
        return [value]
    values = list(value)
    if not values:
        raise ValueError(f"La grille du paramètre {name!r} est vide")
    return values


def _clean_grid(grid: dict[str, Any] | None, strategy_cls: type[Strategy]) -> dict[str, list[Any]]:
    """Grille normalisée ``{param: [valeurs]}`` ; vérifie que les paramètres existent."""
    if grid is None:
        grid = dict(getattr(strategy_cls, "param_grid", {}) or {})
    if not isinstance(grid, dict):
        raise TypeError("`grid` doit être un dictionnaire {paramètre: [valeurs]}")
    cleaned = {str(k): _as_values(str(k), v) for k, v in grid.items()}
    known = getattr(strategy_cls, "default_params", None)
    if isinstance(known, dict) and known:
        unknown = sorted(set(cleaned) - set(known))
        if unknown:
            name = getattr(strategy_cls, "name", strategy_cls.__name__)
            raise ValueError(
                f"Paramètres inconnus dans la grille pour la stratégie {name!r} : {unknown} "
                f"(paramètres possibles : {sorted(known)})"
            )
    collisions = sorted(set(cleaned) & set(metrics_mod.METRIC_KEYS))
    if collisions:
        raise ValueError(
            f"Les paramètres {collisions} portent le nom d'une métrique : renommez-les"
        )
    return cleaned


def _combinations(grid: dict[str, list[Any]], max_combinations: int | None) -> list[dict[str, Any]]:
    """Produit cartésien de la grille, plafonné à ``max_combinations`` (avec avertissement)."""
    if max_combinations is not None and int(max_combinations) <= 0:
        raise ValueError("max_combinations doit être strictement positif (ou None)")
    keys = list(grid)
    total = math.prod(len(grid[k]) for k in keys)
    product: Iterable[tuple[Any, ...]] = itertools.product(*(grid[k] for k in keys))
    if max_combinations is not None and total > max_combinations:
        logger.warning(
            "Grille de %d combinaisons plafonnée à %d : %d combinaisons ignorées "
            "(réduisez la grille ou augmentez max_combinations)",
            total,
            max_combinations,
            total - int(max_combinations),
        )
        product = itertools.islice(product, int(max_combinations))
    return [dict(zip(keys, values, strict=True)) for values in product]


def _sort_table(
    table: pd.DataFrame, metric: str, params: list[dict[str, Any]] | None = None
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Trie ``table`` par ``metric`` décroissant (NaN en dernier), ``params`` aligné.

    Si la colonne ``metric`` n'existe pas, un avertissement est émis et l'ordre est
    conservé.
    """
    params = list(params) if params is not None else [{} for _ in range(len(table))]
    if len(table) == 0:
        return table.reset_index(drop=True), params
    if metric not in table.columns:
        logger.warning(
            "Métrique %r inconnue : résultats non triés (métriques disponibles : %s)",
            metric,
            ", ".join(metrics_mod.METRIC_KEYS),
        )
        return table.reset_index(drop=True), params
    ordered = table.sort_values(metric, ascending=False, na_position="last", kind="stable")
    order = ordered.index.to_numpy()
    params = [params[int(i)] for i in order]
    return ordered.reset_index(drop=True), params


def _evaluate_grid(
    df: pd.DataFrame,
    strategy_cls: type[Strategy],
    combos: list[dict[str, Any]],
    config: BacktestConfig | None,
    risk: RiskManager | None,
    metric: str,
    param_names: list[str],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Backteste chaque combinaison ; renvoie (tableau trié, paramètres alignés)."""
    rows: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []
    for params in combos:
        try:
            strategy = strategy_cls(**params)
            result = run_backtest(df, strategy, config=config, risk=risk)
        except Exception as exc:  # noqa: BLE001 - une combinaison ne doit pas tout arrêter
            logger.warning("Combinaison %s ignorée : %s", params, exc)
            logger.debug("Détail de l'erreur pour %s", params, exc_info=exc)
            continue
        row: dict[str, Any] = dict(params)
        row.update(result.metrics)
        rows.append(row)
        kept.append(dict(params))
    columns = [*param_names, *metrics_mod.METRIC_KEYS]
    if not rows:
        logger.warning("Aucune combinaison n'a pu être évaluée sur %d bougies", len(df))
        return pd.DataFrame(columns=columns), []
    table = pd.DataFrame(rows)
    for col in columns:
        if col not in table.columns:
            table[col] = np.nan
    extra = [c for c in table.columns if c not in columns]
    table = table.loc[:, [*columns, *extra]]
    return _sort_table(table, metric, kept)


def _finite_mean(values: Iterable[float]) -> float:
    """Moyenne en ignorant les NaN ; NaN si aucune valeur."""
    vals = [float(v) for v in values if not math.isnan(float(v))]
    return float(np.mean(vals)) if vals else math.nan


def _overfit_ratio(is_mean: float, oos_mean: float) -> float:
    """``oos_mean / is_mean`` si ``is_mean`` est fini et > 0, sinon NaN."""
    if not math.isfinite(is_mean) or is_mean <= 0 or math.isnan(oos_mean):
        return math.nan
    return float(oos_mean / is_mean)


def _fmt_ts(ts: Any) -> str:
    try:
        return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return str(ts)


def _fmt_num(x: float, digits: int = 2) -> str:
    x = float(x)
    if math.isnan(x):
        return "n/d"
    if math.isinf(x):
        return "∞" if x > 0 else "-∞"
    return f"{x:.{digits}f}"


def _fmt_pct(x: float, signed: bool = True) -> str:
    x = float(x)
    if math.isnan(x):
        return "n/d"
    if math.isinf(x):
        return "∞"
    return f"{x * 100:+.2f} %" if signed else f"{x * 100:.2f} %"


def _fmt_money(x: float) -> str:
    return f"{float(x):,.2f}".replace(",", " ")


def _fmt_params(params: dict[str, Any]) -> str:
    if not params:
        return "(défaut)"
    return ", ".join(f"{k}={v}" for k, v in params.items())


# --------------------------------------------------------------------------- API publique
def default_windows(n_bars: int, n_windows: int = 4, train_test_ratio: int = 3) -> tuple[int, int]:
    """Tailles ``(train_bars, test_bars)`` raisonnables pour ``n_bars`` bougies.

    Donne exactement ``n_windows`` fenêtres de test de même taille (sans reste) avec
    une fenêtre d'entraînement d'environ ``train_test_ratio`` fois la fenêtre de test :
    ``test = n_bars // (n_windows + ratio)``, ``train = n_bars - n_windows × test``.
    Exemple : 300 bougies → ``(132, 42)``.

    :raises ValueError: s'il n'y a pas assez de bougies.
    """
    n_bars = int(n_bars)
    n_windows = int(n_windows)
    train_test_ratio = int(train_test_ratio)
    if n_windows < 1 or train_test_ratio < 1:
        raise ValueError("n_windows et train_test_ratio doivent être ≥ 1")
    test_bars = n_bars // (n_windows + train_test_ratio)
    if test_bars < 1:
        raise ValueError(
            f"Pas assez de bougies ({n_bars}) pour {n_windows} fenêtres de walk-forward : "
            f"il en faut au moins {n_windows + train_test_ratio}"
        )
    train_bars = n_bars - n_windows * test_bars
    return train_bars, test_bars


def grid_search(
    df: pd.DataFrame,
    strategy_cls: type[Strategy],
    grid: dict[str, Any] | None = None,
    config: BacktestConfig | None = None,
    risk: RiskManager | None = None,
    metric: str = "sharpe",
    max_combinations: int | None = 200,
) -> pd.DataFrame:
    """Backteste toutes les combinaisons de paramètres de ``grid`` sur ``df``.

    :param df: bougies OHLCV.
    :param strategy_cls: classe de stratégie (instanciée avec ``strategy_cls(**params)``).
    :param grid: ``{paramètre: [valeurs]}`` ; par défaut ``strategy_cls.param_grid``. Un
        scalaire est accepté à la place d'une liste. Une grille vide évalue les
        paramètres par défaut (une seule ligne).
    :param config: :class:`~tradingbot.backtest.engine.BacktestConfig` (défaut si None).
    :param risk: gestionnaire de risque partagé (remis à zéro à chaque backtest).
    :param metric: colonne de tri, « plus grand = mieux » (``sharpe`` par défaut).
    :param max_combinations: plafond de combinaisons évaluées, dans l'ordre du produit
        cartésien ; le dépassement est signalé par un avertissement. ``None`` = illimité.
    :returns: un ``DataFrame`` avec une colonne par paramètre de la grille puis toutes les
        métriques de :func:`~tradingbot.backtest.metrics.compute_metrics`, trié par
        ``metric`` décroissant (NaN en dernier). Une combinaison qui lève une exception est
        journalisée et ignorée (pas de ligne).
    :raises ValueError: paramètre inconnu, grille vide pour un paramètre, plafond ≤ 0.
    """
    data = _normalise(df)
    cleaned = _clean_grid(grid, strategy_cls)
    combos = _combinations(cleaned, max_combinations)
    name = getattr(strategy_cls, "name", strategy_cls.__name__)
    logger.info(
        "Recherche sur grille : stratégie %s, %d combinaison(s), %d bougies, métrique %s",
        name,
        len(combos),
        len(data),
        metric,
    )
    table, _ = _evaluate_grid(data, strategy_cls, combos, config, risk, metric, list(cleaned))
    return table


@dataclass
class WalkForwardResult:
    """Résultat d'une validation walk-forward (voir :func:`walk_forward`)."""

    #: une entrée par fenêtre : positions ``train_start``/``train_end``/``test_start``/
    #: ``test_end`` (fin exclusive), ``best_params``, ``is_metric``, ``oos_metric`` et des
    #: informations complémentaires (horodatages, nombre de trades, rendement OOS)
    windows: list[dict[str, Any]]
    #: équité hors échantillon recomposée en enchaînant les rendements des fenêtres de
    #: test, à partir de ``config.initial_cash``
    oos_equity: pd.Series
    #: métriques de ``oos_equity`` (:func:`~tradingbot.backtest.metrics.compute_metrics`)
    oos_metrics: dict[str, float | int]
    #: moyenne de la métrique d'optimisation sur les fenêtres d'entraînement
    is_metrics_mean: float
    #: moyenne de la métrique d'optimisation sur les fenêtres de test
    oos_metrics_mean: float
    #: ``oos_metrics_mean / is_metrics_mean`` (NaN si la moyenne IS n'est pas > 0)
    overfit_ratio: float
    strategy_name: str = ""
    metric: str = "sharpe"
    train_bars: int = 0
    test_bars: int = 0
    anchored: bool = False
    #: rendements par bougie de ``oos_equity``
    oos_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=np.float64))
    #: position tenue à chaque bougie OOS
    oos_positions: pd.Series = field(default_factory=lambda: pd.Series(dtype=np.int64))
    #: trades des fenêtres de test (colonnes de ``BacktestResult.trades`` + ``window``) ;
    #: ``pnl`` est exprimé par rapport au capital initial de chaque fenêtre
    oos_trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    periods_per_year: float = 365.0

    @property
    def n_windows(self) -> int:
        return len(self.windows)

    def windows_frame(self) -> pd.DataFrame:
        """Les fenêtres sous forme de tableau (une ligne par fenêtre)."""
        if not self.windows:
            return pd.DataFrame(columns=list(WINDOW_KEYS))
        rows = []
        for w in self.windows:
            row = dict(w)
            row["best_params"] = _fmt_params(w.get("best_params", {}))
            rows.append(row)
        return pd.DataFrame(rows)

    def summary(self) -> str:
        """Compte rendu texte en français : fenêtres, équité OOS et ratio de sur-apprentissage."""
        m = self.oos_metrics
        mode = "ancré (entraînement depuis le début)" if self.anchored else "glissant"
        lines = [f"=== Walk-forward : {self.strategy_name} (métrique : {self.metric}) ==="]
        lines.append(
            f"Fenêtres : {self.n_windows} (entraînement {self.train_bars} bougies, "
            f"test {self.test_bars} bougies, {mode})"
        )
        if len(self.oos_equity):
            lines.append(
                f"Période OOS : {_fmt_ts(self.oos_equity.index[0])} → "
                f"{_fmt_ts(self.oos_equity.index[-1])} ({len(self.oos_equity)} bougies)"
            )
        lines.append("")
        header = (
            f"{'N':>2} | {'Entraînement':<35} | {'Test':<35} | "
            f"{'IS ' + self.metric:>12} | {'OOS ' + self.metric:>12} | Paramètres"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for w in self.windows:
            train = f"{_fmt_ts(w.get('train_start_time'))} → {_fmt_ts(w.get('train_end_time'))}"
            test = f"{_fmt_ts(w.get('test_start_time'))} → {_fmt_ts(w.get('test_end_time'))}"
            lines.append(
                f"{int(w.get('window', 0)):>2} | {train:<35} | {test:<35} | "
                f"{_fmt_num(w.get('is_metric', math.nan)):>12} | "
                f"{_fmt_num(w.get('oos_metric', math.nan)):>12} | "
                f"{_fmt_params(w.get('best_params', {}))}"
            )
        lines.append("")
        if len(self.oos_equity):
            lines.append(
                f"Équité OOS : {_fmt_money(self.oos_equity.iloc[0])} → "
                f"{_fmt_money(self.oos_equity.iloc[-1])} "
                f"({_fmt_pct(m.get('total_return', 0.0))}), CAGR {_fmt_pct(m.get('cagr', 0.0))}, "
                f"Sharpe {_fmt_num(m.get('sharpe', 0.0))}, drawdown max "
                f"{_fmt_pct(m.get('max_drawdown', 0.0))}, {int(m.get('n_trades', 0))} trades"
            )
            if "buy_hold_return" in m:
                lines.append(
                    f"Buy & hold sur la même période OOS : {_fmt_pct(m['buy_hold_return'])}"
                )
        label_w = len(f"Moyenne OOS ({self.metric})")
        lines.append(
            f"{f'Moyenne IS ({self.metric})':<{label_w}} : {_fmt_num(self.is_metrics_mean)}"
        )
        lines.append(
            f"{f'Moyenne OOS ({self.metric})':<{label_w}} : {_fmt_num(self.oos_metrics_mean)}"
        )
        lines.append(f"{'Ratio OOS / IS':<{label_w}} : {_fmt_num(self.overfit_ratio)}")
        lines.append(
            "Lecture : un ratio proche de 1 signifie que la stratégie se comporte hors "
            "échantillon comme en échantillon (robuste) ; un ratio très inférieur à 1, nul ou "
            "négatif révèle du sur-apprentissage (les paramètres optimisés ne généralisent "
            "pas) ; n/d = moyenne IS nulle ou négative, le ratio n'a pas de sens."
        )
        return "\n".join(lines)


def walk_forward(
    df: pd.DataFrame,
    strategy_cls: type[Strategy],
    grid: dict[str, Any] | None = None,
    train_bars: int | None = None,
    test_bars: int | None = None,
    config: BacktestConfig | None = None,
    risk: RiskManager | None = None,
    metric: str = "sharpe",
    anchored: bool = False,
    max_combinations: int | None = 200,
) -> WalkForwardResult:
    """Validation walk-forward : optimise sur une fenêtre, teste sur la suivante.

    Fenêtre ``k`` (positions, fin exclusive) :

    - entraînement ``[test_start - train_bars, test_start)`` (ou ``[0, test_start)`` si
      ``anchored``) ;
    - test ``[test_start, min(test_start + test_bars, n))`` avec
      ``test_start = train_bars + k × test_bars``.

    Les fenêtres de test couvrent donc exactement les bougies ``[train_bars, n)``, sans
    chevauchement (la dernière peut être tronquée). Sur chaque fenêtre d'entraînement,
    :func:`grid_search` choisit les meilleurs paramètres ; le backtest de test est lancé
    sur ``strategy.warmup_bars()`` bougies d'historique + la fenêtre de test (pour amorcer
    les indicateurs), puis l'équité et les trades sont restreints à la fenêtre de test. Les
    rendements des fenêtres de test sont enchaînés à partir de ``config.initial_cash`` pour
    former :attr:`WalkForwardResult.oos_equity`.

    :param train_bars: taille de la fenêtre d'entraînement (obligatoire).
    :param test_bars: taille de la fenêtre de test et pas d'avancement (obligatoire).
    :param anchored: ``True`` = l'entraînement commence toujours à la première bougie.
    :param max_combinations: plafond transmis à la recherche sur grille.
    :raises ValueError: fenêtres impossibles (``train_bars + test_bars > len(df)``,
        tailles non positives ou absentes, grille invalide).
    :raises RuntimeError: aucune combinaison évaluable sur une fenêtre d'entraînement.
    """
    data = _normalise(df)
    n = len(data)
    if train_bars is None or test_bars is None:
        raise ValueError(
            "train_bars et test_bars sont obligatoires : utilisez default_windows(len(df)) "
            "pour obtenir des valeurs raisonnables"
        )
    train_bars = int(train_bars)
    test_bars = int(test_bars)
    if train_bars <= 0 or test_bars <= 0:
        raise ValueError("train_bars et test_bars doivent être strictement positifs")
    if train_bars + test_bars > n:
        raise ValueError(
            f"Pas assez de bougies pour le walk-forward : {n} disponibles, il en faut au "
            f"moins train_bars + test_bars = {train_bars + test_bars}"
        )
    cfg = config if config is not None else BacktestConfig()
    cleaned = _clean_grid(grid, strategy_cls)
    param_names = list(cleaned)
    combos = _combinations(cleaned, max_combinations)
    name = str(getattr(strategy_cls, "name", strategy_cls.__name__))
    logger.info(
        "Walk-forward %s : %d bougies, entraînement %d, test %d, %s, %d combinaison(s)",
        name,
        n,
        train_bars,
        test_bars,
        "ancré" if anchored else "glissant",
        len(combos),
    )

    windows: list[dict[str, Any]] = []
    returns_parts: list[pd.Series] = []
    positions_parts: list[pd.Series] = []
    trades_parts: list[pd.DataFrame] = []
    empty_trades: pd.DataFrame | None = None
    ppy = 365.0
    warned_warmup = False

    test_start = train_bars
    while test_start < n:
        k = len(windows) + 1
        train_start = 0 if anchored else test_start - train_bars
        train_end = test_start
        test_end = min(test_start + test_bars, n)
        train_df = data.iloc[train_start:train_end]

        table, ranked = _evaluate_grid(
            train_df, strategy_cls, combos, cfg, risk, metric, param_names
        )
        if not ranked:
            raise RuntimeError(
                f"Fenêtre {k} : aucune combinaison de paramètres n'a pu être évaluée sur la "
                f"fenêtre d'entraînement [{train_start}, {train_end}) (voir les avertissements)"
            )
        best = ranked[0]
        is_metric = float(table[metric].iloc[0]) if metric in table.columns else math.nan

        strategy = strategy_cls(**best)
        warmup = max(0, int(getattr(strategy, "warmup_bars", lambda: 0)()))
        if warmup > train_bars and not warned_warmup:
            warned_warmup = True
            logger.warning(
                "La période de chauffe (%d bougies) dépasse la fenêtre d'entraînement (%d) : "
                "les métriques IS sont peu fiables, augmentez train_bars",
                warmup,
                train_bars,
            )
        hist_start = max(0, test_start - warmup)
        test_df = data.iloc[hist_start:test_end]
        result = run_backtest(test_df, strategy, config=cfg, risk=risk)
        ppy = float(result.periods_per_year)
        offset = test_start - hist_start

        equity = result.equity.iloc[offset:]
        rets = equity.pct_change().fillna(0.0)
        rets.name = "returns"
        positions = result.positions.iloc[offset:]
        trades = result.trades
        if empty_trades is None:
            empty_trades = trades.iloc[0:0].copy()
        if len(trades):
            trades = trades.loc[trades["entry_time"] >= equity.index[0]]
        oos_m = metrics_mod.compute_metrics(
            equity,
            trades,
            ppy,
            benchmark=test_df["close"].iloc[offset:],
            positions=positions,
        )
        oos_metric = float(oos_m.get(metric, math.nan))

        windows.append(
            {
                "window": k,
                "train_start": int(train_start),
                "train_end": int(train_end),
                "test_start": int(test_start),
                "test_end": int(test_end),
                "best_params": dict(best),
                "is_metric": is_metric,
                "oos_metric": oos_metric,
                "train_start_time": data.index[train_start],
                "train_end_time": data.index[train_end - 1],
                "test_start_time": data.index[test_start],
                "test_end_time": data.index[test_end - 1],
                "n_test_bars": int(test_end - test_start),
                "n_oos_trades": int(len(trades)),
                "oos_return": float(oos_m.get("total_return", 0.0)),
            }
        )
        logger.info(
            "Fenêtre %d : entraînement [%d, %d) → %s (IS %s = %s) ; test [%d, %d) : "
            "OOS %s = %s, rendement %s, %d trades",
            k,
            train_start,
            train_end,
            _fmt_params(best),
            metric,
            _fmt_num(is_metric),
            test_start,
            test_end,
            metric,
            _fmt_num(oos_metric),
            _fmt_pct(oos_m.get("total_return", 0.0)),
            len(trades),
        )
        returns_parts.append(rets)
        positions_parts.append(positions)
        if len(trades):
            trades_parts.append(trades.assign(window=k))
        test_start += test_bars

    oos_returns = pd.concat(returns_parts)
    oos_returns.name = "returns"
    oos_equity = float(cfg.initial_cash) * (1.0 + oos_returns).cumprod()
    oos_equity.name = "equity"
    oos_positions = pd.concat(positions_parts)
    oos_positions.name = "position"
    if trades_parts:
        oos_trades = pd.concat(trades_parts, ignore_index=True)
    else:
        oos_trades = (empty_trades if empty_trades is not None else pd.DataFrame()).assign(
            window=pd.Series(dtype=np.int64)
        )
    oos_metrics = metrics_mod.compute_metrics(
        oos_equity,
        oos_trades,
        ppy,
        benchmark=data["close"].iloc[train_bars:],
        positions=oos_positions,
    )
    is_mean = _finite_mean(w["is_metric"] for w in windows)
    oos_mean = _finite_mean(w["oos_metric"] for w in windows)
    ratio = _overfit_ratio(is_mean, oos_mean)
    logger.info(
        "Walk-forward terminé : %d fenêtres, rendement OOS %s, %s IS moyen %s, OOS moyen %s, "
        "ratio %s",
        len(windows),
        _fmt_pct(oos_metrics.get("total_return", 0.0)),
        metric,
        _fmt_num(is_mean),
        _fmt_num(oos_mean),
        _fmt_num(ratio),
    )
    return WalkForwardResult(
        windows=windows,
        oos_equity=oos_equity,
        oos_metrics=oos_metrics,
        is_metrics_mean=is_mean,
        oos_metrics_mean=oos_mean,
        overfit_ratio=ratio,
        strategy_name=name,
        metric=metric,
        train_bars=train_bars,
        test_bars=test_bars,
        anchored=anchored,
        oos_returns=oos_returns,
        oos_positions=oos_positions,
        oos_trades=oos_trades,
        periods_per_year=ppy,
    )
