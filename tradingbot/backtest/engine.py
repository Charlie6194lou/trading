"""Moteur de backtest bougie par bougie.

Principe (voir ``docs/ARCHITECTURE.md``, section « Règles du moteur ») :

1. La position tenue pendant la bougie ``t`` a été décidée à la clôture de ``t-1``
   (``signal[t-1]``). Les changements s'exécutent à ``open[t]`` ± slippage, avec des
   frais de ``notionnel × fee_rate`` sur chaque ordre.
2. Stop et take-profit sont vérifiés sur chaque bougie où une position est tenue, dans
   cet ordre : gap d'ouverture au-delà du stop (sortie à ``open``), gap d'ouverture
   au-delà du take-profit (sortie à ``open`` : l'ouverture est le premier prix connu,
   l'ambiguïté est levée), puis touches intrabar : stop (au niveau exact) avant
   take-profit. Stop et take-profit touchés **intrabar** dans la même bougie → on suppose
   le stop (pessimiste).
3. Une position ne s'ouvre **jamais** sans stop : ``risk.protective_stop(side, prix
   d'exécution, stop de la stratégie, ATR(14) des bougies ≤ t-1, stop_loss_pct)`` choisit,
   dans l'ordre, le stop de la stratégie (s'il est du bon côté du prix), ``stop_loss_pct``,
   un stop à ``default_stop_atr_mult × ATR`` ou, à défaut, ``default_stop_pct``. La taille
   vient de ``risk.position_size(équité, prix d'exécution, stop)`` où l'équité est celle
   de la clôture précédente : le sizing est donc toujours basé sur le risque (plafonné par
   ``max_position_fraction``).
4. Ré-entrée après une sortie décidée par le moteur (``stop_loss``, ``take_profit``,
   ``daily_limit``) : avec ``reentry_after_stop = "wait"`` (défaut) ce côté est bloqué
   tant que le signal brut n'a pas quitté ce côté au moins une fois (0 ou côté opposé) ;
   une nouvelle transition ouvre à nouveau. ``"immediate"`` = ré-ouverture dès la bougie
   suivante si le signal le demande encore.
5. Le kill switch du :class:`~tradingbot.risk.RiskManager` ferme la position à l'ouverture
   suivante (``exit_reason = "kill_switch"``) et bloque toute réouverture ; la limite de
   perte journalière fait de même (``"daily_limit"``) jusqu'au jour UTC suivant. La
   limite journalière n'a de sens qu'en intrajournalier : si la bougie médiane fait un
   jour ou plus, elle est désactivée (``RiskManager.set_daily_limit_active(False)``,
   ``BacktestResult.daily_limit_active = False``, journalisé en INFO). Chaque
   déclenchement est consigné dans ``BacktestResult.events``.
6. Les bougies de chauffe ne tradent pas.
7. ``equity[t] = cash + position × close[t]`` ; la dernière position est fermée à la
   dernière clôture (``"end"``).
8. Sans ``allow_short`` (config **et** risque), les signaux -1 valent 0.
9. Le buy & hold de référence (``metrics["buy_hold_return"]``) est mesuré sur la période
   réellement tradable par la stratégie (``open[warmup+1]`` → dernière clôture, ou toute
   la série si ``warmup = 0``), net d'un aller-retour de frais et de slippage.

Aucun look-ahead : à la bougie ``t`` seules les données ≤ ``t`` sont utilisées pour
décider, et la décision s'applique à ``t+1``.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from tradingbot import indicators as ind
from tradingbot.backtest import metrics as metrics_mod
from tradingbot.risk import DEFAULT_STOP_ATR_WINDOW, RiskConfig, RiskManager

if TYPE_CHECKING:
    from tradingbot.strategies.base import Strategy

try:  # le chargeur de données peut ne pas être disponible (tests isolés)
    from tradingbot.data.loader import ensure_ohlcv as _ensure_ohlcv
    from tradingbot.data.loader import infer_periods_per_year as _infer_periods_per_year
except ImportError:  # pragma: no cover - dépend de l'état du dépôt
    _ensure_ohlcv = None
    _infer_periods_per_year = None

logger = logging.getLogger(__name__)

__all__ = [
    "TRADE_COLUMNS",
    "BacktestConfig",
    "BacktestResult",
    "Trade",
    "run_backtest",
]

OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")
SIGNAL_COLUMNS: tuple[str, ...] = ("signal", "stop_loss", "take_profit")
EXIT_REASONS: tuple[str, ...] = (
    "signal",
    "stop_loss",
    "take_profit",
    "end",
    "kill_switch",
    "daily_limit",
)
#: politiques de ré-entrée après une sortie décidée par le moteur (cf. règle 2)
REENTRY_POLICIES: tuple[str, ...] = ("wait", "immediate")
#: sorties décidées par le moteur qui bloquent la ré-entrée du même côté (politique "wait")
ENGINE_EXIT_REASONS: frozenset[str] = frozenset({"stop_loss", "take_profit", "daily_limit"})
#: types d'événements de garde-fous consignés dans ``BacktestResult.events``
EVENT_TYPES: tuple[str, ...] = ("kill_switch", "daily_limit")
#: fenêtre (en bougies) de la volatilité réalisée utilisée pour le ciblage de volatilité
VOL_WINDOW = 20
#: nombre de secondes dans une année crypto (marché ouvert 24 h/24, 365 j/an)
SECONDS_PER_YEAR = 365 * 24 * 3600


@dataclass
class BacktestConfig:
    """Paramètres de simulation (coûts et gestion des stops par le moteur)."""

    #: capital de départ, en devise de cotation
    initial_cash: float = 10_000.0
    #: frais par ordre, en fraction du notionnel (0.001 = 0,1 %)
    fee_rate: float = 0.001
    #: glissement défavorable par ordre, en fraction du prix
    slippage: float = 0.0005
    #: stop par défaut (fraction du prix d'entrée) si la stratégie n'en fournit pas
    stop_loss_pct: float | None = None
    #: objectif par défaut (fraction du prix d'entrée) si la stratégie n'en fournit pas
    take_profit_pct: float | None = None
    #: stop suiveur géré par le moteur : distance au meilleur prix atteint
    trailing_stop_pct: float | None = None
    #: autoriser les positions courtes (il faut aussi ``RiskConfig.allow_short``)
    allow_short: bool = False
    #: moment d'exécution ; seule la valeur ``"next_open"`` est supportée
    execute_on: str = "next_open"
    #: après une sortie décidée par le moteur (stop, objectif, limite journalière) :
    #: ``"wait"`` = ne pas ré-ouvrir ce côté tant que le signal ne l'a pas quitté une fois ;
    #: ``"immediate"`` = ré-ouvrir dès la bougie suivante si le signal le demande encore
    reentry_after_stop: str = "wait"


@dataclass
class Trade:
    """Un aller-retour complet (entrée puis sortie)."""

    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: int  # +1 long, -1 short
    entry_price: float  # prix d'exécution, slippage inclus
    exit_price: float  # prix d'exécution, slippage inclus
    size: float  # unités de l'actif
    pnl: float  # gain net (frais déduits), en devise de cotation
    pnl_pct: float  # pnl / (entry_price × size)
    fees: float  # frais totaux des deux ordres
    bars_held: int  # nombre de bougies pendant lesquelles la position a été tenue
    exit_reason: str  # cf. EXIT_REASONS


TRADE_COLUMNS: tuple[str, ...] = tuple(Trade.__dataclass_fields__)


@dataclass
class BacktestResult:
    """Résultat complet d'un backtest."""

    strategy_name: str
    params: dict[str, Any]
    config: BacktestConfig
    equity: pd.Series
    returns: pd.Series
    positions: pd.Series
    trades: pd.DataFrame
    metrics: dict[str, float | int]
    signals: pd.DataFrame
    periods_per_year: float = field(default=365.0)
    #: garde-fous déclenchés : dicts ``{timestamp (ISO), type ("kill_switch" |
    #: "daily_limit"), equity, drawdown (fraction positive), detail (texte FR)}``
    events: list[dict[str, Any]] = field(default_factory=list)
    #: bougies de chauffe de la stratégie (``strategy.warmup_bars()``) ; le buy & hold de
    #: référence démarre à ``open[warmup_bars + 1]`` (ou à la première bougie si 0)
    warmup_bars: int = 0
    #: False si la limite de perte journalière a été désactivée (bougies >= 1 jour)
    daily_limit_active: bool = True

    def guard_rails_text(self) -> str:
        """Résumé en une ligne des garde-fous (vide si rien à signaler)."""
        parts: list[str] = []
        kills = [e for e in self.events if e.get("type") == "kill_switch"]
        if kills:
            parts.append(f"kill switch le {_fmt_ts(kills[0].get('timestamp'))}")
        n_daily = sum(1 for e in self.events if e.get("type") == "daily_limit")
        if n_daily:
            parts.append(f"limite journalière atteinte {n_daily} jour(s)")
        if not self.daily_limit_active:
            parts.append("limite journalière inactive (bougies >= 1 jour)")
        return " ; ".join(parts)

    def summary(self) -> str:
        """Tableau texte (français) des principales métriques."""
        m = self.metrics
        n_bars = len(self.equity)
        if n_bars:
            start = self.equity.index[0]
            end = self.equity.index[-1]
            periode = f"{_fmt_ts(start)} → {_fmt_ts(end)} ({n_bars} bougies)"
        else:
            periode = "aucune bougie"
        rows: list[tuple[str, str]] = [
            ("Stratégie", self.strategy_name),
            ("Paramètres", _fmt_params(self.params)),
            ("Période", periode),
            ("Capital initial", _fmt_money(self.config.initial_cash)),
            ("Capital final", _fmt_money(m.get("final_equity", 0.0))),
            ("Rendement total", _fmt_pct(m.get("total_return", 0.0))),
            ("CAGR", _fmt_pct(m.get("cagr", 0.0))),
            ("Volatilité annualisée", _fmt_pct(m.get("annual_volatility", 0.0), signed=False)),
            ("Sharpe", _fmt_num(m.get("sharpe", 0.0))),
            ("Sortino", _fmt_num(m.get("sortino", 0.0))),
            (
                "Drawdown maximal",
                f"{_fmt_pct(m.get('max_drawdown', 0.0))} "
                f"(durée max {int(m.get('max_drawdown_duration', 0))} bougies)",
            ),
            ("Calmar", _fmt_num(m.get("calmar", 0.0))),
            (
                "Trades",
                f"{int(m.get('n_trades', 0))} "
                f"(taux de réussite {_fmt_pct(m.get('win_rate', 0.0), signed=False)}, "
                f"profit factor {_fmt_num(m.get('profit_factor', 0.0))})",
            ),
            (
                "Trade moyen",
                f"{_fmt_pct(m.get('avg_trade_pct', 0.0))} "
                f"(gain moyen {_fmt_pct(m.get('avg_win_pct', 0.0))}, "
                f"perte moyenne {_fmt_pct(m.get('avg_loss_pct', 0.0))})",
            ),
            (
                "Meilleur / pire trade",
                f"{_fmt_pct(m.get('best_trade_pct', 0.0))} / "
                f"{_fmt_pct(m.get('worst_trade_pct', 0.0))}",
            ),
            (
                "Durée moyenne",
                f"{_fmt_num(m.get('avg_bars_held', 0.0), 1)} bougies, "
                f"exposition {_fmt_pct(m.get('exposure', 0.0), signed=False)}",
            ),
        ]
        if "buy_hold_return" in m:
            rows.append(
                ("Buy & hold", f"{_fmt_pct(m['buy_hold_return'])} (même période, frais inclus)")
            )
        guard = self.guard_rails_text()
        if guard:
            rows.append(("Garde-fous", guard))
        width = max(len(label) for label, _ in rows)
        lines = [f"=== Résumé du backtest : {self.strategy_name} ==="]
        lines.extend(f"{label.ljust(width)} : {value}" for label, value in rows)
        return "\n".join(lines)


# --------------------------------------------------------------------------- formatage
def _fmt_ts(ts: Any) -> str:
    try:
        return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return str(ts)


def _fmt_params(params: dict[str, Any]) -> str:
    if not params:
        return "(défaut)"
    return ", ".join(f"{k}={v}" for k, v in params.items())


def _fmt_money(x: float) -> str:
    return f"{float(x):,.2f}".replace(",", " ")


def _fmt_pct(x: float, signed: bool = True) -> str:
    x = float(x)
    if math.isinf(x):
        return "∞"
    return f"{x * 100:+.2f} %" if signed else f"{x * 100:.2f} %"


def _fmt_num(x: float, digits: int = 2) -> str:
    x = float(x)
    if math.isinf(x):
        return "∞"
    return f"{x:.{digits}f}"


# --------------------------------------------------------------------------- données
def _minimal_ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalisation minimale des bougies quand ``tradingbot.data.loader`` est absent.

    Colonnes en minuscules, index ``DatetimeIndex`` UTC trié sans doublon, valeurs
    ``float64``. Lève ``ValueError`` si une colonne manque ou si un prix est manquant.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Les bougies doivent être un pandas.DataFrame")
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    if not isinstance(out.index, pd.DatetimeIndex):
        if "timestamp" in out.columns:
            out = out.set_index(pd.to_datetime(out["timestamp"], utc=True)).drop(
                columns="timestamp"
            )
        else:
            raise ValueError("L'index des bougies doit être un DatetimeIndex (UTC)")
    missing = [c for c in OHLCV_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"Colonnes OHLCV manquantes : {missing}")
    idx = out.index
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    out.index = idx
    out = out.loc[:, list(OHLCV_COLUMNS)].astype(np.float64)
    if not out.index.is_unique:
        out = out.loc[~out.index.duplicated(keep="last")]
    if not out.index.is_monotonic_increasing:
        out = out.sort_index()
    out.index.name = "timestamp"
    if out[["open", "high", "low", "close"]].isna().to_numpy().any():
        raise ValueError("Les prix OHLC contiennent des valeurs manquantes")
    return out


def _median_bar_delta(index: pd.Index) -> pd.Timedelta | None:
    """Durée médiane d'une bougie (None si moins de deux horodatages)."""
    if len(index) < 2:
        return None
    try:
        diffs = pd.Series(pd.DatetimeIndex(index)).diff().dropna()
    except (TypeError, ValueError, AttributeError):
        return None
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        return None
    return pd.Timedelta(diffs.median())


def _fallback_periods_per_year(index: pd.Index) -> float:
    """Bougies par an d'après l'écart médian de l'index (marché crypto 24/7, 365 jours)."""
    if len(index) < 2:
        return 365.0
    try:
        # Indépendant de l'unité de l'index (ns, us...) : on passe par des Timedelta.
        deltas = pd.Series(pd.DatetimeIndex(index)).diff().dt.total_seconds().to_numpy()
    except (TypeError, ValueError, AttributeError):
        return 365.0
    deltas = deltas[np.isfinite(deltas) & (deltas > 0)]
    if deltas.size == 0:
        return 365.0
    return SECONDS_PER_YEAR / float(np.median(deltas))


def _normalise_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if _ensure_ohlcv is not None:
        return _ensure_ohlcv(df)
    return _minimal_ensure_ohlcv(df)


def _periods_per_year(index: pd.Index) -> float:
    if _infer_periods_per_year is not None:
        try:
            return float(_infer_periods_per_year(index))
        except (TypeError, ValueError):
            logger.debug("infer_periods_per_year a échoué, repli sur l'estimation locale")
    return _fallback_periods_per_year(index)


def _fallback_validate_signals(df: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    """Copie de ``Strategy.validate_signals`` pour les stratégies « duck-typées »."""
    missing = [c for c in SIGNAL_COLUMNS if c not in signals.columns]
    if missing:
        raise ValueError(f"Colonnes de signaux manquantes : {missing}")
    if not signals.index.equals(df.index):
        raise ValueError("L'index des signaux doit être identique à celui des bougies")
    out = signals.loc[:, list(SIGNAL_COLUMNS)].copy()
    sig = out["signal"].fillna(0).astype(np.int64)
    if not sig.isin([-1, 0, 1]).all():
        raise ValueError("`signal` doit valoir -1, 0 ou 1")
    out["signal"] = sig
    out["stop_loss"] = out["stop_loss"].astype(float)
    out["take_profit"] = out["take_profit"].astype(float)
    return out


def _validate_signals(df: pd.DataFrame, signals: Any, strategy: Any) -> pd.DataFrame:
    if not isinstance(signals, pd.DataFrame):
        raise TypeError("generate_signals doit renvoyer un pandas.DataFrame")
    validator = getattr(strategy, "validate_signals", None)
    if callable(validator):
        return validator(df, signals)
    return _fallback_validate_signals(df, signals)


def _realized_vol(close: np.ndarray, periods_per_year: float) -> list[float]:
    """Volatilité annualisée des rendements log sur ``VOL_WINDOW`` bougies passées (NaN
    pendant la chauffe). ``vol[t]`` n'utilise que les clôtures ≤ ``t``."""
    s = pd.Series(close, dtype=np.float64)
    log_ret = np.log(s).diff()
    vol = log_ret.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std(ddof=1) * math.sqrt(
        periods_per_year
    )
    return vol.to_numpy(dtype=np.float64).tolist()


def _pct_or_none(value: float | None, name: str) -> float | None:
    """0 / None → None ; négatif ou NaN → ValueError."""
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or value < 0:
        raise ValueError(f"{name} doit être une fraction positive ou None")
    return value if value > 0 else None


def _check_config(config: BacktestConfig) -> None:
    if config.execute_on != "next_open":
        raise ValueError(
            f"execute_on={config.execute_on!r} non supporté : seule la valeur 'next_open' "
            "(exécution à l'ouverture de la bougie suivante) est implémentée"
        )
    if not math.isfinite(config.initial_cash) or config.initial_cash <= 0:
        raise ValueError("initial_cash doit être strictement positif")
    if config.fee_rate < 0 or config.slippage < 0:
        raise ValueError("fee_rate et slippage doivent être positifs ou nuls")
    if config.reentry_after_stop not in REENTRY_POLICIES:
        raise ValueError(
            f"reentry_after_stop={config.reentry_after_stop!r} invalide : valeurs possibles "
            + ", ".join(repr(p) for p in REENTRY_POLICIES)
        )


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entry_time": pd.Series(dtype="datetime64[ns, UTC]"),
            "exit_time": pd.Series(dtype="datetime64[ns, UTC]"),
            "side": pd.Series(dtype=np.int64),
            "entry_price": pd.Series(dtype=np.float64),
            "exit_price": pd.Series(dtype=np.float64),
            "size": pd.Series(dtype=np.float64),
            "pnl": pd.Series(dtype=np.float64),
            "pnl_pct": pd.Series(dtype=np.float64),
            "fees": pd.Series(dtype=np.float64),
            "bars_held": pd.Series(dtype=np.int64),
            "exit_reason": pd.Series(dtype=object),
        }
    )


def _trades_frame(trades: list[Trade]) -> pd.DataFrame:
    if not trades:
        return _empty_trades()
    frame = pd.DataFrame([asdict(t) for t in trades], columns=list(TRADE_COLUMNS))
    return frame.astype({"side": np.int64, "bars_held": np.int64})


# --------------------------------------------------------------------------- moteur
def run_backtest(
    df: pd.DataFrame,
    strategy: Strategy,
    config: BacktestConfig | None = None,
    risk: RiskManager | None = None,
) -> BacktestResult:
    """Simule ``strategy`` sur les bougies ``df`` et renvoie un :class:`BacktestResult`.

    :param df: bougies OHLCV (normalisées via ``ensure_ohlcv`` si disponible).
    :param strategy: instance de stratégie (``generate_signals``, ``warmup_bars``).
    :param config: coûts et stops par défaut ; :class:`BacktestConfig` par défaut.
    :param risk: gestionnaire de risque ; ``RiskManager(RiskConfig())`` par défaut. Son
        état est remis à zéro au début de la simulation.
    :raises ValueError: données ou configuration invalides.
    """
    t_start = time.perf_counter()
    cfg = config if config is not None else BacktestConfig()
    _check_config(cfg)
    rm = risk if risk is not None else RiskManager(RiskConfig())
    if hasattr(rm, "reset"):
        rm.reset()

    data = _normalise_ohlcv(df)
    n = len(data)
    if n == 0:
        raise ValueError("Aucune bougie à simuler")

    raw_signals = strategy.generate_signals(data)
    signals = _validate_signals(data, raw_signals, strategy)
    warmup = int(getattr(strategy, "warmup_bars", lambda: 0)())
    warmup = max(0, warmup)

    risk_cfg = getattr(rm, "config", RiskConfig())
    allow_short = bool(cfg.allow_short) and bool(getattr(risk_cfg, "allow_short", False))
    min_notional = float(getattr(risk_cfg, "min_trade_notional", 0.0) or 0.0)
    vol_target = getattr(risk_cfg, "vol_target", None)
    protective_stop = getattr(rm, "protective_stop", None)
    if not callable(protective_stop):  # gestionnaire « duck-typé » sans cette méthode
        protective_stop = RiskManager(RiskConfig()).protective_stop
    wait_after_exit = cfg.reentry_after_stop == "wait"

    # Limite journalière : seulement en intrajournalier (règle 5).
    bar_delta = _median_bar_delta(data.index)
    daily_limit_active = not (bar_delta is not None and bar_delta >= pd.Timedelta(days=1))
    setter = getattr(rm, "set_daily_limit_active", None)
    if callable(setter):
        setter(daily_limit_active, "" if daily_limit_active else "bougies >= 1 jour")
    if not daily_limit_active and getattr(risk_cfg, "daily_loss_limit", None):
        logger.info(
            "limite de perte journalière désactivée : bougies >= 1 jour (bougie médiane %s)",
            bar_delta,
        )

    stop_pct = _pct_or_none(cfg.stop_loss_pct, "stop_loss_pct")
    take_pct = _pct_or_none(cfg.take_profit_pct, "take_profit_pct")
    trail_pct = _pct_or_none(cfg.trailing_stop_pct, "trailing_stop_pct")
    fee_rate = float(cfg.fee_rate)
    slip = float(cfg.slippage)

    close_arr = data["close"].to_numpy(dtype=np.float64)
    o = data["open"].to_numpy(dtype=np.float64).tolist()
    h = data["high"].to_numpy(dtype=np.float64).tolist()
    low = data["low"].to_numpy(dtype=np.float64).tolist()
    c = close_arr.tolist()
    sig = signals["signal"].to_numpy(dtype=np.int64).tolist()
    sl = signals["stop_loss"].to_numpy(dtype=np.float64).tolist()
    tp = signals["take_profit"].to_numpy(dtype=np.float64).tolist()
    index = data.index
    timestamps = list(index.to_pydatetime())
    ppy = _periods_per_year(index)
    vols = _realized_vol(close_arr, ppy) if vol_target else None
    # ATR(14) du stop de secours ; atr_arr[t-1] n'utilise que les bougies <= t-1.
    atr_arr = ind.atr(data, DEFAULT_STOP_ATR_WINDOW).to_numpy(dtype=np.float64).tolist()

    equity = [0.0] * n
    positions = [0] * n
    trades: list[Trade] = []
    nan = math.nan

    cash = float(cfg.initial_cash)
    pos = 0  # -1 / 0 / +1
    size = 0.0
    entry_price = 0.0
    entry_idx = -1
    trade_fees = 0.0
    bars_held = 0
    stop = nan
    take = nan
    extreme = nan  # meilleur prix atteint (stop suiveur)
    next_pos = 0  # position décidée à la fin de la bougie précédente
    pending_reason = "signal"
    prev_equity = cash
    n_refused_log = 0
    blocked_side = 0  # côté bloqué après une sortie moteur (politique "wait")
    events: list[dict[str, Any]] = []
    kill_switch_seen = False
    last_daily_event_day = None
    n_daily_hits = 0

    def _close(exit_raw: float, t: int, reason: str) -> None:
        nonlocal cash, pos, size, trade_fees, stop, take, extreme, blocked_side
        exit_exec = exit_raw * (1.0 - slip) if pos == 1 else exit_raw * (1.0 + slip)
        fee = size * exit_exec * fee_rate
        cash += pos * size * exit_exec - fee
        trade_fees += fee
        notional = entry_price * size
        pnl = pos * (exit_exec - entry_price) * size - trade_fees
        trades.append(
            Trade(
                entry_time=index[entry_idx],
                exit_time=index[t],
                side=pos,
                entry_price=entry_price,
                exit_price=exit_exec,
                size=size,
                pnl=pnl,
                pnl_pct=pnl / notional if notional else 0.0,
                fees=trade_fees,
                bars_held=bars_held,
                exit_reason=reason,
            )
        )
        if wait_after_exit and reason in ENGINE_EXIT_REASONS:
            blocked_side = pos
        pos = 0
        size = 0.0
        trade_fees = 0.0
        stop = take = extreme = nan

    for t in range(n):
        op = o[t]
        hi = h[t]
        lo = low[t]
        cl = c[t]

        # 1. ordres décidés à la clôture précédente, exécutés à l'ouverture
        if next_pos != pos:
            if pos != 0:
                _close(op, t, pending_reason)
            if next_pos != 0:
                side = next_pos
                exec_price = op * (1.0 + slip) if side == 1 else op * (1.0 - slip)
                strat_stop = sl[t - 1] if t > 0 else nan
                atr_prev = atr_arr[t - 1] if t > 0 else nan
                realized = vols[t - 1] if (vols is not None and t > 0) else None
                if realized is not None and realized != realized:
                    realized = None
                if exec_price > 0 and math.isfinite(exec_price):
                    # Jamais d'entrée sans stop (règle 3) : le sizing est toujours risk-based.
                    stop_price = float(
                        protective_stop(side, exec_price, strat_stop, atr_prev, stop_pct)
                    )
                    qty = float(rm.position_size(prev_equity, exec_price, stop_price, realized))
                else:
                    stop_price = nan
                    qty = 0.0
                if side == 1 and qty > 0:
                    affordable = cash / (exec_price * (1.0 + fee_rate))
                    if qty > affordable:
                        qty = max(affordable, 0.0)
                if qty > 0 and qty * exec_price < min_notional:
                    qty = 0.0
                if qty > 0:
                    fee = qty * exec_price * fee_rate
                    cash -= side * qty * exec_price + fee
                    pos = side
                    size = qty
                    entry_price = exec_price
                    entry_idx = t
                    trade_fees = fee
                    bars_held = 0
                    stop = stop_price
                    strat_take = tp[t - 1] if t > 0 else nan
                    if strat_take == strat_take:
                        take = strat_take
                    elif take_pct is not None:
                        take = exec_price * (1.0 + side * take_pct)
                    else:
                        take = nan
                    extreme = exec_price
                    if trail_pct is not None:
                        cand = exec_price * (1.0 - side * trail_pct)
                        if (
                            stop != stop
                            or (side == 1 and cand > stop)
                            or (side == -1 and cand < stop)
                        ):
                            stop = cand
                else:
                    n_refused_log += 1
                    if n_refused_log <= 3:
                        logger.debug(
                            "Entrée refusée à %s : taille nulle (équité %.2f, prix %.4f)",
                            timestamps[t].isoformat(),
                            prev_equity,
                            exec_price,
                        )

        positions[t] = pos

        # 2. stop / take-profit sur la bougie courante
        if pos != 0:
            bars_held += 1
            reason: str | None = None
            exit_raw = 0.0
            has_stop = stop == stop
            has_take = take == take
            # Gaps d'ouverture d'abord (l'ouverture est le premier prix connu), puis
            # touches intrabar, stop avant take-profit (pessimiste).
            if pos == 1:
                if has_stop and op <= stop:
                    exit_raw, reason = op, "stop_loss"
                elif has_take and op >= take:
                    exit_raw, reason = op, "take_profit"
                elif has_stop and lo <= stop:
                    exit_raw, reason = stop, "stop_loss"
                elif has_take and hi >= take:
                    exit_raw, reason = take, "take_profit"
            else:
                if has_stop and op >= stop:
                    exit_raw, reason = op, "stop_loss"
                elif has_take and op <= take:
                    exit_raw, reason = op, "take_profit"
                elif has_stop and hi >= stop:
                    exit_raw, reason = stop, "stop_loss"
                elif has_take and lo <= take:
                    exit_raw, reason = take, "take_profit"
            if reason is not None:
                _close(exit_raw, t, reason)

        # 3. équité à la clôture
        eq = cash + pos * size * cl
        equity[t] = eq

        # 4. suivi du risque
        ts = timestamps[t]
        rm.update(ts, eq)

        # 5. décision pour la bougie suivante
        if t < warmup:
            desired = 0
        else:
            desired = sig[t]
            if desired == -1 and not allow_short:
                desired = 0
        if blocked_side != 0:
            if sig[t] != blocked_side:
                blocked_side = 0  # le signal brut a quitté ce côté : ré-entrée possible
            elif desired == blocked_side:
                desired = 0
        reason_next = "signal"
        if rm.kill_switch_active():
            if not kill_switch_seen:
                kill_switch_seen = True
                events.append(_guard_event("kill_switch", ts, eq, rm, risk_cfg))
            desired = 0
            reason_next = "kill_switch"
        elif (pos != 0 or desired != 0) and rm.daily_limit_hit(ts, eq):
            if ts.date() != last_daily_event_day:
                last_daily_event_day = ts.date()
                n_daily_hits += 1
                events.append(_guard_event("daily_limit", ts, eq, rm, risk_cfg))
            desired = 0
            reason_next = "daily_limit"
        elif desired != 0 and desired != pos:
            allowed, why = rm.can_open(desired, ts, eq)
            if not allowed:
                logger.debug("Ouverture refusée à %s : %s", ts.isoformat(), why)
                desired = 0
        next_pos = desired
        pending_reason = reason_next

        # 6. mise à jour des niveaux actifs (stratégie puis stop suiveur). Un stop de
        #    stratégie du mauvais côté de la clôture (ex. stop d'entrée porté par la
        #    stratégie alors que le moteur a dû le remplacer) n'est pas un stop : ignoré.
        if pos != 0:
            s_t = sl[t]
            if s_t == s_t and ((pos == 1 and s_t < cl) or (pos == -1 and s_t > cl)):
                stop = s_t
            k_t = tp[t]
            if k_t == k_t:
                take = k_t
            if trail_pct is not None:
                if pos == 1:
                    if hi > extreme:
                        extreme = hi
                    cand = extreme * (1.0 - trail_pct)
                    if stop != stop or cand > stop:
                        stop = cand
                else:
                    if lo < extreme:
                        extreme = lo
                    cand = extreme * (1.0 + trail_pct)
                    if stop != stop or cand < stop:
                        stop = cand

        prev_equity = eq

    # 7. clôture finale
    if pos != 0:
        _close(c[n - 1], n - 1, "end")
        equity[n - 1] = cash

    equity_s = pd.Series(equity, index=index, name="equity", dtype=np.float64)
    returns_s = equity_s.pct_change().fillna(0.0)
    returns_s.name = "returns"
    positions_s = pd.Series(positions, index=index, name="position", dtype=np.int64)
    trades_df = _trades_frame(trades)
    # Buy & hold de référence sur la période réellement tradable, frais inclus (règle 8).
    first_tradable = min(warmup + 1, n - 1) if warmup > 0 else 0
    metrics = metrics_mod.compute_metrics(
        equity_s,
        trades_df,
        ppy,
        benchmark=data["close"].iloc[first_tradable:],
        positions=positions_s,
    )
    metrics["buy_hold_return"] = _buy_hold_return(o[first_tradable], c[n - 1], fee_rate, slip)
    metrics["kill_switch_triggered"] = int(kill_switch_seen)
    metrics["n_daily_limit_hits"] = int(n_daily_hits)
    strategy_name = str(getattr(strategy, "name", type(strategy).__name__))
    params = dict(getattr(strategy, "params", {}) or {})
    logger.info(
        "Backtest %s : %d bougies, %d trades, rendement %.2f %%, %.2f s",
        strategy_name,
        n,
        len(trades),
        metrics["total_return"] * 100,
        time.perf_counter() - t_start,
    )
    return BacktestResult(
        strategy_name=strategy_name,
        params=params,
        config=cfg,
        equity=equity_s,
        returns=returns_s,
        positions=positions_s,
        trades=trades_df,
        metrics=metrics,
        signals=signals,
        periods_per_year=ppy,
        events=events,
        warmup_bars=warmup,
        daily_limit_active=daily_limit_active,
    )


def _buy_hold_return(entry_open: float, last_close: float, fee_rate: float, slip: float) -> float:
    """Rendement d'un achat à ``entry_open`` revendu à ``last_close``, net d'un aller-retour
    de frais et de slippage (``(1+slip)(1+fee)`` à l'achat, ``(1-slip)(1-fee)`` à la vente)."""
    cost = entry_open * (1.0 + slip) * (1.0 + fee_rate)
    if not (cost > 0) or not math.isfinite(cost) or not math.isfinite(last_close):
        return 0.0
    return float(last_close * (1.0 - slip) * (1.0 - fee_rate) / cost - 1.0)


def _guard_event(kind: str, ts: Any, equity: float, rm: Any, risk_cfg: Any) -> dict[str, Any]:
    """Événement de garde-fou (JSON-safe) pour ``BacktestResult.events``."""
    drawdown = 0.0
    try:
        drawdown = float(rm.drawdown())
    except Exception:  # noqa: BLE001 - gestionnaire duck-typé
        drawdown = 0.0
    if kind == "kill_switch":
        max_dd = getattr(risk_cfg, "max_drawdown", None) or 0.0
        detail = (
            f"drawdown {drawdown * 100:.1f} % > {max_dd * 100:.0f} % : position fermée à "
            "l'ouverture suivante, plus aucune ouverture jusqu'à la fin du backtest"
        )
    else:
        limit = getattr(risk_cfg, "daily_loss_limit", None) or 0.0
        day_start = getattr(rm, "day_start_equity", None)
        loss = 0.0
        if day_start is not None and day_start > 0:
            loss = max(0.0, 1.0 - float(equity) / float(day_start))
        detail = (
            f"perte journalière {loss * 100:.1f} % > {limit * 100:.0f} % : position fermée à "
            "l'ouverture suivante, plus d'ouverture jusqu'au jour UTC suivant"
        )
    return {
        "timestamp": ts.isoformat(),
        "type": kind,
        "equity": float(equity),
        "drawdown": drawdown,
        "detail": detail,
    }
