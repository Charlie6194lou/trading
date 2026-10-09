"""Interface en ligne de commande du robot : ``python -m tradingbot <commande>``.

Commandes (toutes les sorties sont en français) :

- ``list-strategies`` : nom et description de chaque stratégie disponible ;
- ``demo`` : backtest de toutes les stratégies sur des données synthétiques, sans réseau ;
- ``download`` : télécharge des bougies et les met en cache (``data/cache``) ;
- ``backtest`` : backtest d'une stratégie + rapport détaillé ;
- ``compare`` : toutes les stratégies (paramètres par défaut) contre le buy & hold ;
- ``optimize`` : validation walk-forward d'une stratégie (recherche sur grille) ;
- ``paper`` : paper trading (argent fictif) en temps réel sur données publiques ;
- ``live`` : trading réel, verrouillé par trois sécurités (``mode = "live"`` dans
  ``config.toml``, le drapeau ``--i-understand-the-risks`` et les clés API dans ``.env``),
  avec récapitulatif et confirmation ``OUI``.

Codes de sortie : ``0`` succès, ``1`` erreur, ``2`` refus de sécurité ou arguments
invalides. C'est le seul module autorisé à utiliser ``print``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from tradingbot import __version__
from tradingbot.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from tradingbot.backtest.report import format_comparison, format_report, save_report
from tradingbot.backtest.walkforward import WalkForwardResult, default_windows, walk_forward
from tradingbot.config import Settings, load_settings, parse_param_overrides
from tradingbot.data.loader import cache_path, fetch_ohlcv, load_csv, load_or_fetch, save_csv
from tradingbot.data.synthetic import generate_ohlcv
from tradingbot.execution.broker import Broker, BrokerError
from tradingbot.execution.ccxt_broker import CcxtBroker
from tradingbot.execution.paper import PaperBroker
from tradingbot.live.notifier import LogNotifier, MultiNotifier, Notifier, TelegramNotifier
from tradingbot.live.runner import LiveTrader
from tradingbot.risk import RiskConfig, RiskManager
from tradingbot.strategies import STRATEGIES, Strategy, get_strategy, list_strategies

logger = logging.getLogger(__name__)

__all__ = ["DEMO_WARNING", "CliError", "build_public_exchange", "main"]

#: code de sortie d'un refus de sécurité (commande ``live``) ou d'arguments invalides
EXIT_REFUSED = 2
#: nombre de bougies synthétiques de la démo
DEMO_BARS = 3000
#: graine fixe de la démo : le même résultat à chaque exécution
DEMO_SEED = 42
#: répertoire de cache des bougies téléchargées
DEFAULT_CACHE_DIR = "data/cache"
#: nombre de bougies demandées à l'exchange à chaque itération en paper / live
DEFAULT_LOOKBACK = 500
#: marge de bougies au-delà de la période de chauffe pour le paper / live
LOOKBACK_MARGIN = 50
#: phrase affichée par la démo : les données synthétiques ne prouvent rien
DEMO_WARNING = (
    "ATTENTION : ces bougies sont SYNTHÉTIQUES (tirées au hasard par le programme). Elles "
    "servent uniquement à vérifier que le robot fonctionne. Un gain ou une perte sur ces "
    "données ne prouve RIEN sur les profits futurs d'une stratégie sur un vrai marché."
)
#: avertissement affiché avant toute connexion réelle
LIVE_BANNER = "*** MODE LIVE : ARGENT RÉEL ***"
#: niveaux de journalisation acceptés par ``--log-level``
LOG_LEVELS = ("debug", "info", "warning", "error")


class CliError(Exception):
    """Erreur d'utilisation de la ligne de commande (message destiné à l'utilisateur)."""


# ============================================================================ utilitaires
def _print(text: str = "") -> None:
    print(text)


def _eprint(text: str) -> None:
    print(text, file=sys.stderr)


def _safe_name(text: str) -> str:
    """``"BTC/USDT:USDT"`` → ``"BTC-USDT-USDT"`` : fragment sûr pour un nom de fichier."""
    out = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in str(text).strip())
    return out.strip("-") or "unknown"


class _DedupFilter(logging.Filter):
    """N'affiche qu'une fois chaque message identique pendant une commande hors ligne.

    Un ``compare`` ou un walk-forward enchaîne des dizaines de backtests : le même
    avertissement (ex. « matplotlib n'est pas installé ») n'a pas besoin d'être répété.
    Désactivé pour ``paper`` / ``live``, où une répétition est une information utile.
    """

    def __init__(self) -> None:
        super().__init__()
        self.enabled = False
        self._seen: set[tuple[str, int, str]] = set()

    def reset(self, enabled: bool) -> None:
        self.enabled = enabled
        self._seen.clear()

    def filter(self, record: logging.LogRecord) -> bool:
        if not self.enabled:
            return True
        key = (record.name, record.levelno, record.getMessage())
        if key in self._seen:
            return False
        self._seen.add(key)
        return True


def _configure_logging(level_name: str | None, default: str, dedup: bool = False) -> None:
    """Journalisation du paquet ``tradingbot`` sur la sortie d'erreur.

    ``dedup`` : supprime les répétitions d'un même message (commandes hors ligne).
    """
    name = (level_name or default).lower()
    level = getattr(logging, name.upper(), logging.INFO)
    pkg_logger = logging.getLogger("tradingbot")
    pkg_logger.setLevel(level)
    handler = next((h for h in pkg_logger.handlers if getattr(h, "_tradingbot_cli", False)), None)
    if handler is None:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s : %(message)s"))
        handler._tradingbot_cli = True  # type: ignore[attr-defined]
        handler.addFilter(_DedupFilter())
        pkg_logger.addHandler(handler)
    for flt in handler.filters:
        if isinstance(flt, _DedupFilter):
            flt.reset(dedup)


def build_public_exchange(exchange_id: str) -> Any:
    """Instancie un exchange ccxt **public** (sans clé API, limitation de débit activée).

    Remplaçable dans les tests (``monkeypatch.setattr(tradingbot.cli, "build_public_exchange", ...)``).
    """
    import ccxt

    cls = getattr(ccxt, exchange_id, None)
    if cls is None or not isinstance(cls, type):
        raise CliError(
            f"Exchange inconnu pour ccxt : {exchange_id!r}. Exemples valides : binance, kraken, "
            "coinbase, bybit, okx."
        )
    return cls({"enableRateLimit": True})


def _ticker_price(exchange: Any, symbol: str) -> float:
    """Dernier prix d'un ticker ccxt (``last``, sinon ``close``) ; ``BrokerError`` sinon."""
    try:
        ticker = exchange.fetch_ticker(symbol)
    except BrokerError:
        raise
    except Exception as exc:  # noqa: BLE001 - ccxt peut lever n'importe quoi
        raise BrokerError(f"Ticker indisponible pour {symbol} : {exc}") from exc
    ticker = ticker if isinstance(ticker, dict) else {}
    for key in ("last", "close"):
        value = ticker.get(key)
        if value is not None:
            try:
                price = float(value)
            except (TypeError, ValueError):
                continue
            if price > 0:
                return price
    raise BrokerError(
        f"Prix indisponible pour {symbol} : le ticker ne contient ni 'last' ni 'close'."
    )


def _load_config(args: argparse.Namespace) -> Settings:
    """Charge ``config.toml`` (ou les défauts s'il n'existe pas)."""
    path = getattr(args, "config", None) or "config.toml"
    return load_settings(path)


def _strategy_params(args: argparse.Namespace, settings: Settings, name: str) -> dict[str, Any]:
    """Paramètres de la stratégie : ``[strategy.params]`` de la config (si c'est la même
    stratégie) puis les ``--param k=v`` de la ligne de commande."""
    params: dict[str, Any] = {}
    if name == settings.strategy.name:
        params.update(settings.strategy.params)
    params.update(parse_param_overrides(getattr(args, "param", None)))
    return params


def _make_strategy(name: str, params: dict[str, Any]) -> Strategy:
    try:
        return get_strategy(name, **params)
    except KeyError as exc:
        raise CliError(str(exc.args[0]) if exc.args else str(exc)) from None


def _strategy_class(name: str) -> type[Strategy]:
    try:
        return STRATEGIES[name]
    except KeyError:
        raise CliError(
            f"Stratégie inconnue : {name!r}. Disponibles : {', '.join(sorted(STRATEGIES))}"
        ) from None


def _backtest_config(args: argparse.Namespace, settings: Settings) -> BacktestConfig:
    """``BacktestConfig`` construite depuis la config et les surcharges de la ligne de commande."""
    initial_cash = getattr(args, "initial_cash", None)
    fee = getattr(args, "fee", None)
    slippage = getattr(args, "slippage", None)
    return BacktestConfig(
        initial_cash=float(initial_cash) if initial_cash is not None else settings.initial_cash,
        fee_rate=float(fee) if fee is not None else settings.costs.fee_rate,
        slippage=float(slippage) if slippage is not None else settings.costs.slippage,
        allow_short=bool(getattr(args, "allow_short", False)) or settings.risk.allow_short,
    )


def _risk_manager(args: argparse.Namespace, settings: Settings) -> RiskManager:
    cfg: RiskConfig = settings.risk
    if getattr(args, "allow_short", False) and not cfg.allow_short:
        cfg = dataclasses.replace(cfg, allow_short=True)
    return RiskManager(cfg)


def _data_description(args: argparse.Namespace, settings: Settings) -> tuple[str, str, str]:
    exchange = (getattr(args, "exchange", None) or settings.general.exchange).strip().lower()
    symbol = (getattr(args, "symbol", None) or settings.general.symbol).strip()
    timeframe = (getattr(args, "timeframe", None) or settings.general.timeframe).strip()
    return exchange, symbol, timeframe


def _load_data(args: argparse.Namespace, settings: Settings) -> pd.DataFrame:
    """Bougies depuis ``--data`` (CSV) ou depuis l'exchange (cache ``data/cache``)."""
    data_path = getattr(args, "data", None)
    if data_path:
        df = load_csv(data_path)
        _print(f"Données : {len(df)} bougies lues dans {data_path} ({_period(df)})")
        return df
    exchange_id, symbol, timeframe = _data_description(args, settings)
    cache_dir = getattr(args, "cache_dir", None) or DEFAULT_CACHE_DIR
    _print(
        f"Données : {symbol} {timeframe} sur {exchange_id} "
        f"(cache {cache_dir}, depuis {getattr(args, 'since', None) or 'les bougies les plus récentes'})"
    )
    df = load_or_fetch(
        exchange_id,
        symbol,
        timeframe,
        since=getattr(args, "since", None),
        cache_dir=cache_dir,
        refresh=bool(getattr(args, "refresh", False)),
        exchange=build_public_exchange(exchange_id),
    )
    if len(df) == 0:
        raise CliError(
            f"Aucune bougie reçue pour {symbol} {timeframe} sur {exchange_id} : vérifiez le "
            "symbole, le timeframe et votre connexion."
        )
    _print(f"          {len(df)} bougies ({_period(df)})")
    return df


def _period(df: pd.DataFrame) -> str:
    if len(df) == 0:
        return "aucune bougie"
    fmt = "%Y-%m-%d %H:%M"
    return f"{df.index[0].strftime(fmt)} → {df.index[-1].strftime(fmt)}"


def _run_all_strategies(
    df: pd.DataFrame,
    config: BacktestConfig,
    risk: RiskManager,
    names: Sequence[str] | None = None,
) -> list[BacktestResult]:
    """Backteste chaque stratégie (paramètres par défaut) sur ``df``."""
    selected = list(names) if names else sorted(STRATEGIES)
    results: list[BacktestResult] = []
    for name in selected:
        strategy = _strategy_class(name)()
        results.append(run_backtest(df, strategy, config=config, risk=risk))
        m = results[-1].metrics
        logger.info(
            "%s : rendement %+.2f %%, Sharpe %.2f, %d trades",
            name,
            m["total_return"] * 100,
            m["sharpe"],
            m["n_trades"],
        )
    return results


def _save_comparison(results: list[BacktestResult], table: str, out_dir: str | Path) -> list[Path]:
    """Écrit le tableau comparatif et un rapport complet par stratégie dans ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    comparison = out / "comparison.txt"
    comparison.write_text(table + "\n", encoding="utf-8")
    paths.append(comparison)
    for result in results:
        paths.extend(save_report(result, out / _safe_name(result.strategy_name)))
    return paths


def _print_paths(paths: Sequence[Path]) -> None:
    _print("Fichiers écrits :")
    for path in paths:
        _print(f"  - {path}")


def _save_walkforward(result: WalkForwardResult, out_dir: str | Path) -> list[Path]:
    """Écrit le compte rendu walk-forward, les fenêtres, l'équité et les trades OOS."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = out / "walkforward.txt"
    summary.write_text(result.summary() + "\n", encoding="utf-8")
    windows = out / "windows.csv"
    result.windows_frame().to_csv(windows, index=False)
    equity = out / "oos_equity.csv"
    frame = pd.DataFrame(
        {
            "equity": result.oos_equity,
            "returns": result.oos_returns,
            "position": result.oos_positions,
        }
    )
    frame.index.name = "timestamp"
    frame.to_csv(equity, index=True)
    trades = out / "oos_trades.csv"
    result.oos_trades.to_csv(trades, index=False)
    return [summary, windows, equity, trades]


def _build_notifier(settings: Settings) -> Notifier:
    """Journal + Telegram si ``[notifications] telegram = true`` et jeton/chat renseignés."""
    notifiers: list[Notifier] = [LogNotifier()]
    if settings.notifications.telegram and settings.has_telegram:
        notifiers.append(
            TelegramNotifier(str(settings.telegram_token), str(settings.telegram_chat_id))
        )
        logger.info("Notifications Telegram activées (chat %s)", settings.telegram_chat_id)
    elif settings.notifications.telegram:
        logger.warning(
            "Telegram demandé dans la configuration mais TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID "
            "manquent dans .env : notifications désactivées"
        )
    return MultiNotifier(notifiers) if len(notifiers) > 1 else notifiers[0]


def _make_data_source(
    exchange: Any, exchange_id: str, symbol: str, timeframe: str
) -> Callable[[int], pd.DataFrame]:
    """``n_bars -> bougies clôturées`` les plus récentes, via ``fetch_ohlcv`` (exchange injecté).

    On demande ``n_bars + 1`` bougies car la dernière, non clôturée, est retirée.
    """

    def data_source(n_bars: int) -> pd.DataFrame:
        return fetch_ohlcv(exchange_id, symbol, timeframe, limit=int(n_bars) + 1, exchange=exchange)

    return data_source


def _trading_lookback(args: argparse.Namespace, strategy: Strategy) -> int:
    """``--lookback`` relevé si nécessaire au-dessus de la période de chauffe."""
    lookback = int(getattr(args, "lookback", None) or DEFAULT_LOOKBACK)
    minimum = int(strategy.warmup_bars()) + LOOKBACK_MARGIN
    if lookback < minimum:
        logger.info(
            "lookback %d trop court pour la chauffe de %s (%d bougies) : relevé à %d",
            lookback,
            strategy.name,
            strategy.warmup_bars(),
            minimum,
        )
        lookback = minimum
    return lookback


def _run_trader(trader: LiveTrader, args: argparse.Namespace) -> int:
    """``--once`` : une itération et affichage de l'état ; sinon la boucle."""
    if getattr(args, "once", False):
        state = trader.step()
        _print(json.dumps(state, indent=2, ensure_ascii=False, default=str))
        return 0
    max_iterations = getattr(args, "max_iterations", None)
    poll = getattr(args, "poll", None)
    _print("Boucle démarrée (Ctrl-C pour arrêter proprement ; la position reste mémorisée).")
    trader.run(poll_seconds=poll, max_iterations=max_iterations)
    return 0


def _fmt_fraction(value: float | None) -> str:
    if not value:
        return "désactivé"
    return f"{float(value) * 100:.2f} %"


def _recap_lines(
    settings: Settings,
    exchange_id: str,
    symbol: str,
    timeframe: str,
    strategy: Strategy,
    risk: RiskConfig,
    state_dir: Path,
    sandbox: bool,
) -> list[str]:
    """Récapitulatif affiché avant tout trading (paper ou live)."""
    mode = "SANDBOX / TESTNET (argent fictif de l'exchange)" if sandbox else "ARGENT RÉEL"
    rows = [
        ("Exchange", exchange_id),
        ("Symbole", symbol),
        ("Timeframe", timeframe),
        (
            "Stratégie",
            f"{strategy.name} ({', '.join(f'{k}={v}' for k, v in strategy.params.items())})",
        ),
        ("Capital de référence", f"{settings.initial_cash:,.2f}".replace(",", " ")),
        ("Risque par trade", _fmt_fraction(risk.risk_per_trade)),
        ("Position maximale", _fmt_fraction(risk.max_position_fraction) + " du capital"),
        ("Kill switch (drawdown)", _fmt_fraction(risk.max_drawdown)),
        ("Limite de perte journalière", _fmt_fraction(risk.daily_loss_limit)),
        ("Vente à découvert", "autorisée" if risk.allow_short else "désactivée"),
        ("Notionnel minimal", f"{risk.min_trade_notional:g}"),
        ("Fichiers d'état", str(state_dir)),
        ("Mode", mode),
    ]
    width = max(len(label) for label, _ in rows)
    return [f"{label.ljust(width)} : {value}" for label, value in rows]


# ============================================================================ commandes
def cmd_list_strategies(args: argparse.Namespace) -> int:
    """``list-strategies`` : nom, description et paramètres par défaut de chaque stratégie."""
    _print("Stratégies disponibles :")
    _print()
    for name, description in list_strategies():
        cls = STRATEGIES[name]
        params = ", ".join(f"{k}={v}" for k, v in cls.default_params.items())
        _print(f"  {name}")
        _print(f"      {description}")
        _print(f"      Paramètres par défaut : {params}")
        _print()
    _print("Essayez : tradingbot backtest --strategy <nom> --data fichier.csv")
    _print("Détails : docs/STRATEGIES.md")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """``demo`` : toutes les stratégies sur des bougies synthétiques (aucun réseau)."""
    settings = _load_config(args)
    bars = int(args.bars)
    if bars < 300:
        raise CliError("--bars doit valoir au moins 300 (période de chauffe des stratégies).")
    df = generate_ohlcv(n=bars, seed=int(args.seed), freq="1h")
    _print(f"Démo : {bars} bougies horaires synthétiques (graine {args.seed}), {_period(df)}")
    config = _backtest_config(args, settings)
    risk = _risk_manager(args, settings)
    results = _run_all_strategies(df, config, risk)
    table = format_comparison(results, benchmark=df["close"])
    _print()
    _print(table)
    _print()
    _print(DEMO_WARNING)
    _print()
    _print("Étape suivante : téléchargez de vraies bougies (tradingbot download ...) puis lancez")
    _print("tradingbot backtest / compare / optimize dessus. Voir README.md.")
    if args.report_dir:
        _print_paths(_save_comparison(results, table, args.report_dir))
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    """``download`` : télécharge des bougies et les met en cache (``data/cache``)."""
    settings = _load_config(args)
    exchange_id, symbol, timeframe = _data_description(args, settings)
    cache_dir = args.cache_dir or DEFAULT_CACHE_DIR
    _print(
        f"Téléchargement de {symbol} {timeframe} sur {exchange_id} "
        f"(depuis {args.since or 'les bougies les plus récentes'}"
        f"{', jusqu’à ' + args.until if args.until else ''})..."
    )
    kwargs: dict[str, Any] = {}
    if args.until:
        kwargs["until"] = args.until
    df = load_or_fetch(
        exchange_id,
        symbol,
        timeframe,
        since=args.since,
        cache_dir=cache_dir,
        refresh=bool(args.refresh),
        exchange=build_public_exchange(exchange_id),
        **kwargs,
    )
    if len(df) == 0:
        raise CliError(
            f"Aucune bougie reçue pour {symbol} {timeframe} sur {exchange_id} : vérifiez le "
            "symbole (format BASE/QUOTE, ex. BTC/USDT), le timeframe et votre connexion."
        )
    _print(f"{len(df)} bougies ({_period(df)})")
    _print(f"Cache : {cache_path(cache_dir, exchange_id, symbol, timeframe)}")
    if args.out:
        save_csv(df, args.out)
        _print(f"Copie : {args.out}")
    _print("Astuce : --refresh force un nouveau téléchargement même si le cache existe.")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    """``backtest`` : une stratégie, un rapport complet."""
    settings = _load_config(args)
    name = args.strategy or settings.strategy.name
    strategy = _make_strategy(name, _strategy_params(args, settings, name))
    df = _load_data(args, settings)
    config = _backtest_config(args, settings)
    risk = _risk_manager(args, settings)
    result = run_backtest(df, strategy, config=config, risk=risk)
    _print()
    _print(format_report(result))
    if args.report_dir:
        _print()
        _print_paths(save_report(result, args.report_dir))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """``compare`` : toutes les stratégies (paramètres par défaut) contre le buy & hold."""
    settings = _load_config(args)
    names = [n.strip() for n in (args.strategies or "").split(",") if n.strip()] or None
    df = _load_data(args, settings)
    config = _backtest_config(args, settings)
    risk = _risk_manager(args, settings)
    results = _run_all_strategies(df, config, risk, names)
    table = format_comparison(results, benchmark=df["close"])
    _print()
    _print(table)
    _print()
    _print(
        "Lecture : le Sharpe mesure le gain par unité de risque (> 1 = bon), le Max DD la pire "
        "baisse depuis un sommet. Une stratégie qui ne bat pas la ligne « Buy & hold » après "
        "frais n'apporte rien par rapport à un simple achat-conservation."
    )
    if args.report_dir:
        _print()
        _print_paths(_save_comparison(results, table, args.report_dir))
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    """``optimize`` : walk-forward (optimisation sur une fenêtre, test sur la suivante)."""
    settings = _load_config(args)
    name = args.strategy or settings.strategy.name
    strategy_cls = _strategy_class(name)
    df = _load_data(args, settings)
    config = _backtest_config(args, settings)
    risk = _risk_manager(args, settings)
    if args.train_bars is None or args.test_bars is None:
        train_default, test_default = default_windows(len(df))
        train_bars = int(args.train_bars) if args.train_bars is not None else train_default
        test_bars = int(args.test_bars) if args.test_bars is not None else test_default
        _print(
            f"Fenêtres : entraînement {train_bars} bougies, test {test_bars} bougies "
            "(valeurs par défaut : 4 fenêtres de test, entraînement ≈ 3 × test)"
        )
    else:
        train_bars, test_bars = int(args.train_bars), int(args.test_bars)
    grid = strategy_cls.param_grid
    n_combos = 1
    for values in grid.values():
        n_combos *= max(1, len(values))
    _print(
        f"Grille de {name} : {n_combos} combinaison(s) de paramètres, métrique {args.metric}, "
        f"{'ancré' if args.anchored else 'glissant'}. Patience : chaque fenêtre lance "
        f"{n_combos} backtests."
    )
    result = walk_forward(
        df,
        strategy_cls,
        grid=None,
        train_bars=train_bars,
        test_bars=test_bars,
        config=config,
        risk=risk,
        metric=args.metric,
        anchored=bool(args.anchored),
        max_combinations=args.max_combinations,
    )
    _print()
    _print(result.summary())
    _print()
    _print("Détail des fenêtres (rendement et trades hors échantillon) :")
    frame = result.windows_frame()
    columns = [
        c
        for c in (
            "window",
            "test_start_time",
            "test_end_time",
            "n_test_bars",
            "n_oos_trades",
            "oos_return",
            "best_params",
        )
        if c in frame.columns
    ]
    detail = frame.loc[:, columns].copy()
    if "oos_return" in detail.columns:
        detail["oos_return"] = detail["oos_return"].map(lambda x: f"{float(x) * 100:+.2f} %")
    for col in ("test_start_time", "test_end_time"):
        if col in detail.columns:
            detail[col] = detail[col].map(lambda ts: pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M"))
    _print(detail.to_string(index=False))
    if args.report_dir:
        _print()
        _print_paths(_save_walkforward(result, args.report_dir))
    return 0


def _trader_state_dir(
    settings: Settings,
    args: argparse.Namespace,
    mode: str,
    exchange_id: str,
    symbol: str,
    timeframe: str,
) -> Path:
    base = Path(getattr(args, "state_dir", None) or settings.general.state_dir)
    return base / mode / f"{_safe_name(exchange_id)}_{_safe_name(symbol)}_{_safe_name(timeframe)}"


def cmd_paper(args: argparse.Namespace) -> int:
    """``paper`` : paper trading temps réel (argent fictif, données publiques de l'exchange)."""
    settings = _load_config(args)
    exchange_id, symbol, timeframe = _data_description(args, settings)
    name = args.strategy or settings.strategy.name
    strategy = _make_strategy(name, _strategy_params(args, settings, name))
    risk = _risk_manager(args, settings)
    base_dir = Path(args.state_dir or settings.general.state_dir)
    state_dir = _trader_state_dir(settings, args, "paper", exchange_id, symbol, timeframe)
    broker_state = base_dir / f"paper_{_safe_name(exchange_id)}_{_safe_name(symbol)}.json"
    initial_cash = (
        float(args.initial_cash) if args.initial_cash is not None else settings.initial_cash
    )

    exchange = build_public_exchange(exchange_id)
    broker = PaperBroker(
        initial_cash=initial_cash,
        fee_rate=settings.costs.fee_rate if args.fee is None else float(args.fee),
        slippage=settings.costs.slippage if args.slippage is None else float(args.slippage),
        price_feed=lambda sym: _ticker_price(exchange, sym),
        state_path=broker_state,
        allow_short=risk.config.allow_short,
    )
    trader = LiveTrader(
        strategy,
        broker,
        symbol,
        timeframe,
        risk,
        _make_data_source(exchange, exchange_id, symbol, timeframe),
        state_dir=state_dir,
        lookback=_trading_lookback(args, strategy),
        notifier=_build_notifier(settings),
        allow_short=risk.config.allow_short,
    )
    _print("Paper trading (ARGENT FICTIF) : aucun ordre réel ne sera passé.")
    for line in _recap_lines(
        settings, exchange_id, symbol, timeframe, strategy, risk.config, state_dir, sandbox=False
    ):
        if not line.startswith("Mode"):
            _print("  " + line)
    _print(
        f"  {'Compte fictif'.ljust(27)} : {broker_state} (cash {broker.cash:,.2f})".replace(
            ",", " "
        )
    )
    return _run_trader(trader, args)


def _live_refusals(settings: Settings, args: argparse.Namespace) -> list[str]:
    """Conditions manquantes pour autoriser le trading réel (liste vide = autorisé)."""
    missing: list[str] = []
    if not settings.is_live:
        missing.append(
            f'le fichier de configuration indique mode = "{settings.general.mode}" : écrivez '
            'mode = "live" dans config.toml (section [general])'
        )
    if not args.i_understand_the_risks:
        missing.append("le drapeau --i-understand-the-risks n'a pas été passé")
    if not settings.has_api_keys:
        missing.append(
            "les clés API sont absentes : renseignez TRADINGBOT_API_KEY et TRADINGBOT_API_SECRET "
            "dans .env (droits « trade » uniquement, JAMAIS le retrait)"
        )
    return missing


def cmd_live(args: argparse.Namespace) -> int:
    """``live`` : trading réel. Trois verrous, un récapitulatif, une confirmation ``OUI``."""
    settings = _load_config(args)
    refusals = _live_refusals(settings, args)
    if refusals:
        _eprint("REFUS : le trading réel n'est pas autorisé pour les raisons suivantes :")
        for reason in refusals:
            _eprint(f"  - {reason}")
        _eprint("Aucun ordre n'a été envoyé. Lisez docs/PASSER_EN_REEL.md avant de continuer.")
        return EXIT_REFUSED

    exchange_id, symbol, timeframe = _data_description(args, settings)
    name = args.strategy or settings.strategy.name
    strategy = _make_strategy(name, _strategy_params(args, settings, name))
    risk = _risk_manager(args, settings)
    state_dir = _trader_state_dir(
        settings, args, "sandbox" if args.sandbox else "live", exchange_id, symbol, timeframe
    )

    _print(LIVE_BANNER)
    _print("Récapitulatif avant connexion :")
    for line in _recap_lines(
        settings,
        exchange_id,
        symbol,
        timeframe,
        strategy,
        risk.config,
        state_dir,
        bool(args.sandbox),
    ):
        _print("  " + line)
    _print()
    _print(
        "Le robot va passer des ordres au marché avec VOTRE argent. Les pertes sont possibles "
        "et peuvent être totales. Vous pouvez arrêter le robot à tout moment avec Ctrl-C."
    )
    if not args.yes:
        try:
            answer = input("Tapez OUI (en majuscules) pour confirmer, autre chose pour annuler : ")
        except EOFError:
            answer = ""
        if answer.strip() != "OUI":
            _print("Annulé : aucun ordre envoyé.")
            return EXIT_REFUSED

    broker: Broker = CcxtBroker(
        exchange_id,
        str(settings.api_key),
        str(settings.api_secret),
        sandbox=bool(args.sandbox),
        confirm_live=True,
    )
    exchange = getattr(broker, "exchange", None)
    if exchange is None:
        exchange = build_public_exchange(exchange_id)
    trader = LiveTrader(
        strategy,
        broker,
        symbol,
        timeframe,
        risk,
        _make_data_source(exchange, exchange_id, symbol, timeframe),
        state_dir=state_dir,
        lookback=_trading_lookback(args, strategy),
        notifier=_build_notifier(settings),
        allow_short=risk.config.allow_short,
    )
    balance = broker.get_balance(symbol)
    _print(
        f"Connecté à {exchange_id} : cash {balance.cash:,.2f}, équité {balance.equity:,.2f}".replace(
            ",", " "
        )
    )
    return _run_trader(trader, args)


# ============================================================================ argparse
def _add_config_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default="config.toml",
        help="fichier de configuration TOML (défaut : config.toml ; valeurs par défaut s'il n'existe pas)",
    )
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default=None,
        help="niveau de journalisation (debug, info, warning, error)",
    )


def _add_cost_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=None,
        help="capital de départ (surcharge [risk] initial_cash)",
    )
    parser.add_argument(
        "--fee", type=float, default=None, help="frais par ordre, fraction (ex. 0.001 = 0,1 %%)"
    )
    parser.add_argument(
        "--slippage", type=float, default=None, help="glissement par ordre, fraction (ex. 0.0005)"
    )
    parser.add_argument(
        "--allow-short", action="store_true", help="autorise la vente à découvert (signaux -1)"
    )


def _add_data_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("source des bougies (CSV ou exchange)")
    group.add_argument(
        "--data",
        default=None,
        help="fichier CSV de bougies (colonnes timestamp, open, high, low, close, volume)",
    )
    group.add_argument(
        "--exchange", default=None, help="identifiant ccxt (binance, kraken...) ; défaut : config"
    )
    group.add_argument("--symbol", default=None, help="paire, ex. BTC/USDT ; défaut : config")
    group.add_argument(
        "--timeframe",
        default=None,
        help="durée des bougies : 1m, 15m, 1h, 4h, 1d ; défaut : config",
    )
    group.add_argument(
        "--since",
        default=None,
        help="date de début ISO 8601 (ex. 2023-01-01) ; sans cette option, seules les bougies récentes sont téléchargées",
    )
    group.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIR,
        help=f"répertoire du cache (défaut : {DEFAULT_CACHE_DIR})",
    )
    group.add_argument("--refresh", action="store_true", help="ignore le cache et retélécharge")


def _add_strategy_options(parser: argparse.ArgumentParser, required: bool) -> None:
    parser.add_argument(
        "--strategy",
        required=required,
        default=None,
        help="nom de la stratégie (voir list-strategies)"
        + ("" if required else " ; défaut : config"),
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="NOM=VALEUR",
        help="paramètre de stratégie, répétable (ex. --param fast=10 --param slow=30)",
    )


def _add_trading_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--exchange", default=None, help="identifiant ccxt ; défaut : config")
    parser.add_argument("--symbol", default=None, help="paire, ex. BTC/USDT ; défaut : config")
    parser.add_argument("--timeframe", default=None, help="durée des bougies ; défaut : config")
    parser.add_argument(
        "--once", action="store_true", help="une seule itération, puis affiche l'état et quitte"
    )
    parser.add_argument(
        "--poll",
        type=float,
        default=None,
        help="secondes entre deux interrogations (défaut : durée de bougie / 12, au moins 10 s)",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=DEFAULT_LOOKBACK,
        help=f"bougies demandées à chaque itération (défaut : {DEFAULT_LOOKBACK})",
    )
    parser.add_argument(
        "--state-dir",
        default=None,
        help="répertoire des fichiers d'état ; défaut : config [general] state_dir",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="arrête la boucle après N itérations (essais)",
    )
    parser.add_argument(
        "--allow-short",
        action="store_true",
        help="autorise la vente à découvert (si le risque l'autorise aussi)",
    )


def build_parser() -> argparse.ArgumentParser:
    """Construit l'analyseur d'arguments de la CLI."""
    parser = argparse.ArgumentParser(
        prog="tradingbot",
        description=(
            "Robot de trading pédagogique : backtest, walk-forward, paper trading et "
            "(sur demande explicite) trading réel. Commencez par `tradingbot demo`."
        ),
        epilog="Aucun robot ne garantit de profit. Ne risquez jamais de l'argent que vous ne pouvez pas perdre.",
    )
    parser.add_argument("--version", action="version", version=f"tradingbot {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<commande>")
    sub.required = True

    p = sub.add_parser("list-strategies", help="liste les stratégies disponibles")
    _add_config_options(p)
    p.set_defaults(func=cmd_list_strategies, log_default="warning", dedup_logs=True)

    p = sub.add_parser(
        "demo", help="backtest de toutes les stratégies sur des données synthétiques (sans réseau)"
    )
    _add_config_options(p)
    _add_cost_options(p)
    p.add_argument(
        "--bars",
        type=int,
        default=DEMO_BARS,
        help=f"nombre de bougies horaires synthétiques (défaut : {DEMO_BARS})",
    )
    p.add_argument(
        "--seed", type=int, default=DEMO_SEED, help=f"graine aléatoire (défaut : {DEMO_SEED})"
    )
    p.add_argument(
        "--report-dir",
        default=None,
        help="écrit le tableau et un rapport par stratégie dans ce dossier",
    )
    p.set_defaults(func=cmd_demo, log_default="warning", dedup_logs=True)

    p = sub.add_parser("download", help="télécharge des bougies et les met en cache")
    _add_config_options(p)
    p.add_argument("--exchange", default=None, help="identifiant ccxt ; défaut : config")
    p.add_argument("--symbol", default=None, help="paire, ex. BTC/USDT ; défaut : config")
    p.add_argument("--timeframe", default=None, help="durée des bougies ; défaut : config")
    p.add_argument("--since", default=None, help="date de début ISO 8601 (ex. 2023-01-01)")
    p.add_argument("--until", default=None, help="date de fin ISO 8601 (incluse)")
    p.add_argument("--out", default=None, help="chemin d'une copie CSV")
    p.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIR,
        help=f"répertoire du cache (défaut : {DEFAULT_CACHE_DIR})",
    )
    p.add_argument("--refresh", action="store_true", help="retélécharge même si le cache existe")
    p.set_defaults(func=cmd_download, log_default="info")

    p = sub.add_parser("backtest", help="backtest d'une stratégie avec rapport détaillé")
    _add_config_options(p)
    _add_strategy_options(p, required=False)
    _add_data_options(p)
    _add_cost_options(p)
    p.add_argument(
        "--report-dir",
        default=None,
        help="écrit equity.csv, trades.csv, summary.txt, metrics.json (et equity.png)",
    )
    p.set_defaults(func=cmd_backtest, log_default="warning", dedup_logs=True)

    p = sub.add_parser("compare", help="compare toutes les stratégies au buy & hold")
    _add_config_options(p)
    _add_data_options(p)
    _add_cost_options(p)
    p.add_argument(
        "--strategies",
        default=None,
        help="sous-ensemble, séparé par des virgules (défaut : toutes)",
    )
    p.add_argument(
        "--report-dir", default=None, help="écrit le tableau et un rapport par stratégie"
    )
    p.set_defaults(func=cmd_compare, log_default="warning", dedup_logs=True)

    p = sub.add_parser("optimize", help="validation walk-forward (optimisation honnête)")
    _add_config_options(p)
    _add_strategy_options(p, required=False)
    _add_data_options(p)
    _add_cost_options(p)
    p.add_argument(
        "--train-bars", type=int, default=None, help="taille de la fenêtre d'entraînement (bougies)"
    )
    p.add_argument(
        "--test-bars", type=int, default=None, help="taille de la fenêtre de test (bougies)"
    )
    p.add_argument(
        "--metric",
        default="sharpe",
        help="métrique optimisée (sharpe, sortino, calmar, total_return, cagr, profit_factor...)",
    )
    p.add_argument(
        "--anchored", action="store_true", help="entraînement toujours depuis la première bougie"
    )
    p.add_argument(
        "--max-combinations",
        type=int,
        default=200,
        help="plafond de combinaisons par fenêtre (défaut : 200)",
    )
    p.add_argument(
        "--report-dir",
        default=None,
        help="écrit walkforward.txt, windows.csv, oos_equity.csv, oos_trades.csv",
    )
    p.set_defaults(func=cmd_optimize, log_default="warning", dedup_logs=True)

    p = sub.add_parser("paper", help="paper trading temps réel (argent fictif, données publiques)")
    _add_config_options(p)
    _add_strategy_options(p, required=False)
    _add_trading_options(p)
    p.add_argument(
        "--initial-cash",
        type=float,
        default=None,
        help="capital fictif de départ (défaut : config)",
    )
    p.add_argument(
        "--fee", type=float, default=None, help="frais par ordre simulés (défaut : config)"
    )
    p.add_argument(
        "--slippage", type=float, default=None, help="glissement simulé (défaut : config)"
    )
    p.set_defaults(func=cmd_paper, log_default="info")

    p = sub.add_parser(
        "live", help='trading RÉEL (exige mode = "live", --i-understand-the-risks et les clés API)'
    )
    _add_config_options(p)
    _add_strategy_options(p, required=False)
    _add_trading_options(p)
    p.add_argument(
        "--i-understand-the-risks",
        action="store_true",
        help="je comprends que je peux perdre l'argent engagé",
    )
    p.add_argument(
        "--yes", action="store_true", help="ne demande pas de taper OUI (scripts) ; à éviter"
    )
    p.add_argument(
        "--sandbox", action="store_true", help="utilise le mode test (testnet) de l'exchange"
    )
    p.set_defaults(func=cmd_live, log_default="info")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Point d'entrée : analyse les arguments, exécute la commande, renvoie le code de sortie."""
    parser = build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        code = exc.code
        return int(code) if isinstance(code, int) else (0 if code is None else EXIT_REFUSED)
    _configure_logging(
        getattr(args, "log_level", None),
        getattr(args, "log_default", "info"),
        dedup=getattr(args, "dedup_logs", False),
    )
    func = args.func
    try:
        return int(func(args))
    except KeyboardInterrupt:
        _eprint("Arrêt demandé (Ctrl-C).")
        return 130
    except CliError as exc:
        _eprint(f"Erreur : {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001 - dernier filet : message lisible, code 1
        logger.debug("Erreur inattendue", exc_info=exc)
        _eprint(f"Erreur ({type(exc).__name__}) : {exc}")
        return 1
