# Architecture et contrats entre modules

Ce document est la référence pour quiconque modifie le code. Chaque module a un
contrat précis ; les tests (`tests/`) le vérifient. Aucun test ne doit accéder au réseau.

## Conventions globales

- Python ≥ 3.11, pandas ≥ 2.2 (compatible pandas 3 : pas d'affectation chaînée,
  copy-on-write), numpy. Dépendance externe unique pour les exchanges : `ccxt`.
- **Bougies OHLCV** : `pd.DataFrame` avec
  - index `DatetimeIndex` **tz-aware UTC**, trié croissant, sans doublon, nommé `timestamp` ;
  - colonnes `open, high, low, close, volume` en `float64` (toujours en minuscules).
  - La fonction `tradingbot.data.loader.ensure_ohlcv(df)` normalise/valide ce format et
    est appelée à l'entrée de chaque composant.
- **Signaux** : voir `tradingbot/strategies/base.py`. Le signal de la bougie `t` est
  exécuté à l'**ouverture de la bougie `t+1`**. Jamais de look-ahead : une stratégie
  ne doit utiliser que les données ≤ `t` (pas de `shift(-1)`, pas de `center=True`,
  pas de normalisation sur toute la série).
- Tous les montants monétaires sont des `float` en devise de cotation (USDT, USD, EUR).
- Fractions, jamais de pourcentages : `0.01` = 1 %.
- Les messages destinés à l'utilisateur (docstrings publiques, CLI, erreurs) sont en
  **français** ; les identifiants de code en anglais.
- Logging via `logging.getLogger(__name__)`, jamais `print` hors de `cli.py`.

## `tradingbot/indicators.py`

Fonctions pures, vectorisées, qui renvoient une `pd.Series` alignée sur l'entrée
(NaN pendant la période de chauffe). Signatures :

```python
sma(series, window) -> Series
ema(series, span) -> Series                       # ewm(span=..., adjust=False)
rsi(series, window=14) -> Series                  # Wilder (lissage ewm alpha=1/window), 0..100
atr(df, window=14) -> Series                      # true range + lissage Wilder
macd(series, fast=12, slow=26, signal=9) -> DataFrame[macd, signal, hist]
bollinger(series, window=20, n_std=2.0) -> DataFrame[middle, upper, lower, pct_b, bandwidth]
donchian(df, window=20) -> DataFrame[upper, lower, middle]   # calculé sur high/low
                                                             # EXCLUANT la bougie courante
                                                             # (shift(1)) pour éviter
                                                             # l'auto-déclenchement
adx(df, window=14) -> DataFrame[adx, plus_di, minus_di]
rolling_volatility(series, window=20, periods_per_year=None) -> Series  # écart-type
                                                             # des rendements log,
                                                             # annualisé si demandé
returns(series, log=False) -> Series
crossover(a, b) -> Series[bool]   # True à la bougie où a passe au-dessus de b
crossunder(a, b) -> Series[bool]
```

## `tradingbot/data/`

### `loader.py`

```python
ensure_ohlcv(df) -> DataFrame           # normalise colonnes (casse, alias Date/Time/
                                        # timestamp/Adj Close), index UTC trié unique,
                                        # float64 ; lève ValueError si colonne manquante
load_csv(path) -> DataFrame             # CSV avec date en 1re colonne ou colonne
                                        # timestamp/date/time (ISO ou epoch ms/s)
save_csv(df, path) -> None
fetch_ohlcv(exchange_id, symbol, timeframe, since=None, until=None, limit=None,
            exchange=None) -> DataFrame # pagination ccxt ; `exchange` injectable
                                        # (objet avec .fetch_ohlcv/.parse8601/.rateLimit)
                                        # pour les tests ; ne retourne que des bougies
                                        # CLÔTURÉES si drop_last_incomplete=True (défaut)
cache_path(cache_dir, exchange_id, symbol, timeframe) -> Path
load_or_fetch(exchange_id, symbol, timeframe, since=None, cache_dir="data/cache",
              refresh=False, exchange=None) -> DataFrame
timeframe_to_timedelta(tf) -> pd.Timedelta        # "1h" -> 1 heure ; lève ValueError
infer_periods_per_year(index) -> float            # 1h -> 8760, 1d -> 365 (crypto 24/7)
resample_ohlcv(df, rule) -> DataFrame             # ex. "4h"
```

### `synthetic.py`

```python
generate_ohlcv(n=2000, seed=0, start="2022-01-01", freq="1h", start_price=100.0,
               regimes=None, vol=0.01) -> DataFrame
# regimes : liste de (n_bars, drift_par_bar, vol_par_bar) enchaînés ; par défaut un mélange
# haussier / baissier / latéral. Résultat cohérent : low <= min(open, close),
# high >= max(open, close), volume > 0.
```

## `tradingbot/risk.py`

```python
@dataclass
class RiskConfig:
    risk_per_trade: float = 0.01        # fraction du capital perdue si le stop est touché
    max_position_fraction: float = 0.95 # notionnel max / équité
    max_drawdown: float = 0.20          # 0 ou None = désactivé
    daily_loss_limit: float = 0.05      # 0 ou None = désactivé
    allow_short: bool = False
    min_trade_notional: float = 10.0    # en dessous, on ne trade pas (ex. 10 USDT)
    vol_target: float | None = None     # volatilité annualisée cible (ex. 0.20) ; si
                                        # définie, la taille est pondérée par
                                        # vol_target / vol_réalisée (plafonnée à 1 levier)

class RiskManager:
    def __init__(self, config: RiskConfig): ...
    def position_size(self, equity, price, stop_price=None, realized_vol=None) -> float:
        # unités de l'actif. Si stop_price : size = equity*risk_per_trade/|price-stop|.
        # Sinon : size = equity*max_position_fraction/price. Toujours plafonné par
        # max_position_fraction et par min_trade_notional (→ 0.0 si trop petit).
    def update(self, timestamp, equity) -> None     # alimente peak, equity du jour
    def kill_switch_active(self) -> bool            # drawdown > max_drawdown
    def daily_limit_hit(self, timestamp, equity) -> bool
    def can_open(self, side: int, timestamp, equity) -> tuple[bool, str]
        # (autorisé, raison) ; short refusé si not allow_short ; refusé si kill switch /
        # limite journalière
    def reset_daily(self, timestamp) -> None
    def state(self) -> dict / load_state(d)         # sérialisable JSON
```

## `tradingbot/backtest/`

### `engine.py`

```python
@dataclass
class BacktestConfig:
    initial_cash: float = 10_000.0
    fee_rate: float = 0.001         # par ordre, sur le notionnel
    slippage: float = 0.0005        # par ordre, défavorable
    stop_loss_pct: float | None = None     # défaut si la stratégie ne fournit pas de stop
    take_profit_pct: float | None = None
    trailing_stop_pct: float | None = None # ratchet géré par le moteur
    allow_short: bool = False
    execute_on: str = "next_open"   # seule valeur supportée (documentée)

@dataclass
class Trade:
    entry_time, exit_time, side(+1/-1), entry_price, exit_price, size, pnl, pnl_pct,
    fees, bars_held, exit_reason  # "signal" | "stop_loss" | "take_profit" | "end" |
                                  # "kill_switch" | "daily_limit"

@dataclass
class BacktestResult:
    strategy_name: str
    params: dict
    config: BacktestConfig
    equity: pd.Series          # valeur du portefeuille à la clôture de chaque bougie
    returns: pd.Series         # equity.pct_change().fillna(0)
    positions: pd.Series       # position effective (-1/0/1) tenue pendant la bougie
    trades: pd.DataFrame       # une ligne par Trade (colonnes = champs de Trade)
    metrics: dict[str, float]  # cf. metrics.compute_metrics
    signals: pd.DataFrame      # signaux bruts de la stratégie
    def summary(self) -> str   # tableau texte en français

def run_backtest(df, strategy, config=None, risk=None) -> BacktestResult
```

Règles du moteur (boucle bougie par bougie, en numpy pour la vitesse) :
1. À la bougie `t`, la position tenue est celle décidée en `t-1`. Les ordres de
   changement de position s'exécutent à `open[t]` ± slippage, frais = notionnel × fee_rate.
2. Stops : pour un long, si `open[t] <= stop` → sortie à `open[t]` (gap) ; sinon si
   `low[t] <= stop` → sortie au prix du stop. Symétrique pour take-profit (high) et pour
   les shorts. Si stop ET take-profit sont touchés dans la même bougie, on suppose
   le **stop** (pessimiste). Une sortie sur stop ne rentre pas à nouveau avant la
   bougie suivante où le signal le demande encore.
3. Taille : `risk.position_size(equity, open[t], stop_price)` ; si la stratégie ne
   fournit pas de stop, utilise `stop_loss_pct` s'il existe ; sinon taille =
   `max_position_fraction`. Sans `risk`, on utilise `RiskManager(RiskConfig())`.
4. Le kill switch de `RiskManager` ferme la position et bloque toute réouverture
   jusqu'à la fin du backtest (exit_reason "kill_switch").
5. Les bougies de chauffe (`strategy.warmup_bars()`) ne tradent pas.
6. `equity[t]` = cash + position × close[t]. Dernière position fermée à la dernière
   clôture (exit_reason "end").
7. `allow_short=False` : les signaux -1 sont traités comme 0.

### `metrics.py`

```python
compute_metrics(equity, trades, periods_per_year, risk_free=0.0, benchmark=None) -> dict
# clés : total_return, cagr, annual_volatility, sharpe, sortino, max_drawdown (négatif,
# ex. -0.23), max_drawdown_duration (en bougies), calmar, n_trades, win_rate,
# profit_factor, avg_trade_pct, avg_win_pct, avg_loss_pct, best_trade_pct,
# worst_trade_pct, avg_bars_held, exposure (fraction des bougies en position),
# buy_hold_return (si benchmark = série de prix fournie), final_equity
drawdown_series(equity) -> Series
sharpe_ratio(returns, periods_per_year, risk_free=0.0) -> float   # 0.0 si vol nulle
sortino_ratio(...), cagr(equity, periods_per_year), max_drawdown(equity)
```

### `walkforward.py`

```python
grid_search(df, strategy_cls, grid=None, config=None, risk=None, metric="sharpe",
            max_combinations=200) -> pd.DataFrame   # une ligne par combinaison, triée
walk_forward(df, strategy_cls, grid=None, train_bars=..., test_bars=..., config=None,
             risk=None, metric="sharpe", anchored=False) -> WalkForwardResult
@dataclass
class WalkForwardResult:
    windows: list[dict]        # {train_start, train_end, test_start, test_end,
                               #  best_params, is_metric, oos_metric}
    oos_equity: pd.Series      # équité OOS recomposée (chaînage des rendements)
    oos_metrics: dict
    is_metrics_mean: float ; oos_metrics_mean: float ; overfit_ratio: float
    def summary(self) -> str
```

### `report.py`

```python
format_report(result: BacktestResult, benchmark_return=None) -> str   # texte FR
format_comparison(results: list[BacktestResult]) -> str               # tableau
save_report(result, out_dir) -> list[Path]  # equity.csv, trades.csv, summary.txt,
                                            # equity.png si matplotlib installé
plot_result(result, path) -> Path | None    # import matplotlib paresseux ; None sinon
```

## `tradingbot/execution/`

- `broker.py` : contrat (déjà écrit, ne pas modifier sans mettre à jour ce doc).
- `paper.py` : `PaperBroker(initial_cash, fee_rate=0.001, slippage=0.0005,
  price_feed: Callable[[str], float], state_path: str | Path | None = None)`.
  Exécution immédiate au prix `price_feed(symbol)` ± slippage ; refuse (BrokerError)
  un achat dépassant le cash ou une vente à découvert si `allow_short=False`.
  Journal des ordres `self.orders: list[Order]`. État persistant JSON (`save()`,
  `load()` automatique à l'init si le fichier existe).
- `ccxt_broker.py` : `CcxtBroker(exchange_id, api_key, api_secret, sandbox=False,
  confirm_live=False, exchange=None)`. **Lève `BrokerError` à l'init si
  `confirm_live` est False** (double sécurité avec la CLI). Ordres `create_order(
  symbol, "market", side, amount)` avec `amount` arrondi via `amount_to_precision`,
  vérification `min amount`/`min cost` des `markets`. `exchange` injectable (fake)
  pour les tests. Jamais de retrait, jamais de levier.

## `tradingbot/live/`

- `runner.py` :

```python
class LiveTrader:
    def __init__(self, strategy, broker, symbol, timeframe, risk: RiskManager,
                 data_source: Callable[[int], pd.DataFrame],  # n_bars -> OHLCV clôturées
                 state_dir="state", lookback=500, notifier=None, allow_short=False)
    def step(self, now=None) -> dict   # une itération : fetch, signal sur la DERNIÈRE
                                       # bougie CLÔTURÉE, risk check, rebalance ; retourne
                                       # un dict d'état {"timestamp", "signal", "action",
                                       # "position", "equity", "reason"} ; idempotent :
                                       # ne re-traite pas une bougie déjà traitée
                                       # (mémorisé dans state_dir/last_bar.json)
    def run(self, poll_seconds=None, max_iterations=None) -> None  # boucle, Ctrl-C propre
```

  Chaque action (entrée, sortie, refus par le risk manager) est loggée et envoyée au
  `notifier` s'il existe. Le `LiveTrader` gère les stops lui-même entre deux bougies :
  à chaque `step`, si le prix courant franchit le stop actif → clôture.
- `notifier.py` : `Notifier` (interface `send(text)`), `LogNotifier`,
  `TelegramNotifier(token, chat_id, session=None)` (requests injectable, timeouts, jamais
  d'exception propagée : une notification qui échoue ne doit pas arrêter le bot).

## `tradingbot/config.py`

```python
@dataclass
class Settings:  # miroir de config.example.toml, avec sous-dataclasses
    general: GeneralSettings ; strategy: StrategySettings ; risk: RiskConfig ;
    costs: CostSettings ; notifications: NotificationSettings
    api_key: str | None ; api_secret: str | None ; telegram_token/chat_id

load_settings(path="config.toml", env=None) -> Settings   # TOML via tomllib ; valeurs
                                                          # manquantes = défauts ;
                                                          # lit `.env` (format KEY=VALUE)
                                                          # puis os.environ (prioritaire)
load_dotenv(path=".env") -> dict[str, str]                # sans dépendance externe
```

## `tradingbot/cli.py`

`python -m tradingbot <commande>` (ou `tradingbot <commande>` après `pip install -e .`).
Toutes les sorties utilisateur en français, `argparse`, code de sortie ≠ 0 en erreur.

| Commande | Rôle |
|---|---|
| `list-strategies` | nom + description de chaque stratégie |
| `demo` | backtest sur données synthétiques, sans réseau (premier contact) |
| `download --exchange --symbol --timeframe [--since] [--out]` | télécharge et met en cache |
| `backtest --strategy S (--data fichier.csv \| --exchange E --symbol X --timeframe T) [--param k=v ...] [--report-dir]` | backtest + rapport |
| `compare (--data ...)` | toutes les stratégies vs buy & hold |
| `optimize --strategy S ... [--train-bars --test-bars --metric]` | walk-forward |
| `paper --strategy S --exchange --symbol --timeframe [--once] [--poll]` | paper trading temps réel sur données publiques |
| `live ... --i-understand-the-risks` | trading réel ; exige `mode = "live"` dans config.toml, le flag, et les clés dans `.env` ; affiche un récapitulatif et demande de taper `OUI` |

`--param` accepte `k=v` avec conversion automatique int/float/bool.

## Tests

`pytest` sans réseau : les objets `exchange` et `session` sont injectés (fakes). Chaque
module a son fichier `tests/test_<module>.py`. Les tests du moteur de backtest
utilisent de petites séries construites à la main dont le résultat est calculé à la
main (frais, slippage, stops), pas seulement « ça tourne ».
