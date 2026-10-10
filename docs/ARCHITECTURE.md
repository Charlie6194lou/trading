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
true_range(df) -> Series                          # brique de atr / adx
wilder_smooth(series, window) -> Series           # lissage Wilder (ewm alpha=1/window)
crossover(a, b) -> Series[bool]   # True à la bougie où a passe au-dessus de b ;
crossunder(a, b) -> Series[bool]  # b : Series ou scalaire (ex. crossunder(rsi, 30))
```

## `tradingbot/strategies/`

`base.py` : contrat `Strategy` (déjà écrit, ne pas modifier sans mettre à jour ce doc).
Registre dans `__init__.py` :

```python
STRATEGIES: dict[str, type[Strategy]]        # clé = Strategy.name
register(cls) -> cls                         # décorateur ; ValueError si nom absent/dupliqué
get_strategy(name, **params) -> Strategy     # KeyError (avec la liste des noms) si inconnue
list_strategies() -> list[tuple[str, str]]   # (nom, description) trié par nom
```

Stratégies livrées : `sma_cross`, `ema_trend`, `rsi_reversion`, `bollinger_reversion`,
`donchian_breakout` (détails pour débutant dans `docs/STRATEGIES.md`).

Contrat du stop exporté : une stratégie porte son `stop_loss` sur **chaque bougie** où
`signal != 0` (jamais NaN en position). `sma_cross`, `donchian_breakout`, `rsi_reversion`
et `bollinger_reversion` répètent le stop d'entrée (constant) tant que
`signal[t] == signal[t-1] != 0` ; `ema_trend` exporte son chandelier à chaque bougie et
sort (signal 0, plat une bougie) dès que `low <= stop` (`high >= stop` pour un short),
c'est-à-dire exactement quand le moteur / le bot exécutent ce stop. Ainsi un moteur ou un
bot qui (ré)entre en cours de position dispose toujours d'un stop.

## `tradingbot/data/`

### `loader.py`

```python
ensure_ohlcv(df, *, dayfirst=None) -> DataFrame
                                        # normalise colonnes (casse, alias timestamp/
                                        # datetime/date/time ; colonnes Date + Time
                                        # séparées combinées ; `Adj Close` IGNORÉE : prix
                                        # bruts non ajustés), index UTC trié unique,
                                        # float64 ; lève ValueError si colonne manquante.
                                        # Dates : ISO 8601, epoch s/ms/us/ns (unité devinée),
                                        # JJ/MM/AAAA ou MM/JJ/AAAA avec ordre détecté sur
                                        # TOUTE la colonne (jamais ligne par ligne) ;
                                        # `dayfirst` force l'ordre ; AmbiguousDateError
                                        # (sous-classe de ValueError) si indécidable
load_csv(path, *, dayfirst=None) -> DataFrame
                                        # CSV avec date en 1re colonne ou colonne
                                        # timestamp/date/time, ou Date + Time ; séparateur
                                        # , ; ou tabulation ; virgule décimale détectée
                                        # (export Excel FR) ; fichier SANS en-tête (1re
                                        # ligne numérique, >= 5 colonnes : Binance Vision,
                                        # Kraken OHLCVT) -> colonnes timestamp,open,high,
                                        # low,close,volume (Kraken : vwap puis volume),
                                        # avertissement journalisé
save_csv(df, path) -> None
fetch_ohlcv(exchange_id, symbol, timeframe, since=None, until=None, limit=None,
            exchange=None, *, drop_last_incomplete=True, page_size=1000,
            sleep=time.sleep, now_ms=None) -> DataFrame
                                        # pagination ccxt ; `exchange` injectable
                                        # (objet avec .fetch_ohlcv/.parse8601/.rateLimit/
                                        # .milliseconds/.timeframes) pour les tests ;
                                        # ne retourne que des bougies CLÔTURÉES si
                                        # drop_last_incomplete=True (défaut) ;
                                        # `sleep` / `now_ms` injectables (tests sans attente).
                                        # since/until : ISO 8601 (AAAA-MM-JJ[THH:MM:SSZ]),
                                        # datetime/Timestamp ou epoch s/ms (unité devinée
                                        # comme pour les CSV) ; JJ/MM/AAAA REFUSÉ
                                        # (ValueError) ; `until` inclusif sur l'OUVERTURE
                                        # de la bougie, et une chaîne date seule
                                        # ("2024-12-31") inclut toute la journée.
                                        # `limit` = nb max de bougies brutes (dont la
                                        # bougie en cours) : les premières depuis `since`,
                                        # sinon les plus récentes, même si limit >
                                        # page_size ou > plafond par appel de l'exchange
                                        # (pagination depuis une fenêtre « maintenant -
                                        # (limit+1) × timeframe »).
                                        # La pagination s'arrête sur page vide, until/limit
                                        # atteint, bougie en cours reçue, ou absence de
                                        # progrès — JAMAIS sur une page plus courte que
                                        # page_size (coinbase/okx plafonnent à 300).
                                        # WARNING journalisé si la 1re bougie reçue est
                                        # > 2 timeframes après `since` (ex. Kraken : 720
                                        # dernières bougies seulement)
cache_path(cache_dir, exchange_id, symbol, timeframe) -> Path
load_or_fetch(exchange_id, symbol, timeframe, since=None, cache_dir="data/cache",
              refresh=False, exchange=None, **fetch_kwargs) -> DataFrame
                                        # fetch_kwargs transmis à fetch_ohlcv (ex. until=,
                                        # limit=). Cache utilisé s'il COUVRE la demande ;
                                        # sinon le début (since < 1re bougie) et/ou la fin
                                        # (until > dernière bougie) manquants sont
                                        # téléchargés, fusionnés et sauvegardés ; limit >
                                        # taille du cache -> téléchargement complet. Sans
                                        # `until`, pas de rafraîchissement automatique
                                        # (refresh=True / `download`). Résultat restreint à
                                        # [since, until] et à `limit`. Décisions en INFO,
                                        # WARNING si l'exchange ne fournit pas le complément
timeframe_to_timedelta(tf) -> pd.Timedelta        # "1h" -> 1 heure ; lève ValueError
infer_periods_per_year(index) -> float            # marché 24/7 (crypto) : 365 j / écart
                                                  # médian : 1h -> 8760, 1d -> 365 ;
                                                  # marché fermé le week-end (< 5 % des
                                                  # bougies un sam./dim. sur >= 11 jours,
                                                  # bougies <= 1 j) : 252 jours de bourse ×
                                                  # bougies/jour (médiane par date) :
                                                  # 1d actions -> 252, 1h forex -> 6048,
                                                  # 1h actions (7 bougies/séance) -> 1764
resample_ohlcv(df, rule) -> DataFrame             # ex. "4h"
```

### `synthetic.py`

```python
generate_ohlcv(n=2000, seed=0, start="2022-01-01", freq="1h", start_price=100.0,
               regimes=None, vol=0.01) -> DataFrame   # seed=None → non reproductible
default_regimes(n, vol) -> list[Regime]               # Regime = (n_bars, drift, vol)
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
    max_drawdown: float | None = 0.20   # 0 ou None = désactivé
    daily_loss_limit: float | None = 0.05  # 0 ou None = désactivé
    allow_short: bool = False
    min_trade_notional: float = 10.0    # en dessous, on ne trade pas (ex. 10 USDT)
    vol_target: float | None = None     # volatilité annualisée cible (ex. 0.20) ; si
                                        # définie, la taille est pondérée par
                                        # vol_target / vol_réalisée (plafonnée à 1 levier)
    default_stop_atr_mult: float = 3.0  # stop de secours = 3 × ATR(14) si la stratégie
                                        # n'en donne pas (ni stop_loss_pct)
    default_stop_pct: float = 0.05      # stop de dernier recours 5 % si l'ATR est NaN / 0

DEFAULT_STOP_ATR_WINDOW = 14            # fenêtre de l'ATR du stop de secours

class RiskManager:
    def __init__(self, config: RiskConfig | None = None): ...   # None → RiskConfig()
        # ValueError (en français) si risk_per_trade <= 0, si max_position_fraction n'est
        # pas dans ]0, 1] (1 = tout le capital, jamais de levier), si default_stop_atr_mult
        # <= 0 ou si default_stop_pct n'est pas dans ]0, 1[.
    def protective_stop(self, side, price, strategy_stop=None, atr=None,
                        stop_loss_pct=None) -> float
        # prix du stop d'une entrée (jamais None/NaN) : (1) strategy_stop s'il est fini ET
        # du bon côté du prix (long : stop < price ; short : stop > price) ; (2) sinon
        # price*(1 ∓ stop_loss_pct) si fourni et > 0 ; (3) sinon price ∓
        # default_stop_atr_mult*atr si atr fini et > 0 ; (4) sinon price*(1 ∓
        # default_stop_pct) avec un WARNING. ValueError si side ∉ {-1, 1} ou price <= 0.
        # Moteur et LiveTrader l'appellent AVANT position_size : une position ne s'ouvre
        # jamais sans stop, le sizing est toujours basé sur le risque.
    def position_size(self, equity, price, stop_price=None, realized_vol=None) -> float:
        # unités de l'actif. Si stop_price : size = equity*risk_per_trade/|price-stop|.
        # Sinon : size = equity*max_position_fraction/price. Toujours plafonné par
        # max_position_fraction et par min_trade_notional (→ 0.0 si trop petit).
    def update(self, timestamp, equity) -> None     # alimente peak, equity du jour ;
                                                    # kill switch / limite journalière
                                                    # journalisés en INFO (simulés des
                                                    # centaines de fois en backtest ; le
                                                    # LiveTrader avertit et notifie).
                                                    # Changement de jour UTC : référence =
                                                    # dernière équité connue, sauf si elle
                                                    # date de plus d'un jour (redémarrage,
                                                    # bougies espacées) → équité courante
    def drawdown(self) -> float                     # fraction positive depuis le plus haut
    def kill_switch_active(self) -> bool            # drawdown > max_drawdown
    def daily_limit_hit(self, timestamp, equity) -> bool   # False si limite désactivée
    def can_open(self, side: int, timestamp, equity) -> tuple[bool, str]
        # (autorisé, raison) ; short refusé si not allow_short ; refusé si kill switch /
        # limite journalière
    def set_daily_limit_active(self, active: bool, reason: str = "") -> None
        # la limite journalière n'a de sens qu'en intrajournalier : l'appelant (moteur,
        # LiveTrader) la désactive quand la bougie fait >= 1 jour. Désactivée : update ne
        # verrouille jamais, daily_limit_hit renvoie False, can_open ne la vérifie plus.
        # Décision de contexte (timeframe) : ni sérialisée par state(), ni effacée par reset()
    daily_limit_active -> bool ; daily_limit_inactive_reason -> str   # propriétés
    def reset_daily(self, timestamp, equity=None) -> None
    def reset(self) -> None                         # remise à zéro complète
    def state(self) -> dict / load_state(d)         # sérialisable JSON
    peak_equity / day_start_equity                  # propriétés (None avant le 1er update)
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
    reentry_after_stop: str = "wait"  # "wait" | "immediate" (règle 2) ; ValueError sinon

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
    periods_per_year: float = 365.0   # déduit de l'index (infer_periods_per_year)
    events: list[dict] = []    # garde-fous déclenchés : {timestamp (ISO str), type
                               #  "kill_switch" | "daily_limit", equity, drawdown (fraction
                               #  positive), detail (phrase FR)} ; un événement par
                               #  déclenchement du kill switch, un par jour UTC bloqué
    warmup_bars: int = 0       # strategy.warmup_bars() ; le buy & hold de référence part
                               # de open[warmup_bars + 1] (open[0] si 0)
    daily_limit_active: bool = True   # False si la limite journalière a été désactivée
                                      # (bougies >= 1 jour)
    def guard_rails_text(self) -> str  # une ligne FR (vide si rien) ; summary() l'affiche
    def summary(self) -> str   # tableau texte en français

def run_backtest(df, strategy, config=None, risk=None) -> BacktestResult
```

`metrics` contient, en plus des clés de `compute_metrics`, `kill_switch_triggered` (0/1) et
`n_daily_limit_hits` (int), et `buy_hold_return` est **remplacé** par le buy & hold net sur
la même période tradable (voir règle 8).

Règles du moteur (boucle bougie par bougie, en numpy pour la vitesse) :
1. À la bougie `t`, la position tenue est celle décidée en `t-1`. Les ordres de
   changement de position s'exécutent à `open[t]` ± slippage, frais = notionnel × fee_rate.
2. Stops, dans cet ordre pour un long : `open[t] <= stop` → sortie à `open[t]` (gap) ;
   sinon `open[t] >= take` → sortie à `open[t]` (gap au-delà de l'objectif : l'ouverture
   est le premier prix connu, elle lève l'ambiguïté) ; sinon `low[t] <= stop` → sortie au
   prix du stop ; sinon `high[t] >= take` → sortie à l'objectif. Symétrique pour les
   shorts. Stop ET take-profit touchés **intrabar** dans la même bougie → on suppose le
   **stop** (pessimiste). Ré-entrée après une sortie décidée par le moteur (`stop_loss`,
   `take_profit`, `daily_limit`) : avec `reentry_after_stop="wait"` (défaut) le côté sorti
   est **bloqué** tant que le signal brut n'a pas quitté ce côté au moins une fois (0 ou
   côté opposé sur une bougie ultérieure) ; une nouvelle transition 0→côté (ou un
   retournement) ouvre à nouveau. `"immediate"` = ancien comportement (ré-ouverture dès la
   bougie suivante si le signal le demande encore). Le `LiveTrader` applique la même règle
   (`blocked_side` persisté dans `position.json`). En position, un `stop_loss` de la
   stratégie du mauvais côté de la clôture (long : stop >= close) est ignoré.
3. Taille : une position ne s'ouvre **jamais** sans stop. `stop = risk.protective_stop(side,
   prix d'exécution, stop_loss[t-1], ATR(14)[t-1], stop_loss_pct)` (priorité : stop de la
   stratégie s'il est du bon côté du prix → `stop_loss_pct` → `default_stop_atr_mult × ATR`
   → `default_stop_pct`), puis `risk.position_size(equity, prix, stop)` : le sizing est
   toujours basé sur `risk_per_trade`, plafonné par `max_position_fraction` et par le cash
   disponible (longs). Sans `risk`, on utilise `RiskManager(RiskConfig())`.
4. Le kill switch de `RiskManager` ferme la position et bloque toute réouverture
   jusqu'à la fin du backtest (exit_reason "kill_switch") ; la limite journalière ferme et
   bloque jusqu'au jour UTC suivant ("daily_limit"). Chaque déclenchement est consigné dans
   `result.events` et compté dans `metrics["kill_switch_triggered"]` /
   `metrics["n_daily_limit_hits"]`. La limite journalière n'a de sens qu'en intrajournalier :
   si la durée médiane d'une bougie est >= 1 jour, le moteur appelle
   `risk.set_daily_limit_active(False, "bougies >= 1 jour")`, journalise une fois en INFO
   « limite de perte journalière désactivée : bougies >= 1 jour » et expose
   `result.daily_limit_active = False` (le rapport l'indique).
5. Les bougies de chauffe (`strategy.warmup_bars()`) ne tradent pas.
6. `equity[t]` = cash + position × close[t]. Dernière position fermée à la dernière
   clôture (exit_reason "end").
7. `allow_short=False` : les signaux -1 sont traités comme 0.
8. Buy & hold de référence : `metrics["buy_hold_return"]` = achat à
   `open[warmup + 1]` (première exécution possible ; `open[0]` si `warmup = 0`) revendu à la
   dernière clôture, net d'un aller-retour `(1+slippage)(1+fee)` / `(1−slippage)(1−fee)` ;
   `compute_metrics` reçoit `close[warmup+1:]` comme benchmark. Le rapport doit comparer sur
   cette même période tradable (« Buy & hold sur la même période tradable, frais inclus »).

### `metrics.py`

```python
compute_metrics(equity, trades, periods_per_year, risk_free=0.0, benchmark=None,
                positions=None) -> dict
# clés : total_return, cagr, annual_volatility, sharpe, sortino, max_drawdown (négatif,
# ex. -0.23), max_drawdown_duration (en bougies), calmar, n_trades, win_rate,
# profit_factor, avg_trade_pct, avg_win_pct, avg_loss_pct, best_trade_pct,
# worst_trade_pct, avg_bars_held, exposure (fraction des bougies en position, d'après
# `positions`), buy_hold_return (si benchmark = série de prix fournie), final_equity.
# Ne lève JAMAIS (entrées vides, équité constante, croissance hors plage).
drawdown_series(equity) -> Series
sharpe_ratio(returns, periods_per_year, risk_free=0.0) -> float   # 0.0 si vol nulle
sortino_ratio(...), max_drawdown(equity), max_drawdown_duration(equity) -> int,
annual_volatility(returns, periods_per_year), calmar_ratio(cagr_value, max_dd)  # 0.0 si
                                                             # cagr infini ou sans drawdown
years_elapsed(equity, periods_per_year) -> float  # années couvertes : temps écoulé entre le
                                                  # 1er et le dernier point / 365 jours si
                                                  # `equity` est une Series à DatetimeIndex
                                                  # (données à trous correctement
                                                  # annualisées) ; sinon (len-1)/ppy
annualize_return(total_return, years) -> float    # (1+r)^(1/years) - 1 en espace log :
                                                  # `inf` au lieu d'OverflowError si hors
                                                  # plage float ; -1.0 si r <= -1 ; 0.0 si
                                                  # years <= 0
cagr(equity, periods_per_year) -> float           # annualize_return(last/first - 1,
                                                  # years_elapsed(...)) ; 0.0 si < 2 points ;
                                                  # -1.0 si équité finale <= 0 ; peut valoir
                                                  # `inf` (jamais d'exception ; le rapport
                                                  # l'affiche « ∞ », metrics.json `null`)
win_rate(trades), profit_factor(trades), exposure(positions)
```

### `walkforward.py`

```python
grid_search(df, strategy_cls, grid=None, config=None, risk=None, metric="sharpe",
            max_combinations=200) -> pd.DataFrame   # une ligne par combinaison, triée ;
                                        # métriques calculées à partir de la fin de la
                                        # chauffe de CHAQUE combinaison (bougies plates
                                        # exclues : total_return / max_drawdown / n_trades
                                        # identiques à run_backtest, Sharpe / CAGR /
                                        # exposure non dilués → combinaisons de chauffe
                                        # différente comparables)
default_windows(n_bars, n_windows=4, train_test_ratio=3) -> (train_bars, test_bars)
walk_forward(df, strategy_cls, grid=None, train_bars=None, test_bars=None, config=None,
             risk=None, metric="sharpe", anchored=False, max_combinations=200)
             -> WalkForwardResult      # train/test None → default_windows(len(df))
@dataclass
class WalkForwardResult:
    windows: list[dict]        # {window, train_start, train_end, test_start, test_end,
                               #  best_params, is_metric, oos_metric, n_test_bars,
                               #  n_oos_trades, oos_return, oos_kill_switch (bool)}
                               # (bornes = indices de bougies)
    oos_equity: pd.Series      # équité OOS recomposée (chaînage des rendements)
    oos_metrics: dict          # dont buy_hold_return = buy & hold NET (règle 8 du moteur)
                               # de open[train_bars+1] à la dernière clôture
    is_metrics_mean: float ; oos_metrics_mean: float   # pondérées par n_test_bars, valeurs
                               # infinies / NaN ignorées
    overfit_ratio: float       # is_mean > 0 : oos/is ; is_mean < 0 et oos_mean < 0 :
                               # is/oos (= |IS|/|OOS|, métriques négatives comme
                               # max_drawdown) ; sinon NaN. Toujours : ≈ 1 robuste,
                               # < 1 sur-apprentissage, > 1 mieux hors échantillon
    strategy_name: str ; metric: str ; train_bars: int ; test_bars: int ; anchored: bool
    oos_returns: pd.Series ; oos_positions: pd.Series ; oos_trades: pd.DataFrame
    periods_per_year: float
    n_boundary_reopens: int ; boundary_fees: float   # aller-retours forcés par les bornes
                               # de fenêtres (sortie "end" à la dernière bougie + ré-entrée
                               # du même côté à open[test_start+1] suivant) et leurs frais
    max_drawdown_limit: float | None                 # RiskConfig.max_drawdown utilisé
    oos_kill_switch_time: pd.Timestamp | None        # 1re bougie où la courbe OOS enchaînée
                               # dépasse max_drawdown (en réel le robot se serait arrêté là)
    oos_kill_switch_drawdown: float                  # drawdown (fraction positive) alors
    n_window_kill_switches: int                      # fenêtres où le moteur l'a déclenché
    daily_limit_active: bool                         # False si bougies >= 1 jour
    n_windows -> int (propriété) ; windows_frame() -> pd.DataFrame  # une ligne par fenêtre
    def guard_rails_lines(self) -> list[str]
    def summary(self) -> str
```

Règles du walk-forward :
1. **Même base IS / OOS** (`_backtest_window`) : chaque backtest (entraînement ou test)
   reçoit `strategy.warmup_bars()` bougies d'historique avant sa fenêtre quand elles
   existent ; équité, positions et trades sont restreints à la fenêtre ; la métrique est
   calculée à partir de la première bougie de décision (`start` si l'historique est complet,
   sinon la fin de la chauffe). Invariant testé : avec `train_bars == test_bars`,
   `windows[k+1]["is_metric"] == windows[k]["oos_metric"]`. La première fenêtre
   d'entraînement (`train_start = 0`) n'a pas d'historique : elle est jugée sur moins de
   bougies actives, c'est inévitable et documenté.
2. La position n'est **pas portée** d'une fenêtre de test à la suivante (le moteur n'a pas
   d'état initial) : fermée `"end"` à la dernière bougie, rouverte au plus tôt à
   `open[test_start + 1]`. `summary()` le dit et chiffre ces aller-retours
   (`n_boundary_reopens`, `boundary_fees`, déjà inclus dans `oos_equity`).
3. Le `RiskManager` est remis à zéro à chaque backtest (kill switch et limite journalière
   repartent de zéro par fenêtre). Après enchaînement, `oos_equity` est vérifiée contre
   `max_drawdown` (`risk.config`, ou `RiskConfig()` si `risk` est None) : première bougie
   sous la limite → `oos_kill_switch_time`, WARNING journalisé et avertissement dans
   `summary()` (« ⚠ En réel, le kill switch … aurait arrêté le robot le … »).
4. Moyennes IS / OOS pondérées par `n_test_bars` (dernière fenêtre tronquée) et calculées
   sur les valeurs finies ; `summary()` indique le nombre de fenêtres écartées.
5. `daily_limit_active` vient du moteur (bougies >= 1 jour → False, ligne dans `summary()`).

### `report.py`

```python
format_report(result: BacktestResult, benchmark_return=None) -> str   # texte FR
    # en-tête (dont « Stop de protection » et « Ré-entrée après stop »), métriques, trades,
    # « Comparaison avec le buy & hold » (buy & hold sur la même période tradable, frais
    # inclus = metrics["buy_hold_return"] du moteur ; une ligne le rappelle, sauf si
    # benchmark_return est fourni par l'appelant), section « Garde-fous déclenchés » (une
    # phrase pour débutant par événement de result.events : « le robot a cessé de trader le
    # … après une baisse de X % depuis le sommet » ; limite journalière inactive si
    # result.daily_limit_active est False ; « Aucun garde-fou déclenché » sinon),
    # pires / meilleurs trades (prix en chiffres significatifs : 4 décimales à partir de 1,
    # sinon 6 chiffres significatifs, scientifique sous 1e-4), lecture pour débutant
format_comparison(results: list[BacktestResult], benchmark=None) -> str  # tableau ;
                                            # benchmark = série de prix (close) → ligne
                                            # « Buy & hold » sur la période tradable
                                            # COMMUNE (après la plus longue chauffe,
                                            # max(warmup_bars) + 1), nette d'un
                                            # aller-retour frais + glissement, Sharpe / DD /
                                            # CAGR (temps écoulé) ; sans benchmark,
                                            # metrics["buy_hold_return"] de la stratégie à
                                            # la plus longue chauffe. Nom suffixé
                                            # « ⚠ arrêt » (kill switch) / « ⚠ limite journ.
                                            # ×N » ; notes sous le tableau (période du buy &
                                            # hold, explication des ⚠, limite journalière
                                            # inactive si bougies >= 1 jour)
save_report(result, out_dir) -> list[Path]  # equity.csv, trades.csv, summary.txt,
                                            # metrics.json, equity.png si matplotlib
                                            # installé (sinon avertissement journalisé)
plot_result(result, path) -> Path | None    # import matplotlib paresseux ; None sinon
```

## `tradingbot/execution/`

- `broker.py` : contrat (déjà écrit, ne pas modifier sans mettre à jour ce doc).
- `paper.py` : `PaperBroker(initial_cash, fee_rate=0.001, slippage=0.0005,
  price_feed: Callable[[str], float] | None = None, state_path: str | Path | None = None,
  allow_short=False)`.
  Exécution immédiate au prix `price_feed(symbol)` ± slippage (ou au dernier prix fixé
  par `set_price(symbol, price)` sans `price_feed`) ; refuse (BrokerError) un achat
  dépassant le cash ou une vente à découvert si `allow_short=False`.
  Journal des ordres `self.orders: list[Order]`. État persistant JSON (`state()`,
  `load_state(d)`, `save()`, `load()` automatique à l'init si le fichier existe, `reset()`).
- `ccxt_broker.py` : `CcxtBroker(exchange_id, api_key, api_secret, sandbox=False,
  confirm_live=False, exchange=None)`. **Lève `BrokerError` à l'init si
  `confirm_live` est False** (double sécurité avec la CLI). Ordres `create_order(
  symbol, "market", side, amount)` avec `amount` arrondi via `amount_to_precision`,
  vérification `min amount`/`min cost` des `markets`. `exchange` injectable (fake)
  pour les tests. Jamais de retrait, jamais de levier. Règles :
  - **statut des ordres** : réponse `canceled` / `cancelled` / `rejected` / `expired`
    sans exécution → `BrokerError` (« rejeté par l'exchange ») ; annulée après exécution
    partielle → `Order(status="partial", amount=quantité exécutée)` (WARNING) ; un ordre
    `closed` avec `filled < sent` garde la quantité réellement exécutée. `Order.amount` est
    donc toujours ce qui a été exécuté, et seul ce qui a été exécuté est suivi
    (`_track_fill`) ;
  - **poussière** : `get_position` est plate si le solde est sous `limits.amount.min` **ou**
    si `solde × dernier prix < limits.cost.min` (invendable) ; le ticker n'est consulté que
    si le solde est non nul et que `cost.min` existe ;
  - **achat au marché avec prix** : si `exchange.options["createMarketBuyOrderRequiresPrice"]`
    est vrai (Coinbase, Gate, HTX…), `create_order(symbol, "market", "buy", amount, prix)`
    reçoit le prix de référence (ccxt convertit en montant à dépenser) ; les ventes et les
    autres exchanges n'envoient pas de prix ;
  - **secrets** : `scrub_secrets(text, secrets) -> str` masque (`***`) la clé et le secret
    API, les paramètres d'URL sensibles (`AccessKeyId=`, `Signature=`, `apiKey=`…) et
    tronque la chaîne de requête de toute URL. Appliqué à chaque `BrokerError` construite
    depuis une `ccxt.BaseError` **et** aux `args` de l'exception d'origine (gardée en
    `__cause__`) : ni le journal, ni Telegram, ni `--once`, ni une trace debug ne révèlent
    la clé. `CcxtBroker.scrub(text)` expose la même fonction avec les secrets du broker ;
    la clé n'est stockée que dans l'attribut privé `_secrets`.

## `tradingbot/live/`

- `runner.py` :

```python
class LiveTrader:
    def __init__(self, strategy, broker, symbol, timeframe, risk: RiskManager,
                 data_source: Callable[[int], pd.DataFrame],  # n_bars -> OHLCV clôturées
                 state_dir="state", lookback=500, notifier=None, allow_short=False,
                 sleep=time.sleep, clock=None,     # sleep / clock injectables (tests)
                 adopt_existing_position=False,    # solde déjà présent : confié au robot ?
                 reentry_after_stop="wait",        # "wait" | "immediate" (comme le moteur)
                 lock_state_dir=True)              # verrou exclusif state_dir/lock
    def step(self, now=None) -> dict   # une itération (voir l'ordre ci-dessous) ; retourne
                                       # un dict d'état {"timestamp", "now", "symbol",
                                       # "signal", "action", "position", "amount",
                                       # "entry_price", "stop_loss", "take_profit",
                                       # "price", "equity", "reason", "order"} ;
                                       # idempotent : ne re-traite pas une bougie déjà
                                       # traitée (mémorisée dans state_dir/last_bar.json)
                                       # ni une page de bougies plus ancienne (source en
                                       # retard → "skip" + WARNING, last_bar ne recule pas)
    def run(self, poll_seconds=None, max_iterations=None) -> None
        # boucle ; s'arrête sur kill switch / max_iterations ; sur Ctrl-C : journal,
        # position mémorisée, verrou libéré, puis KeyboardInterrupt RELANCÉE (la CLI → 130)
    def close(self) -> None            # libère le verrou (aussi via `with trader:`)
    blocked_side: int                  # côté bloqué (politique "wait"), persisté
    unmanaged_amount: float            # solde présent sur le compte mais non géré
    daily_limit_active -> bool         # False si timeframe >= 1 jour

class StateDirLock:                    # verrou inter-processus (fcntl.flock / msvcrt)
    acquire() / release() / held       # acquire → StateLockedError si déjà pris
class StateLockedError(RuntimeError)   # message FR ; la CLI doit le mapper au code 2
```

  Ordre d'un `step` (le même que le moteur, bougie par bougie) :
  1. prix du broker + synchronisation de la position ; **stop / objectif vérifiés à chaque
     appel, avant toute récupération de bougies** (une panne de la source de données ne
     désarme jamais le stop) ; la clôture ferme la **quantité du robot** (jamais plus que
     le solde du compte) ; un ordre rejeté / partiel laisse l'enregistrement (et son stop)
     en place, met à jour la quantité restante et renvoie `error` (nouvel essai au prochain
     appel) ;
  2. **kill switch / limite journalière évalués à chaque appel** (`risk.update(now,
     équité)`), pas seulement à la clôture d'une bougie : position ouverte → fermée
     (`kill_switch` / `daily_limit`) ;
  3. bougies clôturées (`data_source`), idempotence, chauffe ;
  4. signal de la dernière bougie clôturée puis rebalance. **Jamais d'entrée sans stop** :
     `risk.protective_stop(signal, prix, stop de la stratégie, ATR(14) des bougies
     clôturées)` puis `risk.position_size(équité, prix, stop)` ; un achat est en plus
     plafonné par `cash × (1 - 0,5 %)` (frais + glissement). En position, un stop de la
     stratégie du mauvais côté du prix est ignoré. Une entrée partiellement exécutée
     enregistre la quantité réellement exécutée (WARNING).

  Ré-entrée (`reentry_after_stop`) : avec `"wait"` (défaut), une sortie décidée par le
  robot (`stop_loss`, `take_profit`, `daily_limit`) bloque ce côté (`blocked_side`,
  persisté dans `position.json`) tant que le **signal brut** d'une bougie clôturée
  ultérieure ne l'a pas quitté (0 ou côté opposé) ; une nouvelle transition 0→côté ou un
  retournement ouvre à nouveau (`hold` avec « ré-entrée … bloquée » en attendant).
  `"immediate"` : ré-ouverture dès la prochaine bougie clôturée traitée si le signal le
  demande encore.

  Position existante : si le compte détient déjà l'actif de base alors que le robot est
  plat, elle n'est **pas** adoptée par défaut (WARNING une fois, `unmanaged_amount`) : le
  robot ne la vend pas, ne la protège pas et **refuse d'ouvrir** (`refused`, « position
  existante non gérée par le robot ») tant qu'elle est là ; `adopt_existing_position=True`
  (CLI : `--adopt-position`) la confie au robot (sans stop jusqu'au prochain stop de la
  stratégie). Le robot ne possède jamais plus que le solde du compte, mais peut posséder
  moins (achat manuel en plus) : il ne clôture que sa propre quantité.

  Limite journalière : `timeframe >= 1 jour` → `risk.set_daily_limit_active(False,
  "bougies >= 1 jour")` à la construction, INFO « limite de perte journalière désactivée :
  bougies >= 1 jour » une fois ; `daily_limit_active` l'expose à la CLI.

  `action` ∈ {`open`, `close`, `reverse`, `hold`, `skip`, `stop_loss`, `take_profit`,
  `kill_switch`, `daily_limit`, `refused`, `error`}. Fichiers d'état dans `state_dir` :
  `last_bar.json`, `position.json` (côté, taille, prix d'entrée, stop, objectif,
  `blocked_side`), `risk.json` (état du `RiskManager`, pour qu'un redémarrage ne remette
  pas le kill switch à zéro) et `lock` (PID du robot qui tient le verrou ; un second robot
  sur le même répertoire → `StateLockedError`). Chaque action (entrée, sortie, refus par
  le risk manager, erreur) est loggée et envoyée au `notifier` s'il existe.
  `poll_seconds=None` → `default_poll_seconds(timeframe)` (durée de bougie / 12, ≥ 10 s).
- `notifier.py` : `Notifier` (interface `send(text) -> bool`),
  `LogNotifier(level=logging.INFO, logger_name=None)`, `MultiNotifier(notifiers)` (diffuse
  à plusieurs), `TelegramNotifier(token, chat_id, session=None, timeout=10.0)` (session
  type `requests` injectable, `urllib` par défaut, timeouts, jamais d'exception
  propagée : une notification qui échoue ne doit pas arrêter le bot).

## `tradingbot/config.py`

```python
@dataclass
class Settings:  # miroir de config.example.toml, avec sous-dataclasses
    general: GeneralSettings ; strategy: StrategySettings ; risk: RiskConfig ;
    costs: CostSettings ; notifications: NotificationSettings
    initial_cash: float = 1000.0          # [risk] initial_cash (hors RiskConfig)
    api_key: str | None ; api_secret: str | None ; telegram_token/chat_id
    is_live / has_api_keys / has_telegram # propriétés ; to_dict() masque les secrets

load_settings(path="config.toml", env=None) -> Settings   # TOML via tomllib ; valeurs
                                                          # manquantes = défauts ;
                                                          # lit `.env` (format KEY=VALUE)
                                                          # puis os.environ (prioritaire)
load_dotenv(path=".env") -> dict[str, str]                # sans dépendance externe
parse_param_overrides(items) -> dict[str, Any]            # ["fast=10", "x=1.5"] → dict ;
                                                          # ValueError si format invalide
parse_scalar(text) -> Any                                 # "10"→10, "1.5"→1.5, "true"→True
```

## `tradingbot/cli.py`

`python -m tradingbot <commande>` (ou `tradingbot <commande>` après `pip install -e .`).
Toutes les sorties utilisateur en français, `argparse`, code de sortie ≠ 0 en erreur.

| Commande | Rôle |
|---|---|
| `list-strategies` | nom, description et paramètres par défaut de chaque stratégie |
| `demo [--bars 3000] [--seed 42] [--report-dir]` | backtest de toutes les stratégies sur données synthétiques, sans réseau (premier contact) |
| `download [--exchange --symbol --timeframe] [--since --until] [--out] [--cache-dir --refresh]` | télécharge et met en cache |
| `backtest --strategy S (--data fichier.csv \| --exchange E --symbol X --timeframe T [--since]) [--param k=v ...] [--initial-cash --fee --slippage --allow-short] [--report-dir]` | backtest + rapport |
| `compare (--data ... \| --exchange ...) [--strategies a,b] [--report-dir]` | toutes les stratégies vs buy & hold |
| `optimize --strategy S ... [--train-bars --test-bars --metric --anchored --max-combinations] [--report-dir]` | walk-forward (fenêtres par défaut : `default_windows`) |
| `paper [--strategy S --exchange --symbol --timeframe] [--once] [--poll] [--lookback --state-dir --max-iterations] [--initial-cash --fee --slippage]` | paper trading temps réel sur données publiques |
| `live ... --i-understand-the-risks [--yes] [--sandbox]` | trading réel ; exige `mode = "live"` dans config.toml, le flag, et les clés dans `.env` ; affiche un récapitulatif et demande de taper `OUI` |

Options communes : `--config` (défaut `config.toml`, facultatif : valeurs par défaut s'il
n'existe pas) et `--log-level`. `--exchange/--symbol/--timeframe/--strategy` retombent sur
`config.toml` quand ils sont omis ; `--param` accepte `k=v` avec conversion automatique
int/float/bool et surcharge `[strategy.params]`.

Codes de sortie : `0` succès, `1` erreur (fichier introuvable, stratégie inconnue,
`BrokerError`…), `2` refus de sécurité de `live` ou arguments invalides, `130` Ctrl-C.
Journalisation du paquet sur stderr (niveau par défaut `warning` hors ligne, `info` pour
`download`/`paper`/`live`) ; hors ligne, un message identique n'est affiché qu'une fois.

Fichiers d'état : boucle `<state_dir>/<paper|sandbox|live>/<exchange>_<symbole>_<timeframe>/`
(`last_bar.json`, `position.json`, `risk.json`, `lock`), afin que paper, testnet et réel
ne se mélangent jamais ; le compte paper est propre à chaque boucle :
`<state_dir>/paper/<exchange>_<symbole>_<timeframe>/account.json` (deux paper tradings sur
le même symbole mais des timeframes différents ne partagent jamais un compte). Un second
`paper` / `live` sur le même répertoire d'état s'arrête (`StateLockedError`, code 2).

## Tests

`pytest` sans réseau : les objets `exchange` et `session` sont injectés (fakes). Chaque
module a son fichier `tests/test_<module>.py`. Les tests du moteur de backtest
utilisent de petites séries construites à la main dont le résultat est calculé à la
main (frais, slippage, stops), pas seulement « ça tourne ».
