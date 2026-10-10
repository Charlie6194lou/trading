# tradingbot — un robot de trading pour apprendre, sans se ruiner

> Vous n'avez jamais tradé ? C'est précisément pour vous que ce projet est écrit. Lisez ce
> fichier en entier avant de lancer quoi que ce soit : il ne contient aucune formule magique,
> seulement ce qu'il faut savoir pour ne pas perdre d'argent bêtement.

## 1. Ce que ce robot est… et ce qu'il n'est pas

**Ce qu'il est.** Un programme Python qui :

- télécharge l'historique des prix d'une crypto-monnaie (via la bibliothèque `ccxt`) ;
- **simule** cinq stratégies de trading classiques sur cet historique (*backtest*), frais et
  glissement compris, et vous explique les résultats en français ;
- vérifie honnêtement si une stratégie « tient la route » hors de l'échantillon sur lequel
  elle a été réglée (*walk-forward*) ;
- trade **avec de l'argent fictif** en temps réel sur les vrais prix (*paper trading*) ;
- peut, si et seulement si vous le demandez trois fois de suite, trader avec de l'argent
  réel, avec des garde-fous (taille des positions, stop, kill switch, limite de perte
  journalière).

**Ce qu'il n'est pas.** Une machine à gagner de l'argent. Soyons directs :

- **Aucun robot ne garantit de profit.** Les marchés changent de régime sans prévenir ; une
  stratégie qui gagnait l'an dernier peut perdre toute l'année prochaine.
- **La plupart des stratégies simples perdent contre le simple « acheter et garder »
  (buy & hold) une fois les frais déduits.** Le robot vous le montrera noir sur blanc dans
  ses rapports : c'est une fonctionnalité, pas un bug.
- Un backtest brillant est le plus souvent le signe d'un sur-apprentissage (la stratégie a
  « appris par cœur » le passé) ou d'une erreur de programmation.

C'est un **outil d'apprentissage** livré avec un réglage par défaut sûr : mode `paper`
(argent fictif), pas de vente à découvert, pas de levier, **toujours un stop** sur chaque
position et 1 % du capital risqué par trade (la taille de la position est calculée à partir de
la distance au stop ; sur un actif peu volatil, c'est le plafond de 95 % du capital qui borne la
taille, et le risque effectif est alors *inférieur* à 1 %, voir le guide du débutant).

## 2. Les 5 règles d'or

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ 1. N'investissez JAMAIS un argent dont la perte totale changerait votre vie.     │
│ 2. Faites du paper trading (argent fictif) pendant des SEMAINES avant le réel.   │
│ 3. Commencez petit : un montant que vous accepteriez de voir disparaître.        │
│ 4. Toujours un stop (le robot en met un, même si la stratégie n'en donne pas).  │
│    Toujours un kill switch (activé par défaut).                                  │
│ 5. Pas de levier, pas d'emprunt, pas de « rattrapage » après une perte.          │
└──────────────────────────────────────────────────────────────────────────────────┘
```

## 3. Installation

Prérequis : Python 3.11 ou plus récent (`python --version`).

```bash
git clone <adresse du dépôt> trading
cd trading
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -e ".[dev]"          # le robot + les outils de test
```

Vérifiez l'installation :

```bash
tradingbot --version
tradingbot --help
```

Si la commande `tradingbot` n'est pas trouvée, `python -m tradingbot` fait exactement la
même chose. Pour les graphiques (optionnels) : `pip install matplotlib`.

## 4. Démarrage rapide : les huit étapes

Chaque étape est une commande, suivie de **comment lire ce qui s'affiche**. Suivez l'ordre.

### Étape 1 — `demo` : voir le robot tourner, sans réseau

```bash
tradingbot demo
```

Génère 3 000 bougies horaires **synthétiques** (tirées au hasard), lance les cinq
stratégies dessus et affiche un tableau comparatif. Le message d'avertissement à la fin
(« ATTENTION : ces bougies sont SYNTHÉTIQUES… ») est important : ces données ne prouvent
rien. La démo sert juste à vérifier que tout fonctionne.

**Attendez-vous à du rouge.** Avec la graine par défaut (`--seed 42`), le marché synthétique
perd environ 70 % sur la période : il est normal que toutes les stratégies affichent des
rendements et des Sharpe négatifs. Seule la ligne « Buy & hold » sert de repère : une
stratégie à –10 % quand le marché fait –70 % a *perdu moins*, elle n'a pas gagné. Essayez
`tradingbot demo --seed 7` pour un autre régime de marché.

```
Stratégie           | Rendement |   CAGR  | Sharpe |  Max DD  | Trades | Réussite
...
Buy & hold          |   ...     |   ...   |  ...   |   ...    |      1 |   –
```

- **Rendement** : variation du capital sur la période (+10 % = 1 000 → 1 100).
- **CAGR** : le même rendement ramené à un an (attention, sur une période courte il
  devient vite absurde : c'est normal).
- **Sharpe** : gain par unité de risque. Grossièrement : < 0 la stratégie perd, 0–1 médiocre,
  1–2 bon, > 2 suspect (trop beau).
- **Max DD** (*drawdown* maximal) : la pire baisse du capital depuis son plus haut.
  –25 % signifie qu'à un moment vous auriez vu un quart de votre argent disparaître.
  C'est le chiffre qui dit si vous *tiendrez* psychologiquement.
- **Trades** et **Réussite** : nombre d'allers-retours et part des trades gagnants.
- **Buy & hold** : ce qu'aurait donné un simple achat au début, conservé jusqu'à la fin —
  mesuré sur la **même période que les stratégies** (après leur période de chauffe, c'est-à-dire
  à partir de la première bougie où elles pouvaient trader) et **frais et glissement inclus**,
  comme un trade. Une stratégie qui ne fait pas mieux n'apporte rien.
- Un nom suivi de **`⚠ arrêt`** signale que le kill switch a arrêté cette stratégie en cours de
  route (son rendement s'arrête donc là) ; `⚠ limite journ. ×N` compte les jours où la limite
  de perte journalière l'a bloquée. Une note sous le tableau l'explique.

### Étape 2 — `list-strategies` : les stratégies disponibles

```bash
tradingbot list-strategies
```

Affiche le nom (à utiliser dans `--strategy`), une description en une phrase et les
paramètres par défaut. Pour comprendre chacune : [`docs/STRATEGIES.md`](docs/STRATEGIES.md).

### Étape 3 — `download` : de vraies bougies

```bash
tradingbot download --exchange binance --symbol BTC/USDT --timeframe 1h --since 2023-01-01
```

Télécharge les bougies horaires de BTC/USDT depuis le 1er janvier 2023 (données publiques,
aucune clé nécessaire) et les met en cache dans `data/cache/binance_BTC-USDT_1h.csv`. Les
commandes suivantes réutiliseront ce cache. Options : `--out fichier.csv` pour une copie,
`--refresh` pour tout retélécharger, `--until 2024-12-31` pour borner.

Sans `--since`, l'exchange ne renvoie que les ~1 000 dernières bougies : trop peu pour un
backtest sérieux. Visez **au moins deux ans** de données.

**Le cache et `--since`.** Si un cache existe déjà (par exemple parce que vous avez lancé un
`backtest` ou un `compare` avant cette étape, avec seulement ~1 000 bougies), il n'est pas
ignoré : s'il **couvre** la période demandée, il est réutilisé tel quel et rien n'est
téléchargé ; si `--since` est antérieur à sa première bougie (ou `--until` postérieur à la
dernière), seul le **complément manquant** est téléchargé puis fusionné. La commande dit
toujours ce qu'elle a fait (« cache réutilisé tel quel, aucun téléchargement », « cache
complété / mis à jour ») : lisez cette ligne et vérifiez la période affichée. Si l'exchange
ne fournit pas plus ancien (Kraken ne donne que les 720 dernières bougies), un avertissement
le signale et le cache reste inchangé. `--refresh` force un téléchargement complet.

### Étape 4 — `backtest` : une stratégie, un rapport complet

```bash
tradingbot backtest --strategy ema_trend --exchange binance --symbol BTC/USDT --timeframe 1h
# ou sur un CSV à vous :
tradingbot backtest --strategy ema_trend --data mes_bougies.csv --report-dir reports/ema
```

Le rapport est long, volontairement. Les parties à lire :

- **Performance** : capital final, rendement, Sharpe, Sortino, drawdown maximal et sa
  **durée** (combien de bougies passées sous le dernier plus haut : des mois sans gagner,
  ça arrive).
- **Trades** : nombre de trades, **taux de réussite** (part des trades gagnants) et
  **profit factor** (somme des gains ÷ somme des pertes ; > 1 la stratégie gagne plus
  qu'elle ne perd ; < 1.2 c'est fragile). Un taux de réussite de 35 % peut être excellent
  si les gains sont trois fois plus gros que les pertes : les deux chiffres vont ensemble.
- **Comparaison avec le buy & hold** et le verdict en une phrase. Le buy & hold est mesuré
  sur la **même période tradable** (à partir de la première bougie après la chauffe de la
  stratégie) et **frais inclus** : la comparaison est équitable.
- **Garde-fous déclenchés** : la date et le capital de chaque déclenchement du kill switch
  (« le robot a cessé de trader le … après une baisse de X % depuis le sommet ») ou de la
  limite de perte journalière. Un rendement de +4 % avec un kill switch déclenché après six
  mois signifie que la stratégie n'a *plus rien fait* pendant tout le reste de la période :
  lisez cette section avant le rendement. Sur des bougies journalières ou plus longues, elle
  indique que la limite journalière était **inactive** (voir §5).
- **Les 5 pires / 5 meilleurs trades** : regardez surtout les pires et leur raison de
  sortie (`stop`, `signal`, `kill switch`, `limite journ.`…).
- **Lecture (pour débutant)** : le paragraphe qui traduit tout cela en français courant.

L'en-tête rappelle deux réglages du moteur : **Stop de protection** (toujours présent : le
stop de la stratégie, sinon un stop de secours à `3 × ATR(14)`, sinon 5 % du prix ; la taille
de chaque position en découle) et **Ré-entrée après stop** (`wait` : après un stop, un
objectif ou la limite journalière, le robot attend que le signal soit repassé par 0 avant de
rentrer du même côté, au lieu de racheter dès la bougie suivante).

Avec `--report-dir`, le robot écrit `equity.csv` (courbe du capital), `trades.csv`,
`summary.txt`, `metrics.json` et `equity.png` si matplotlib est installé.

Changer un paramètre : `--param fast=10 --param slow=30` (les noms sont dans
`list-strategies`). Changer les coûts : `--initial-cash 5000 --fee 0.001 --slippage 0.0005`.

### Étape 5 — `compare` : toutes les stratégies face au marché

```bash
tradingbot compare --exchange binance --symbol BTC/USDT --timeframe 1h
```

Même tableau que la démo, mais sur de vraies données. Attendez-vous à voir plusieurs
stratégies **sous** la ligne buy & hold : c'est la réalité. Celles qui ressortent avec un
Sharpe > 1 **et** un drawdown supportable **et** sans `⚠ arrêt` (kill switch déclenché)
méritent l'étape suivante.

### Étape 6 — `optimize` : le test de vérité (walk-forward)

```bash
tradingbot optimize --strategy ema_trend --exchange binance --symbol BTC/USDT --timeframe 1h
```

Le robot découpe l'historique en fenêtres. Sur chaque fenêtre d'**entraînement**, il
choisit les meilleurs paramètres ; puis il les applique, sans les retoucher, sur la fenêtre
de **test** qui suit (jamais vue). Il avance, et recommence. Le résultat qui compte est la
performance **hors échantillon** (OOS, *out-of-sample*).

Comment lire :

- le tableau par fenêtre : paramètres retenus, métrique IS (entraînement) et OOS (test) ;
- **Équité OOS** : ce que la stratégie aurait réellement donné en étant ré-optimisée au fil
  du temps ;
- **Ratio OOS / IS** (*overfit ratio*) : proche de 1 → la stratégie se comporte pareil sur
  des données inconnues (robuste) ; très inférieur à 1, nul ou négatif → sur-apprentissage :
  les beaux chiffres du backtest étaient du bruit. **Ne mettez jamais en paper/réel une
  stratégie dont le ratio est mauvais.**

Options : `--train-bars 4000 --test-bars 1000` (tailles des fenêtres, sinon 4 fenêtres de
test automatiques), `--metric sharpe|sortino|calmar|total_return`, `--anchored`.

### Étape 7 — `paper --once` : une itération avec de l'argent fictif

```bash
tradingbot paper --strategy ema_trend --exchange binance --symbol BTC/USDT --timeframe 1h --once
```

Le robot récupère les dernières bougies **clôturées**, calcule le signal, et passe (ou non)
un ordre sur un **compte fictif** aux vrais prix. Il affiche un récapitulatif puis l'état
sous forme JSON :

```json
{
  "timestamp": "2025-01-10T13:00:00+00:00",   // dernière bougie clôturée traitée
  "signal": 1,                                 // 1 achat souhaité, 0 rien, -1 vente
  "action": "open",                            // open / close / hold / skip / stop_loss ...
  "position": 1,                               // position tenue après l'itération
  "equity": 998.57,                            // valeur du compte fictif
  "reason": "entrée sur signal"
}
```

Relancez la même commande : `"action": "skip"`, `"bougie déjà traitée"`. Le robot ne
passe jamais deux fois le même ordre, même après un redémarrage (état dans `state/`).

Le récapitulatif affiché avant le JSON rappelle la règle du stop de protection, la politique
de ré-entrée, si la limite de perte journalière est active pour ce timeframe (elle ne l'est
pas sur des bougies d'un jour ou plus) et ce que contient déjà le compte (« Position
existante »). Code de sortie : `0`, ou `1` si l'itération s'est terminée en `"action":
"error"` (exchange injoignable, ordre refusé…) : un script ou un `cron` qui enchaîne des
`--once` peut ainsi détecter l'échec.

### Étape 8 — `paper` : la boucle, pendant des semaines

```bash
tradingbot paper --strategy ema_trend --exchange binance --symbol BTC/USDT --timeframe 1h
```

Même chose en continu : interrogation du marché toutes les `durée de bougie / 12`
(5 minutes en 1h), **stop, kill switch et limite journalière vérifiés à chaque
interrogation** (le signal, lui, n'est évalué qu'une fois par bougie clôturée), arrêt propre
par `Ctrl-C` (la position fictive est mémorisée et reprise au prochain lancement ; code de
sortie `130`).

Chaque boucle a son dossier d'état `state/paper/<exchange>_<symbole>_<timeframe>/`
(`last_bar.json`, `position.json`, `risk.json`) et son **compte fictif**
`state/paper/binance_BTC-USDT_1h/account.json` : deux paper tradings sur le même symbole
mais des timeframes différents ne partagent jamais un compte. Un fichier `lock` dans ce
dossier garantit **un seul robot à la fois** : lancer une seconde fois la même boucle (ou
deux `--once` qui se chevauchent) s'arrête net avec « REFUS : le répertoire d'état … est déjà
utilisé par un autre robot (PID …) » et le code de sortie `2`. Pour faire tourner deux
robots, donnez-leur un `--state-dir` différent.

Si le compte détient déjà l'actif (vous l'avez acheté à la main) alors que le robot est plat,
il **ne le vend pas** et refuse d'ouvrir tant qu'il est là ; `--adopt-position` lui confie
cette position (voir [`docs/PASSER_EN_REEL.md`](docs/PASSER_EN_REEL.md)).

**Laissez tourner plusieurs semaines** et comparez avec le backtest de la même période.
Si le paper trading ne ressemble pas au backtest, ne passez pas en réel.

## 5. Configuration

Copiez `config.example.toml` en `config.toml` et adaptez : exchange, symbole, timeframe,
stratégie et ses paramètres (`[strategy.params]`), capital de départ et limites de risque
(`[risk]`), frais estimés (`[costs]`), notifications Telegram. Toutes les commandes lisent
ce fichier (`--config autre.toml` pour en utiliser un autre) ; les options de la ligne de
commande ont priorité.

Les réglages de risque sont les plus importants :

| Clé | Défaut | Rôle |
|---|---|---|
| `risk_per_trade` | 0.01 | fraction du capital perdue si le stop est touché (1 %) |
| `max_position_fraction` | 0.95 | jamais plus de 95 % du capital engagé (1.0 au plus : jamais de levier) |
| `max_drawdown` | 0.20 | **kill switch** : à –20 % depuis le plus haut, tout est fermé et bloqué |
| `daily_loss_limit` | 0.05 | –5 % sur la journée UTC : fermé et bloqué jusqu'au lendemain. **Bougies < 1 jour seulement** : sur des bougies journalières ou plus longues, chaque bougie est une nouvelle journée et la limite est automatiquement **inactive** (le rapport et le récapitulatif le disent) |
| `allow_short` | false | vente à découvert désactivée (marché au comptant). `true` est **refusé par `live`** (impossible au comptant) et seulement simulé en `paper` |
| `default_stop_atr_mult` | 3.0 | stop de secours quand la stratégie n'en donne pas (ou le donne du mauvais côté du prix) : `prix ∓ 3 × ATR(14)` |
| `default_stop_pct` | 0.05 | stop de dernier recours (5 % du prix) si l'ATR est indisponible ; un avertissement est journalisé |

Une position **ne s'ouvre jamais sans stop**. Ordre de priorité : le stop de la stratégie s'il
est du bon côté du prix (toutes les stratégies livrées le portent sur **chaque** bougie où
elles sont en position, pas seulement à l'entrée) ; sinon le stop de secours à `3 × ATR` ;
sinon 5 % du prix. La taille de la position vient toujours de ce stop (`risk_per_trade`),
plafonnée par `max_position_fraction` et par le cash disponible.

Les secrets (clés API, jeton Telegram) ne vont **jamais** dans `config.toml` mais dans un
fichier `.env` (copiez `.env.example`), ignoré par git. Ils ne sont nécessaires que pour le
mode réel ou Telegram.

## 6. Passer en réel (lisez d'abord [`docs/PASSER_EN_REEL.md`](docs/PASSER_EN_REEL.md))

La commande `live` refuse de démarrer tant que **les trois conditions** ne sont pas
réunies : `mode = "live"` dans `config.toml`, le drapeau `--i-understand-the-risks`, et des
clés API dans `.env`. Elle affiche ensuite un récapitulatif (exchange, symbole, stratégie,
limites de risque, **ARGENT RÉEL**) et exige que vous tapiez `OUI`.

```bash
tradingbot live --i-understand-the-risks            # d'abord avec --sandbox si l'exchange le propose
```

`live` refuse aussi `allow_short = true` (ou `--allow-short`) : le robot ne trade qu'au
comptant, et on ne peut pas vendre ce qu'on ne détient pas.

Avant cela : des semaines de paper trading, des clés API **sans droit de retrait**,
restreintes à votre adresse IP, et un capital que vous pouvez perdre. Le guide détaille
chaque étape, comment arrêter le robot et comment fonctionne le kill switch.

## 7. Architecture

```
tradingbot/
  data/        loader.py (CSV, téléchargement ccxt, cache), synthetic.py (bougies fictives)
  indicators.py  SMA, EMA, RSI, ATR, MACD, Bollinger, Donchian, ADX, ...
  strategies/  base.py (contrat), sma_cross, ema_trend, rsi_reversion, donchian_breakout,
               bollinger_reversion
  risk.py      taille des positions, kill switch, limite journalière
  backtest/    engine.py (simulation), metrics.py, walkforward.py, report.py
  execution/   broker.py (contrat), paper.py (fictif), ccxt_broker.py (réel)
  live/        runner.py (boucle temps réel), notifier.py (journal, Telegram)
  config.py    config.toml + .env
  cli.py       les commandes ci-dessus
```

Règle absolue du projet : **aucun regard dans le futur**. Le signal calculé à la clôture
d'une bougie est exécuté à l'ouverture de la suivante, en backtest comme en réel.
Détails des contrats entre modules : [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) ;
explication des stratégies : [`docs/STRATEGIES.md`](docs/STRATEGIES.md) ; notions de
trading et lecture des rapports : [`docs/GUIDE_DEBUTANT.md`](docs/GUIDE_DEBUTANT.md).

## 8. Tests et qualité

```bash
make test        # python -m pytest  (aucun test n'accède au réseau)
make lint        # ruff check + ruff format --check
make format      # ruff format
make demo        # tradingbot demo
```

Sous Windows, `make` n'est généralement pas installé : lancez directement les commandes
équivalentes, qui sont les mêmes partout :

```
python -m pytest                      # make test
ruff check . && ruff format --check . # make lint   (PowerShell : ruff check . ; ruff format --check .)
ruff format .                         # make format
python -m tradingbot demo             # make demo
```

Les tests du moteur, des indicateurs et des stratégies comparent les résultats à des
valeurs calculées à la main (frais, slippage, stops) et vérifient l'absence de look-ahead.
L'intégration continue (`.github/workflows/ci.yml`) rejoue tout cela sur Python 3.11 et 3.12.

## 9. Glossaire

- **Bougie (candle)** : résumé du prix sur une durée fixe : ouverture, plus haut, plus bas,
  clôture, volume. Une bougie « 1h » = une heure de marché.
- **OHLCV** : *Open, High, Low, Close, Volume*, les cinq colonnes d'une bougie.
- **Timeframe** : la durée d'une bougie (1m, 15m, 1h, 4h, 1d). Plus il est court, plus il y
  a de trades, de bruit et de frais.
- **Long / short** : être *long* = avoir acheté, on gagne si le prix monte. Être *short* =
  avoir vendu à découvert, on gagne si le prix baisse (désactivé par défaut ici).
- **Range (marché sans tendance, « en sur-place », latéral)** : le prix oscille entre un
  plancher et un plafond sans direction durable. Les stratégies de tendance y perdent, celles
  de retour à la moyenne y gagnent.
- **Stop-loss (stop)** : prix auquel on coupe automatiquement une position perdante.
  La ceinture de sécurité. Ici, chaque position en a un, toujours.
- **Take-profit (objectif)** : prix auquel on encaisse automatiquement un gain.
- **Drawdown** : baisse du capital depuis son plus haut précédent. Le *max drawdown* est la
  pire de ces baisses.
- **Slippage (glissement)** : différence entre le prix affiché et le prix réellement obtenu.
  Toujours défavorable ; le robot le simule.
- **Frais** : commission prélevée par l'exchange à chaque ordre (~0,1 % au comptant).
  Un aller-retour coûte donc ~0,2 % plus le glissement.
- **Levier** : emprunter pour trader plus gros que son capital. Multiplie les gains **et**
  les pertes ; ce robot n'en utilise jamais.
- **Backtest** : simulation d'une stratégie sur le passé. Utile, mais le passé ne se
  répète pas.
- **Overfitting (sur-apprentissage)** : régler une stratégie si finement sur le passé
  qu'elle n'en décrit plus que le bruit. Se détecte avec le walk-forward.
- **Walk-forward** : optimiser sur une période, tester sur la suivante, avancer, recommencer.
- **Paper trading** : trader avec de l'argent fictif aux vrais prix. L'étape obligatoire
  avant le réel.
- **Sharpe** : rendement moyen divisé par la volatilité (le « bruit ») ; le gain par unité
  de risque.
- **Profit factor** : somme des gains divisée par la somme des pertes.
- **Kill switch** : arrêt d'urgence automatique quand la perte depuis le plus haut dépasse
  un seuil.
- **Exchange** : la plateforme où s'échangent les cryptos (Binance, Kraken, Coinbase…).
- **API / clé API** : l'accès programmatique à votre compte sur l'exchange ; une clé est
  un mot de passe pour robots, à protéger comme tel.

## 10. Avertissement

Ce logiciel est fourni **à des fins pédagogiques**, sans aucune garantie, sous licence MIT
(texte complet dans le fichier [`LICENSE`](LICENSE)).
Il ne constitue pas un conseil en investissement. Le trading de crypto-actifs est très
risqué et peut entraîner la perte totale du capital engagé. Vous êtes seul responsable
des ordres passés avec vos clés API, des réglages choisis et des conséquences financières
et fiscales. Les auteurs ne pourront être tenus responsables d'aucune perte.

Si vous avez un doute, restez en mode `paper`. Il n'y a aucune urgence à risquer de
l'argent réel : le marché sera encore là demain.
