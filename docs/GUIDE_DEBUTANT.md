# Guide du débutant : comprendre le trading et lire les rapports du robot

Ce guide suppose que vous n'avez **jamais** acheté ni vendu un actif financier. Il
explique, dans l'ordre : ce qu'est un marché et une bougie, ce que fait concrètement une
stratégie, pourquoi le risque passe avant le rendement, comment lire chaque chiffre que le
robot affiche, puis la liste de contrôle à cocher avant d'envisager de l'argent réel.

Prenez votre temps. Rien ici n'est urgent.

---

## 1. Le marché, en deux minutes

Une **crypto-monnaie** comme le bitcoin s'échange en continu, 24 h/24 et 7 j/7, sur des
plateformes appelées **exchanges** (Binance, Kraken, Coinbase…). Le prix affiché est
simplement celui de la dernière transaction entre un acheteur et un vendeur.

On trade des **paires** : `BTC/USDT` signifie « du bitcoin payé en USDT » (un dollar
numérique). Acheter `BTC/USDT` = échanger des USDT contre du BTC. Le robot raisonne toujours
dans la **devise de cotation** (ici l'USDT) : le capital, les gains et les pertes sont en
USDT.

### La bougie

Regarder chaque transaction serait illisible. On résume donc le prix par tranche de temps
fixe (la **bougie**, *candle*) : pour chaque heure, par exemple, on note

- `open` : le premier prix de l'heure ;
- `high` : le plus haut atteint ;
- `low` : le plus bas atteint ;
- `close` : le dernier prix de l'heure ;
- `volume` : la quantité échangée.

C'est le format **OHLCV**. La durée d'une bougie est le **timeframe** : `1m`, `15m`, `1h`,
`4h`, `1d`. Tout ce que fait le robot part de cette liste de bougies.

Un détail capital : une bougie n'est connue qu'une fois **clôturée**. Tant que l'heure
n'est pas finie, son `close` n'existe pas. Le robot ne décide donc jamais sur une bougie
en cours ; il attend sa clôture, puis agit à l'ouverture de la suivante. Cette règle
s'appelle « pas de *look-ahead* » et elle vaut en simulation comme en réel.

### Monter, descendre, ne rien faire

- Être **long** = avoir acheté. On gagne si le prix monte, on perd s'il baisse.
- Être **short** = avoir vendu quelque chose qu'on ne possède pas (emprunté) pour le
  racheter moins cher plus tard. On gagne si le prix baisse. Cela demande de la marge, du
  levier, et c'est plus dangereux : **le robot ne le fait pas par défaut**.
- Être **plat** (*flat*) = ne rien détenir. Ne rien faire est une position à part entière,
  et souvent la bonne.

---

## 2. Ce qu'est une stratégie

Une stratégie est une **règle** qui, à la clôture de chaque bougie, répond à une seule
question : *quelle position veux-je tenir pendant la prochaine bougie ?* Long (`1`),
plat (`0`) ou short (`-1`). Elle ajoute un **stop** : le prix auquel couper la position si
le marché part dans le mauvais sens. Les cinq stratégies livrées le fournissent sur
**chaque bougie** où elles sont en position (pas seulement à l'entrée), et si une stratégie
n'en donnait pas, le robot en placerait un lui-même (voir §3) : **aucune position ne s'ouvre
sans stop**.

Les cinq stratégies livrées se rangent en deux familles :

- **Suivi de tendance** (`sma_cross`, `ema_trend`, `donchian_breakout`) : « ce qui monte
  va continuer à monter ». Gagne dans les longues tendances, perd dans les marchés qui
  font du sur-place (beaucoup de faux départs).
- **Retour à la moyenne** (`rsi_reversion`, `bollinger_reversion`) : « ce qui a trop baissé
  va rebondir ». Gagne dans les marchés qui oscillent, perd quand le marché s'effondre
  sans rebondir.

Aucune n'est « la meilleure » : chacune gagne dans un régime de marché et perd dans
l'autre, et **personne ne sait quel régime vient ensuite**. Le détail de chaque stratégie,
avec ses paramètres, est dans [`STRATEGIES.md`](STRATEGIES.md).

### Les indicateurs

Les stratégies s'appuient sur des **indicateurs**, de simples calculs sur les bougies
passées :

- **Moyenne mobile** (SMA, EMA) : la moyenne des N dernières clôtures. Lisse le bruit et
  montre la direction de fond.
- **RSI** (0 à 100) : la vitesse des hausses par rapport aux baisses récentes. Sous 30, le
  prix a chuté « trop vite » ; au-dessus de 70, il a monté « trop vite ».
- **ATR** : de combien le prix bouge habituellement par bougie. Sert à placer le stop à une
  distance raisonnable (ni touché par le bruit, ni trop loin).
- **ADX** : la force de la tendance (haut = tendance nette, bas = sur-place).
- **Bandes de Bollinger** : un couloir autour de la moyenne, large quand le marché est
  nerveux, étroit quand il est calme.
- **Canal de Donchian** : le plus haut et le plus bas des N dernières bougies.

Aucun indicateur ne prédit l'avenir. Ils décrivent le passé récent, c'est tout.

---

## 3. Le risque d'abord : pourquoi la taille de position compte plus que la stratégie

Imaginez deux traders avec la même stratégie. Le premier engage tout son capital à chaque
trade ; le second risque 1 % par trade. Après une série de cinq pertes (banale, même pour
une bonne stratégie), le premier a perdu une grande partie de son capital et doit gagner
beaucoup plus pour revenir à zéro ; le second a perdu environ 5 %.

C'est la raison d'être du **gestionnaire de risque** du robot (`[risk]` dans
`config.toml`) :

1. **Un stop, toujours.** Avant d'ouvrir, le robot choisit le stop dans cet ordre : le stop
   de la **stratégie** s'il est du bon côté du prix (sous le prix pour un achat) ; sinon un
   **stop de secours** à `default_stop_atr_mult` (3) fois l'ATR des 14 dernières bougies,
   c'est-à-dire à trois « mouvements habituels » du prix ; sinon, si l'ATR n'est pas
   calculable (trop peu de bougies), un stop de **dernier recours** à `default_stop_pct`
   (5 %) du prix, avec un avertissement dans le journal. En pratique, les cinq stratégies
   livrées donnent leur stop à chaque bougie et c'est presque toujours lui qui sert ; le
   stop de secours est un filet, pas la règle.
2. **`risk_per_trade` (1 %)** : la taille de la position est calculée pour qu'un stop touché
   coûte 1 % du capital : `taille = capital × 1 % ÷ distance au stop`. Stop loin → petite
   position ; stop proche → position plus grande (plafonnée, voir le point suivant).
3. **`max_position_fraction` (95 %)** : jamais plus de 95 % du capital engagé, quoi qu'il
   arrive, et jamais de levier. **Cas fréquent à comprendre** : sur un actif peu volatil
   (une paire de devises en bougies horaires, par exemple), le stop à 3 × ATR est tout près
   du prix (0,3 %) ; la règle du 1 % demanderait alors d'acheter pour trois fois le capital.
   Le plafond s'applique : la position vaut ~95 % du compte, et le risque réellement couru
   si le stop est touché n'est que de `0,3 % × 95 % ≈ 0,3 %` du capital, **moins** que le 1 %
   visé. Le risque par trade est donc plus petit que prévu, mais presque tout le compte est
   engagé (au comptant, sans levier : la perte reste bornée par le stop). Si cela vous gêne,
   baissez `max_position_fraction`.
4. **`max_drawdown` (20 %)** : le **kill switch**. Si le capital a perdu 20 % depuis son plus
   haut, le robot ferme la position, refuse d'ouvrir quoi que ce soit et arrête la boucle,
   jusqu'à ce que *vous* remettiez le compteur à zéro (en supprimant
   `state/<mode>/<...>/risk.json` après réflexion). En paper / live, il est évalué à **chaque
   interrogation du marché** (toutes les `durée de bougie / 12`), pas seulement à la
   clôture d'une bougie ; en backtest, à chaque bougie. Entre deux interrogations, rien
   n'est surveillé : c'est une borne, pas une garantie.
5. **`daily_loss_limit` (5 %)** : à –5 % depuis le capital de début de journée (UTC), le
   robot ferme tout et attend le lendemain. Elle n'a de sens qu'en **intrajournalier** : sur
   des bougies d'un jour ou plus, chaque bougie est une nouvelle journée et une seule bougie
   à –5 % la déclencherait à tort ; elle est alors automatiquement **inactive** (le rapport
   et le récapitulatif l'indiquent ; le kill switch, lui, reste actif).
6. **Ré-entrée après une sortie forcée** (`wait`, politique par défaut) : après un stop, un
   objectif ou la limite journalière, le robot n'achète pas à nouveau dès la bougie suivante
   sous prétexte que la stratégie dit encore « long » ; il attend que le signal soit repassé
   par plat (ou se soit retourné) avant de rentrer du même côté. Sans cela, dans un marché
   qui baisse par paliers, on serait stoppé, racheté, stoppé, racheté… en payant des frais à
   chaque fois.
7. **`min_trade_notional`** : pas d'ordre ridiculement petit (les frais le mangeraient).

Le tableau des pertes à rattraper est à méditer :

| Perte | Gain nécessaire pour revenir à zéro |
|---|---|
| –10 % | +11 % |
| –20 % | +25 % |
| –50 % | +100 % |
| –80 % | +400 % |

Le kill switch à 20 % n'est pas une coquetterie : au-delà, le trou devient difficile à
combler.

### Les coûts, l'ennemi silencieux

Chaque ordre coûte des **frais** (~0,1 % au comptant) et du **glissement** (on est
exécuté un peu plus mal que le prix affiché). Un aller-retour coûte donc 0,2 à 0,3 % du
montant. Une stratégie qui fait 500 trades par an doit d'abord gagner 100 à 150 % brut
rien que pour payer ses coûts. C'est pour cela que :

- le robot inclut **toujours** frais et glissement dans ses simulations ;
- les stratégies lentes (bougies 4h, 1d) résistent mieux que les rapides (1m, 5m) ;
- un rapport avec beaucoup de petits trades et un gain minuscule est un rapport perdant.

---

## 4. Lire un rapport de backtest, ligne par ligne

Lancez par exemple :

```bash
tradingbot backtest --strategy ema_trend --exchange binance --symbol BTC/USDT --timeframe 1h --since 2023-01-01
```

### En-tête

Stratégie, paramètres, période, nombre de bougies, capital initial, coûts par ordre, puis
deux lignes sur les règles du moteur : **Stop de protection** (toujours présent, voir §3) et
**Ré-entrée après stop** (`wait`). Si la période fait moins d'un an ou moins de quelques
centaines de bougies, tout ce qui suit est peu fiable.

### Performance

- **Capital final / Rendement total** : le résultat brut. Ne regardez jamais ce chiffre
  seul.
- **CAGR** : le rendement ramené à un an. Sur une période de quelques jours, il explose
  (+8 000 %) : c'est mathématique, pas une bonne nouvelle. Fiable seulement sur plus d'un
  an.
- **Volatilité annualisée** : à quel point la courbe du capital secoue. 20 % est calme pour
  une crypto, 80 % est brutal.
- **Sharpe** : rendement ÷ volatilité. La lecture courante :
  - négatif : la stratégie perd ;
  - 0 à 1 : gain faible par rapport aux secousses ; médiocre ;
  - 1 à 2 : bon ;
  - au-delà de 2 sur plusieurs années : méfiance, c'est presque toujours trop beau
    (sur-apprentissage, erreur, ou période exceptionnelle).
- **Sortino** : comme Sharpe mais ne compte que les secousses vers le bas (celles qui font
  mal). Même échelle.
- **Calmar** : CAGR ÷ drawdown maximal. Au-dessus de 1, le gain annuel dépasse la pire
  baisse.
- **Drawdown maximal** : la pire baisse depuis un plus haut. **Le chiffre le plus
  important pour un débutant.** Demandez-vous honnêtement : si mon capital passait de
  1 000 à 750, est-ce que je continuerais à faire tourner le robot ? Si non, cette stratégie
  (ou ce réglage de risque) n'est pas pour vous.
- **Durée max du drawdown** : combien de bougies passées sous le dernier plus haut. 2 000
  bougies horaires, c'est presque trois mois sans nouveau sommet. Tiendriez-vous ?
- **Exposition** : part du temps passé en position. 30 % signifie que 70 % du temps
  l'argent dort (et n'est pas exposé au marché, ce qui n'est pas un défaut).

### Trades

- **Nombre de trades** : en dessous de 30, les statistiques ne veulent rien dire (le hasard
  domine). En dessous de 100, restez prudent.
- **Taux de réussite** : part des trades gagnants. **Un taux faible n'est pas un problème en
  soi** : les stratégies de tendance gagnent 30 à 45 % du temps mais leurs gains sont bien
  plus gros que leurs pertes.
- **Profit factor** : somme des gains ÷ somme des pertes. 1 = on fait du sur-place (et on a
  payé les frais pour rien). 1,3 est honnête, 2 est très bon, 5 est suspect.
- **Gain moyen / perte moyenne / meilleur / pire trade** : si un seul trade fait toute la
  performance, elle ne tient qu'à un fil.
- **Durée moyenne** : des trades de deux bougies en 1h, c'est une stratégie de bruit.
- **Frais totaux** : comparez-les au gain. S'ils en représentent la moitié, la stratégie
  travaille pour l'exchange.

### Comparaison avec le buy & hold

Le **buy & hold** (acheter au début, ne rien faire) est l'adversaire à battre. Pour que la
comparaison soit équitable, il est mesuré **sur la même période tradable** que la stratégie
(achat à la première bougie après sa période de chauffe, revente à la dernière clôture) et
**frais et glissement inclus**, comme un trade ; le rapport le rappelle sous la comparaison.
Beaucoup de stratégies perdent contre lui dans un marché haussier, tout en ayant un drawdown
plus faible. C'est un vrai arbitrage : moins gagner, mais moins souffrir. À vous de voir ce
que vous recherchez ; mais une stratégie qui fait moins bien **et** souffre plus est à
jeter.

### Garde-fous déclenchés

Cette section dit, en une phrase par événement, si le kill switch ou la limite de perte
journalière sont intervenus : « Kill switch le 2021-05-19 04:00 (capital 802.00) : le robot
a cessé de trader le 2021-05-19 04:00 après une baisse de 20.20 % depuis le sommet, et n'a
plus rien ouvert jusqu'à la fin du backtest… ». C'est capital pour
lire le reste : une stratégie arrêtée par le kill switch au tiers de la période affiche un
rendement qui ne couvre que ce tiers ; les deux autres tiers, elle n'a rien fait. Dans le
tableau de `compare`, ces stratégies portent la marque `⚠ arrêt`. Sur des bougies
journalières ou plus longues, la section précise que la limite journalière était inactive
(§3).

### Les pires et les meilleurs trades

Regardez la colonne **Raison** des pires trades :

- `stop` : le stop a fait son travail (la perte est restée bornée à ~`risk_per_trade`) ;
- `signal` : la stratégie a changé d'avis en perte ;
- `objectif` : le take-profit a encaissé ;
- `kill switch` / `limite journ.` : le gestionnaire de risque a coupé ;
- `fin` : position fermée à la dernière bougie du backtest.

Si les pires pertes dépassent largement `risk_per_trade`, c'est qu'un **gap** (le prix a
sauté par-dessus le stop à l'ouverture d'une bougie) est passé par là : cela arrive, et
c'est pour cela qu'on ne met pas tout son argent sur une position.

### Lecture (pour débutant)

Le dernier paragraphe redit tout cela en phrases simples. Lisez-le en premier si vous êtes
pressé, mais revenez aux chiffres.

---

## 5. Lire `compare` et `optimize`

### `compare`

Un tableau, une ligne par stratégie, trié par Sharpe, plus la ligne buy & hold (sur la
période tradable commune à toutes, après la plus longue chauffe, frais inclus). Un nom
suivi de `⚠ arrêt` a été stoppé par le kill switch en cours de route ; `⚠ limite journ. ×N`
compte les jours bloqués par la limite journalière ; les notes sous le tableau l'expliquent.
Ne retenez pas « la première » : retenez celles qui combinent un Sharpe > 1, un drawdown
que vous tolérez, un nombre de trades suffisant et aucun `⚠`. Puis passez-les au
walk-forward.

### `optimize` (walk-forward) : le détecteur de sur-apprentissage

Le piège numéro un du trading algorithmique : avec assez de paramètres, on trouve
**toujours** un réglage qui aurait gagné dans le passé. Il perdra demain, parce qu'il a
appris le bruit, pas une régularité du marché.

Le walk-forward joue le jeu honnêtement : il règle les paramètres sur une fenêtre
(*in-sample*, IS), les applique tels quels sur la fenêtre suivante (*out-of-sample*,
OOS), avance, recommence. Dans le rapport :

- la **ligne par fenêtre** montre les paramètres retenus et la métrique IS puis OOS. Des
  paramètres qui changent du tout au tout d'une fenêtre à l'autre sont un mauvais signe
  (la stratégie n'a pas de « bon » réglage stable) ;
- **Équité OOS** : la courbe de ce que vous auriez vraiment vécu ;
- **Moyenne IS / Moyenne OOS / Ratio OOS ÷ IS** : le ratio de sur-apprentissage (les
  moyennes sont pondérées par le nombre de bougies de chaque fenêtre de test). Lecture, la
  même partout dans le robot et ses documents : **bas = sur-apprentissage**.
  - proche de 1 (0,7 à 1,2) : robuste ;
  - entre 0,5 et 0,7 : acceptable mais fragile (c'est le minimum de la liste de contrôle
    du §8) ;
  - nettement inférieur (0,3) : les beaux chiffres de l'entraînement étaient du bruit ;
  - nul ou négatif : la stratégie perd hors échantillon. Abandonnez ;
  - supérieur à 1 : mieux hors échantillon qu'en entraînement, rare, souvent le hasard ;
  - `n/d` : moyenne IS nulle, infinie ou de signe opposé à la moyenne OOS — le ratio n'a
    pas de sens (par exemple la stratégie ne gagnait même pas sur les données
    d'entraînement). Pour une métrique négative comme `max_drawdown` (IS et OOS tous deux
    négatifs), le ratio vaut |IS| / |OOS| et se lit de la même façon.
- Le compte rendu signale aussi, après coup, si la courbe OOS enchaînée aurait déclenché le
  **kill switch** en réel (« ⚠ En réel, le kill switch … aurait arrêté le robot le … ») et
  rappelle que la position est fermée à la fin de chaque fenêtre de test (les fenêtres
  courtes pénalisent les stratégies qui gardent longtemps leurs positions).

Une règle simple : **ne mettez jamais en paper, et encore moins en réel, une stratégie
dont la courbe OOS perd de l'argent ou dont le ratio est mauvais**, quel que soit son
backtest.

---

## 6. Lire l'état du paper trading

`tradingbot paper ... --once` affiche un JSON. Les clés qui comptent :

| Clé | Sens |
|---|---|
| `timestamp` | la dernière bougie clôturée qui a été traitée |
| `signal` | ce que la stratégie souhaite : `1` long, `0` plat, `-1` short (ignoré sans `allow_short`) |
| `action` | ce que le robot a fait : `open` (entrée), `close` (sortie), `reverse`, `hold` (rien à changer), `skip` (bougie déjà traitée, chauffe, taille trop petite), `stop_loss`, `take_profit`, `kill_switch`, `daily_limit`, `refused` (le gestionnaire de risque a dit non), `error` |
| `position` | position tenue après l'itération (`1`, `0`, `-1`) et `amount` sa taille |
| `entry_price`, `stop_loss`, `take_profit` | prix d'entrée et niveaux de protection actifs |
| `equity` | valeur du compte fictif |
| `reason` | explication en français |

Deux comportements à connaître :

- **Idempotence** : relancer sur la même bougie donne `skip` / « bougie déjà traitée ».
  Tout l'état d'une boucle est dans **un seul dossier**
  `state/paper/<exchange>_<symbole>_<timeframe>/` : `last_bar.json`, `position.json` (côté,
  taille, stop, objectif, côté bloqué par la politique de ré-entrée), `risk.json` et le compte
  fictif `account.json`. Deux boucles sur le même symbole mais des timeframes différents ont
  donc chacune leur compte. Pour repartir de zéro, supprimez ce dossier.
- **Un seul robot par dossier** : le fichier `lock` du dossier est verrouillé par le robot
  qui tourne. Une seconde commande `paper` (ou `live`) sur le même dossier s'arrête tout de
  suite (« REFUS : le répertoire d'état … est déjà utilisé par un autre robot (PID …) »,
  code de sortie 2) : impossible que deux processus se partagent un compte sans le savoir.
- **Code de sortie de `--once`** : `0` si l'itération s'est bien passée (y compris `hold`,
  `skip`, `refused`), `1` si elle s'est terminée en `error` (exchange injoignable, ordre
  refusé) : utile dans un script.
- **Démarrage au milieu d'une tendance** : si la stratégie est déjà « long » depuis
  longtemps quand vous lancez le robot, il ouvre la position immédiatement — **avec un
  stop** et une taille basée sur le risque (§3) : toutes les stratégies livrées portent leur
  stop sur chaque bougie où elles sont en position, et `ema_trend` (la stratégie par défaut
  de `config.example.toml`) fournit même un stop suiveur qui monte avec le prix. Si une
  stratégie ne donnait pas de stop, le stop de secours à 3 × ATR s'appliquerait. C'est quand
  même une raison de commencer en paper : entrer tard dans une tendance, c'est souvent
  entrer juste avant le repli.
- **Position existante sur le compte** : si le compte détient déjà l'actif (achat manuel,
  ancien robot) alors que `position.json` est vide, le robot **ne le vend pas**, ne le
  protège pas et refuse d'ouvrir (`refused`, « position existante non gérée par le robot »)
  tant qu'il est là ; le récapitulatif l'affiche sur la ligne « Position existante ».
  Relancez avec `--adopt-position` pour lui confier cette position (sans stop jusqu'au
  prochain stop de la stratégie), ou vendez-la à la main.
- **Après un stop** : la stratégie peut dire encore « long » à la bougie suivante ; le robot
  attend tout de même que le signal soit repassé par plat (politique `wait`, §3). La raison
  « ré-entrée longue bloquée après une sortie forcée » dans le JSON est normale.

---

## 7. Les erreurs classiques du débutant

1. **Croire un backtest.** Un backtest est une hypothèse, pas une promesse.
2. **Optimiser jusqu'à ce que ça marche.** C'est exactement la définition du
   sur-apprentissage.
3. **Tester sur une seule période.** 2020-2021 (hausse continue) ne dit rien sur 2022
   (baisse). Testez sur plusieurs années, plusieurs actifs.
4. **Mettre les frais à zéro « pour voir ».** Vous verrez une stratégie qui n'existe pas.
5. **Augmenter la taille après une perte pour « se refaire ».** C'est ainsi qu'on perd tout.
6. **Désactiver le stop ou le kill switch parce qu'ils « coûtent des trades ».** Ils coûtent
   des petits trades pour éviter la grosse perte.
7. **Intervenir à la main pendant que le robot tourne.** Soit vous lui faites confiance,
   soit vous l'arrêtez (`Ctrl-C`) ; les deux à la fois produisent le pire des deux.
8. **Trader un timeframe court parce que « ça bouge plus ».** Ça coûte surtout plus.
9. **Passer en réel après trois jours de paper.** Trois jours, c'est du hasard. Des
   semaines, c'est un début d'information.
10. **Oublier les impôts.** Les plus-values sur crypto sont imposables dans la plupart des
    pays ; tenez un journal (le robot écrit `trades.csv`).

---

## 8. Liste de contrôle avant de passer en réel

Ne cochez une case que si c'est vraiment vrai.

**La stratégie**

- [ ] Backtestée sur **au moins deux ans** de données, incluant un marché baissier.
- [ ] Walk-forward : courbe OOS positive **et** ratio OOS/IS raisonnable (≥ 0,5 environ,
  proche de 1 de préférence ; voir §5) **et** aucun kill switch signalé dans le compte rendu.
- [ ] Au moins 100 trades dans le backtest.
- [ ] Le drawdown maximal du backtest est un chiffre que je **tolérerais** en réel, et la
  section « Garde-fous déclenchés » du rapport est vide (pas de kill switch).
- [ ] Les frais totaux restent une petite fraction du gain.
- [ ] Je comprends **pourquoi** elle gagne et **quand** elle perd (voir `STRATEGIES.md`).

**Le paper trading**

- [ ] Au moins **quatre semaines** de paper trading continu, sans intervention manuelle.
- [ ] Le comportement observé ressemble au backtest sur la même période (nombre de
  trades, gains et pertes typiques).
- [ ] J'ai vu le robot gérer un stop, un redémarrage, et de préférence une coupure réseau.

**Le compte et la sécurité** (détails dans [`PASSER_EN_REEL.md`](PASSER_EN_REEL.md))

- [ ] Clés API avec le droit **trade** seulement, **jamais** le retrait.
- [ ] Clés restreintes à mon adresse IP, stockées dans `.env` (jamais dans git).
- [ ] Testé sur le testnet/sandbox de l'exchange quand il existe.
- [ ] Le compte ne contient **que** le capital que j'accepte de perdre, et rien d'autre.

**Le risque**

- [ ] `risk_per_trade` ≤ 1 %, `max_drawdown` ≤ 20 %, `daily_loss_limit` ≤ 5 %.
- [ ] `allow_short = false` (sauf si je sais exactement ce que je fais).
- [ ] Je sais arrêter le robot (`Ctrl-C`) et fermer la position à la main sur l'exchange.
- [ ] Je sais ce que fait le kill switch et comment le réarmer.

**Moi**

- [ ] Je n'ai pas emprunté cet argent, et sa perte ne changerait rien à ma vie.
- [ ] Je ne vais pas regarder le compte toutes les cinq minutes.
- [ ] Je tiens un journal de ce que je fais et pourquoi.

Si une seule case reste vide, restez en paper. Le marché sera toujours là plus tard.
