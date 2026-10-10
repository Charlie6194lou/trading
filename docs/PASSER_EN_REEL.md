# Passer en réel, étape par étape

> Avant de lire ceci, vous devriez avoir fait tourner le robot en paper trading pendant
> plusieurs semaines et coché la liste de contrôle de [`GUIDE_DEBUTANT.md`](GUIDE_DEBUTANT.md).
> Si ce n'est pas le cas, fermez ce document et revenez plus tard. Vraiment.

Le mode `live` passe des **ordres réels** avec **votre argent**. Le robot est construit pour
limiter les dégâts (stops, taille des positions, kill switch, limite journalière, jamais de
levier, jamais de retrait), mais **aucune protection logicielle ne remplace la prudence** :
un exchange peut tomber, une connexion peut se couper, un prix peut faire un trou.

## 0. Ce que le robot fait et ne fait pas sur votre compte

Il **fait** :

- lire votre solde et votre position (`fetch_balance`, `fetch_ticker`) ;
- passer des ordres **au marché** (`create_order(symbol, "market", "buy"/"sell", quantité)`)
  pour entrer, sortir, ou couper sur stop ;
- arrondir la quantité aux règles du marché et refuser un ordre sous les minimums.

Il **ne fait jamais** :

- de retrait (`withdraw`) : le code ne contient aucun appel de ce type ;
- de levier, de marge, de contrats à terme : marché au comptant uniquement ;
- de vente à découvert : au comptant, on ne peut pas vendre ce qu'on ne détient pas, chaque
  signal short deviendrait un ordre refusé par l'exchange. La commande `live` **refuse donc
  de démarrer** si `allow_short = true` est dans `config.toml` ou si `--allow-short` est
  passé (code de sortie 2, explication affichée). En `paper`, la vente à découvert est
  seulement simulée, avec un avertissement.

Il ne gère **qu'une paire à la fois**. Mettez sur le compte **uniquement** l'argent que vous
confiez au robot. S'il y trouve déjà l'actif de base (du BTC acheté à la main, par
exemple) alors qu'il n'a aucune position mémorisée, il **ne le touche pas** et refuse
d'ouvrir tant qu'il est là ; voir `--adopt-position` au §6.

## 1. Créer un compte sur un exchange et le sécuriser

1. Choisissez un exchange sérieux, supporté par `ccxt` (Binance, Kraken, Coinbase, Bybit,
   OKX…). Vérifiez qu'il est autorisé dans votre pays.
2. Activez la **double authentification (2FA)** avec une application (pas par SMS).
3. Créez, si possible, un **sous-compte** dédié au robot : en cas de problème, seul ce
   sous-compte est exposé.
4. Déposez sur ce compte **un petit capital** : une somme dont la perte totale ne vous
   empêcherait pas de dormir. Les minimums d'ordre tournent autour de 10 USDT ; un capital
   de quelques centaines d'USDT suffit pour commencer, pas plus.

## 2. Créer une clé API avec les bons droits (et surtout pas les mauvais)

Une **clé API** est un identifiant + un secret qui permettent à un programme d'agir sur
votre compte. Elle se crée dans les paramètres de l'exchange (« API Management », « Clés
API »). À la création :

| Droit | Réglage | Pourquoi |
|---|---|---|
| Lecture (*read*) | **oui** | le robot lit le solde et les prix |
| Trading au comptant (*spot trade*) | **oui** | passer les ordres |
| **Retrait (*withdraw*)** | **NON, jamais** | si la clé fuit, personne ne peut vider le compte |
| Marge / futures / prêt | **non** | le robot n'en a pas besoin ; moins de droits = moins de risque |
| Transferts internes / sous-comptes | non | idem |

Puis :

- **Restreignez la clé à votre adresse IP** (« IP whitelist ») : celle de la machine qui fait
  tourner le robot. Une clé volée devient inutilisable ailleurs. Si votre IP change (box
  domestique), il faudra la mettre à jour ; c'est une contrainte, mais c'est la bonne.
- Notez le **secret** immédiatement : la plupart des exchanges ne l'affichent qu'une fois.
- Donnez à la clé une **date d'expiration** si l'exchange le propose, et renouvelez-la.
- Ne partagez jamais une clé, même avec « le support » : un support légitime ne la demande
  jamais.

## 3. Mettre la clé dans `.env` (et nulle part ailleurs)

```bash
cp .env.example .env          # Linux / macOS
copy .env.example .env        # Windows (cmd ; sous PowerShell, cp fonctionne aussi)
```

Puis éditez `.env` (avec n'importe quel éditeur de texte) :

```
TRADINGBOT_API_KEY=votre_cle
TRADINGBOT_API_SECRET=votre_secret
```

- `.env` est listé dans `.gitignore` : il ne sera jamais envoyé sur git. Ne le copiez pas
  dans `config.toml`, un script, un message, une capture d'écran.
- Sur un serveur partagé (Linux / macOS), restreignez les droits du fichier : `chmod 600
  .env`. Sous Windows, `chmod` n'existe pas : vérifiez que seul votre compte utilisateur a
  accès au dossier (clic droit → Propriétés → Sécurité).
- Les variables d'environnement du système (`export TRADINGBOT_API_KEY=...`) sont acceptées
  et prioritaires sur `.env`.
- Le robot **n'affiche jamais** les secrets : clé, secret et jeton Telegram apparaissent
  comme `***` dans les journaux et les messages d'erreur (y compris les URL signées que
  certains exchanges recopient dans leurs erreurs), et l'identifiant de chat Telegram n'est
  pas journalisé.

## 4. Configurer `config.toml`

```toml
[general]
mode = "live"             # le premier des trois verrous
exchange = "binance"
symbol = "BTC/USDT"
timeframe = "1h"          # 1h ou 4h pour débuter : moins de trades, moins de frais
state_dir = "state"

[strategy]
name = "ema_trend"        # celle qui a passé le walk-forward ET le paper trading

[risk]
initial_cash = 300.0      # indicatif en live : le vrai solde est lu sur l'exchange
risk_per_trade = 0.01
max_position_fraction = 0.95
max_drawdown = 0.20
daily_loss_limit = 0.05   # bougies < 1 jour seulement (inactive en 1d et au-delà)
allow_short = false       # obligatoire en live (spot)
default_stop_atr_mult = 3.0   # stop de secours si la stratégie n'en donne pas
default_stop_pct = 0.05       # stop de dernier recours si l'ATR est indisponible
```

Trois remarques :

- gardez la **même stratégie et les mêmes paramètres** qu'en paper trading. Changer quelque
  chose au moment de passer en réel revient à repartir de zéro ;
- en live, `initial_cash` n'est **qu'affiché** dans le récapitulatif : la taille des
  positions et les garde-fous se calculent sur l'équité réelle lue sur l'exchange (cash +
  valeur de la position). En paper, c'est le capital fictif de départ (`--initial-cash`
  le remplace) ;
- toutes les stratégies livrées portent leur stop sur **chaque** bougie en position, et le
  robot n'ouvre **jamais** sans stop (stratégie, sinon 3 × ATR, sinon 5 % du prix) : un robot
  qui démarre au milieu d'une tendance entre protégé et dimensionné sur le risque.
  `ema_trend` a en plus un stop **suiveur**, qui remonte avec le prix.

## 5. D'abord le testnet / sandbox, si l'exchange en propose un

Certains exchanges (Binance, Bybit, OKX…) offrent un environnement de **test** avec de la
fausse monnaie mais de vrais mécanismes d'ordres. Créez-y un compte et une clé API
dédiés (ce sont des clés différentes de celles du compte réel), mettez-les dans `.env`,
puis :

```bash
tradingbot live --i-understand-the-risks --sandbox --once
tradingbot live --i-understand-the-risks --sandbox
```

L'état est rangé dans `state/sandbox/…`, séparé du paper et du réel. Vérifiez sur le site
du testnet que les ordres apparaissent bien, avec les bonnes quantités. Faites-le tourner
quelques jours. Un exchange sans testnet se teste… avec un capital minuscule.

## 6. Le premier lancement réel

Les trois verrous doivent être levés, sinon la commande refuse (code de sortie 2) et
explique ce qui manque :

1. `mode = "live"` dans `config.toml` ;
2. le drapeau `--i-understand-the-risks` sur la ligne de commande ;
3. `TRADINGBOT_API_KEY` et `TRADINGBOT_API_SECRET` dans `.env`.

```bash
tradingbot live --i-understand-the-risks --once
```

Le robot affiche :

```
*** MODE LIVE : ARGENT RÉEL ***
Récapitulatif avant connexion :
  Exchange                    : binance
  Symbole                     : BTC/USDT
  Timeframe                   : 1h
  Stratégie                   : ema_trend (fast=20, slow=50, ...)
  Capital de référence        : 300.00 (indicatif : le solde réel vient de l'exchange)
  Risque par trade            : 1.00 %
  Position maximale           : 95.00 % du capital
  Stop de protection          : toujours présent : stop de la stratégie s'il est du bon côté du prix, sinon 3 × ATR(14), sinon 5.00 % du prix ; taille = risque par trade
  Ré-entrée après stop        : attente d'un nouveau signal après un stop / objectif / limite journalière (wait)
  Kill switch (drawdown)      : 20.00 %
  Limite de perte journalière : 5.00 %
  Vente à découvert           : désactivée
  Notionnel minimal           : 10
  Fichiers d'état             : state/live/binance_BTC-USDT_1h (un seul robot à la fois : verrou « lock »)
  Mode                        : ARGENT RÉEL
Tapez OUI (en majuscules) pour confirmer, autre chose pour annuler :
```

Avec `--timeframe 1d` (ou plus), la ligne « Limite de perte journalière » se termine par
« INACTIVE sur ce timeframe (bougies >= 1 jour…) » : c'est voulu, voir §9.

Lisez chaque ligne. Tapez `OUI` seulement si tout est exactement ce que vous voulez.
(`--yes` saute cette question : réservez-le à un lancement automatisé que vous avez déjà
validé à la main.) Après la connexion, le robot affiche le solde réel et la ligne
« Position existante » : ce que le compte détient déjà et ce qu'il va en faire.

`--once` fait une seule itération et affiche l'état : c'est la façon la plus sûre de
vérifier que la connexion, le solde et les quantités sont corrects avant la boucle. Code
de sortie `0` si l'itération s'est bien passée, `1` si elle s'est terminée en `"action":
"error"` (exchange injoignable, ordre refusé…).

**L'actif est déjà sur le compte ?** Si le robot trouve du BTC alors qu'il n'a aucune
position mémorisée (`position.json` vide), il considère cette position comme **non gérée** :
il ne la vend pas, ne lui met pas de stop et **refuse d'ouvrir** (`refused`, « position
existante non gérée par le robot ») tant qu'elle est là. Deux issues : vendez-la à la main,
ou relancez avec **`--adopt-position`** pour la lui confier ; il la gérera alors comme la
sienne (sans stop jusqu'au prochain stop de la stratégie, donc sous la seule protection du
kill switch jusque-là). Inversement, le robot ne vend jamais plus que **sa propre** quantité :
un achat manuel fait *en plus* de sa position n'est pas touché.

Puis la boucle :

```bash
tradingbot live --i-understand-the-risks
```

Le robot interroge le marché toutes les `durée de bougie / 12` (5 min en 1h ; `--poll 60`
pour changer), vérifie le stop à chaque interrogation, et n'agit sur le signal qu'une fois
par bougie clôturée. Si Telegram est configuré (`[notifications] telegram = true`,
`TELEGRAM_BOT_TOKEN` et `TELEGRAM_CHAT_ID` dans `.env`), chaque entrée, sortie, stop, refus
ou erreur vous est envoyé.

## 7. Faire tourner le robot sans interruption

Le robot doit tourner en continu, sinon il ne verra pas ses stops (il les vérifie à chaque
interrogation ; il ne pose **pas** d'ordre stop sur l'exchange). Options :

- une machine toujours allumée (petit serveur, Raspberry Pi, VPS) ;
- un gestionnaire de processus (`systemd`, `supervisor`, `tmux`/`screen` au minimum) qui
  relance le robot s'il plante : l'état est persistant, un redémarrage reprend la position
  et son stop sans passer d'ordre en double ;
- surveillez les journaux (`--log-level info`) et les notifications ;
- **un seul robot par dossier d'état** : le fichier `state/live/<exchange>_<symbole>_<timeframe>/lock`
  est verrouillé par le robot qui tourne. Si votre gestionnaire de processus (ou vous)
  lance une seconde instance sur le même dossier, elle s'arrête immédiatement avec « REFUS :
  le répertoire d'état … est déjà utilisé par un autre robot (PID …) » et le code de sortie
  `2` — jamais deux robots sur le même compte sans le savoir. Un `--once` rend le verrou dès
  la fin de son itération.

Si le robot est arrêté longtemps avec une position ouverte, **fermez-la à la main** sur
l'exchange ou relancez-le vite : pendant l'arrêt, personne ne surveille le stop.

## 8. Arrêter le robot

- **Arrêt propre** : `Ctrl-C` dans le terminal. Le robot termine l'itération en cours,
  journalise « Arrêt demandé », **ne ferme pas la position** (elle est mémorisée dans
  `state/live/<exchange>_<symbole>_<timeframe>/position.json`), libère le verrou et quitte
  avec le code de sortie `130` (la convention des shells pour « interrompu ») : un
  superviseur distingue ainsi un arrêt volontaire d'une fin normale (`0`) ou d'une erreur
  (`1`). La position reprend au prochain lancement.
- **Fermer la position** : soit sur le site de l'exchange (vendez la quantité détenue),
  soit en relançant le robot : il constatera que le broker est plat et remettra son état à
  zéro.
- **Arrêt d'urgence absolu** : sur l'exchange, **supprimez la clé API**. Le robot ne peut
  plus rien faire, immédiatement, quoi qu'il arrive à la machine qui le fait tourner.
  C'est le bouton rouge ; sachez où il est avant d'en avoir besoin.

## 9. Le kill switch et la limite journalière

Le gestionnaire de risque suit l'équité (cash + valeur de la position) à **chaque
interrogation du marché** (toutes les `durée de bougie / 12`, soit 5 minutes en 1h ; `--poll`
pour changer), et pas seulement à la clôture d'une bougie. Précisément : à chaque
interrogation, le robot vérifie d'abord le stop, puis le kill switch et la limite
journalière ; le signal de la stratégie, lui, n'est évalué qu'une fois par bougie clôturée.
Entre deux interrogations, **rien n'est surveillé** : avec `--poll 300`, une chute de 30 %
en deux minutes n'est vue qu'à l'interrogation suivante. Ces garde-fous bornent les dégâts
*après* la perte ; ils ne la préviennent pas.

- **Kill switch** (`max_drawdown`, 20 % par défaut) : dès qu'une interrogation constate une
  équité tombée de plus de 20 % sous son plus haut, le robot **ferme la position** (ordre au
  marché), refuse toute nouvelle ouverture et **arrête la boucle**. L'état est écrit dans
  `risk.json` : relancer le robot ne le réarme pas. Pour le réarmer, après avoir compris ce
  qui s'est passé, supprimez `state/live/<exchange>_<symbole>_<timeframe>/risk.json` (ou
  tout le dossier d'état pour repartir de zéro). Ne le faites pas à chaud, sous le coup de
  la perte.
- **Limite journalière** (`daily_loss_limit`, 5 % par défaut) : à –5 % depuis l'équité de
  début de journée (UTC), position fermée, aucune ouverture jusqu'au lendemain. Elle se
  réarme seule au changement de jour. Elle n'a de sens qu'en **intrajournalier** : avec un
  timeframe d'un jour ou plus (`1d`, `1w`), chaque bougie est une nouvelle journée et elle
  est automatiquement **inactive** (le récapitulatif l'indique ; le kill switch reste
  actif). Après une sortie sur limite journalière, le côté sorti reste bloqué (voir « Stop »).
- **Stop** : vérifié contre le prix courant à chaque interrogation, **avant** même de
  demander les bougies (une panne de la source de données ne le désarme pas) ; s'il est
  franchi, la position est fermée au marché. Ensuite, politique **`wait`** : le robot ne
  rentre pas du même côté tant que le signal de la stratégie n'est pas repassé par plat (ou
  ne s'est pas retourné) sur une bougie clôturée ultérieure — même si la stratégie dit
  toujours « long » à la bougie suivante. Le côté bloqué est mémorisé dans `position.json`
  (`blocked_side`) et survit à un redémarrage. La même règle s'applique après un objectif
  atteint et après la limite journalière.
- **Refus** : si le gestionnaire de risque refuse une ouverture (`refused`), c'est voulu.
  Ne « forcez » pas.

Ces mécanismes agissent **après** la perte, pas avant : ils bornent les dégâts, ils ne les
empêchent pas. D'où le capital petit. En backtest, la section « Garde-fous déclenchés » du
rapport vous montre à l'avance combien de fois ils seraient intervenus.

## 10. Ce qui peut mal tourner (et quoi faire)

| Problème | Conséquence | Que faire |
|---|---|---|
| Coupure réseau / API en panne | l'itération renvoie `error` (code de sortie 1 avec `--once`), le robot réessaie à l'interrogation suivante ; le stop est quand même vérifié dès que le prix est à nouveau disponible, avant les bougies | surveiller les notifications ; au besoin fermer à la main |
| Deux robots lancés sur le même dossier d'état | le second s'arrête aussitôt (`REFUS … déjà utilisé par un autre robot`, code 2) | arrêter l'autre, ou choisir un `--state-dir` différent |
| Gap de prix (le marché saute par-dessus le stop) | perte plus grande que `risk_per_trade` | normal, c'est pour cela que la position est plafonnée |
| Ordre refusé (quantité sous le minimum, fonds insuffisants) | `error` ou `skip` ; pas d'ordre, l'état mémorisé et son stop sont conservés | augmenter le capital ou choisir un actif moins cher |
| Ordre de clôture partiellement exécuté | `error`, la quantité restante est mémorisée et la clôture retentée à l'interrogation suivante | surveiller ; au besoin finir à la main |
| Clé API expirée / IP changée | `error` à chaque itération | créer une nouvelle clé, mettre à jour `.env` |
| Machine éteinte avec une position ouverte | personne ne surveille le stop | relancer vite ou fermer à la main |
| Vous vendez à la main la position du robot | il constate que le compte est plat et remet son état à zéro | prévenez-le plutôt : arrêtez-le d'abord |
| Vous achetez l'actif à la main alors que le robot est plat | il ne touche pas ce solde, le signale et **refuse d'ouvrir** tant qu'il est là | vendez-le, ou `--adopt-position` pour le lui confier |
| Vous achetez l'actif à la main en plus de la position du robot | il ne gère et ne vend que **sa** quantité | évitez ; sinon, arrêtez le robot d'abord |

## 11. Et ensuite

- Tenez un **journal** : date, réglages, pourquoi vous lancez/arrêtez. Les fichiers
  `trades.csv` des rapports et les ordres dans `state/` vous aideront pour les impôts.
- Comparez chaque mois le réel au paper et au backtest sur la même période. Un écart
  important (plus de trades, plus de glissement) est une information : vos coûts estimés
  étaient trop optimistes.
- N'augmentez le capital que lentement, après des mois, et jamais après une perte « pour
  se refaire ».
- Si vous ne comprenez plus ce que fait le robot, arrêtez-le. Le marché sera encore là.
