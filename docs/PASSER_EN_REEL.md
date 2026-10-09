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
- de vente à découvert, sauf si vous activez `allow_short` (déconseillé, et impossible
  au comptant sur la plupart des exchanges).

Il ne gère **qu'une paire à la fois**. Mettez sur le compte **uniquement** l'argent que vous
confiez au robot.

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
cp .env.example .env
```

Puis éditez `.env` :

```
TRADINGBOT_API_KEY=votre_cle
TRADINGBOT_API_SECRET=votre_secret
```

- `.env` est listé dans `.gitignore` : il ne sera jamais envoyé sur git. Ne le copiez pas
  dans `config.toml`, un script, un message, une capture d'écran.
- Sur un serveur partagé, restreignez les droits du fichier : `chmod 600 .env`.
- Les variables d'environnement du système (`export TRADINGBOT_API_KEY=...`) sont acceptées
  et prioritaires sur `.env`.
- Le robot **n'affiche jamais** les secrets (ils apparaissent comme `***` dans les
  journaux).

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
initial_cash = 300.0      # capital de référence : ce que vous avez mis sur le compte
risk_per_trade = 0.01
max_position_fraction = 0.95
max_drawdown = 0.20
daily_loss_limit = 0.05
allow_short = false
```

Deux remarques :

- gardez la **même stratégie et les mêmes paramètres** qu'en paper trading. Changer quelque
  chose au moment de passer en réel revient à repartir de zéro ;
- `ema_trend` fournit un stop suiveur à chaque bougie : c'est la stratégie la plus adaptée
  à un robot qui démarre au milieu d'une tendance (les autres n'émettent leur stop qu'à la
  bougie d'entrée ; voir le guide du débutant).

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
  Capital de référence        : 300.00
  Risque par trade            : 1.00 %
  Position maximale           : 95.00 % du capital
  Kill switch (drawdown)      : 20.00 %
  Limite de perte journalière : 5.00 %
  Vente à découvert           : désactivée
  ...
  Mode                        : ARGENT RÉEL
Tapez OUI (en majuscules) pour confirmer, autre chose pour annuler :
```

Lisez chaque ligne. Tapez `OUI` seulement si tout est exactement ce que vous voulez.
(`--yes` saute cette question : réservez-le à un lancement automatisé que vous avez déjà
validé à la main.)

`--once` fait une seule itération et affiche l'état : c'est la façon la plus sûre de
vérifier que la connexion, le solde et les quantités sont corrects avant la boucle.

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
- surveillez les journaux (`--log-level info`) et les notifications.

Si le robot est arrêté longtemps avec une position ouverte, **fermez-la à la main** sur
l'exchange ou relancez-le vite : pendant l'arrêt, personne ne surveille le stop.

## 8. Arrêter le robot

- **Arrêt propre** : `Ctrl-C` dans le terminal. Le robot termine l'itération en cours,
  journalise « Arrêt demandé », **ne ferme pas la position** (elle est mémorisée dans
  `state/live/<exchange>_<symbole>_<timeframe>/position.json`) et reprendra au prochain
  lancement.
- **Fermer la position** : soit sur le site de l'exchange (vendez la quantité détenue),
  soit en relançant le robot : il constatera que le broker est plat et remettra son état à
  zéro.
- **Arrêt d'urgence absolu** : sur l'exchange, **supprimez la clé API**. Le robot ne peut
  plus rien faire, immédiatement, quoi qu'il arrive à la machine qui le fait tourner.
  C'est le bouton rouge ; sachez où il est avant d'en avoir besoin.

## 9. Le kill switch et la limite journalière

Le gestionnaire de risque suit l'équité (cash + valeur de la position) à chaque itération.

- **Kill switch** (`max_drawdown`, 20 % par défaut) : dès que l'équité est tombée de plus
  de 20 % sous son plus haut, le robot **ferme la position**, refuse toute nouvelle
  ouverture et **arrête la boucle**. L'état est écrit dans `risk.json` : relancer le robot
  ne le réarme pas. Pour le réarmer, après avoir compris ce qui s'est passé, supprimez
  `state/live/<exchange>_<symbole>_<timeframe>/risk.json` (ou tout le dossier d'état pour
  repartir de zéro). Ne le faites pas à chaud, sous le coup de la perte.
- **Limite journalière** (`daily_loss_limit`, 5 % par défaut) : à –5 % depuis l'équité de
  début de journée (UTC), position fermée, aucune ouverture jusqu'au lendemain. Elle se
  réarme seule au changement de jour.
- **Stop** : vérifié contre le prix courant à chaque interrogation ; s'il est franchi, la
  position est fermée au marché, et aucune ré-entrée n'a lieu avant la bougie suivante.
- **Refus** : si le gestionnaire de risque refuse une ouverture (`refused`), c'est voulu.
  Ne « forcez » pas.

Ces mécanismes agissent **après** la perte, pas avant : ils bornent les dégâts, ils ne les
empêchent pas. D'où le capital petit.

## 10. Ce qui peut mal tourner (et quoi faire)

| Problème | Conséquence | Que faire |
|---|---|---|
| Coupure réseau / API en panne | l'itération renvoie `error`, le robot réessaie à l'interrogation suivante ; le stop n'est pas surveillé pendant ce temps | surveiller les notifications ; au besoin fermer à la main |
| Gap de prix (le marché saute par-dessus le stop) | perte plus grande que `risk_per_trade` | normal, c'est pour cela que la position est plafonnée |
| Ordre refusé (quantité sous le minimum, fonds insuffisants) | `error` ou `skip` ; pas d'ordre | augmenter le capital ou choisir un actif moins cher |
| Clé API expirée / IP changée | `error` à chaque itération | créer une nouvelle clé, mettre à jour `.env` |
| Machine éteinte avec une position ouverte | personne ne surveille le stop | relancer vite ou fermer à la main |
| Vous intervenez à la main sur l'exchange | le robot adopte la position du broker (sans stop) ou remet son état à zéro | évitez ; sinon, arrêtez le robot d'abord |

## 11. Et ensuite

- Tenez un **journal** : date, réglages, pourquoi vous lancez/arrêtez. Les fichiers
  `trades.csv` des rapports et les ordres dans `state/` vous aideront pour les impôts.
- Comparez chaque mois le réel au paper et au backtest sur la même période. Un écart
  important (plus de trades, plus de glissement) est une information : vos coûts estimés
  étaient trop optimistes.
- N'augmentez le capital que lentement, après des mois, et jamais après une perte « pour
  se refaire ».
- Si vous ne comprenez plus ce que fait le robot, arrêtez-le. Le marché sera encore là.
