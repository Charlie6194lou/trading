# Les stratégies, expliquées à un débutant

Ce document décrit les cinq stratégies livrées avec le robot. Il ne suppose **aucune**
connaissance préalable du trading. Lisez d'abord le vocabulaire, puis chaque stratégie,
et surtout la dernière section : *Aucune stratégie ne gagne toujours*.

Pour voir la liste depuis le terminal : `tradingbot list-strategies`.
Pour en tester une sur des données fictives, sans réseau : `tradingbot demo`.

## Vocabulaire minimal

- **Bougie (candle)** : résumé du prix sur une période fixe (1 heure, 1 jour…) : prix
  d'ouverture (`open`), plus haut (`high`), plus bas (`low`), clôture (`close`) et volume.
- **Position** : ce que le robot détient. *Long* (`1`) = il a acheté et gagne si le prix
  monte. *Short* (`-1`) = il a vendu à découvert et gagne si le prix baisse. *Plat* (`0`) =
  il ne détient rien. Par défaut la vente à découvert est **désactivée**
  (`allow_short = false`) : un signal `-1` est alors simplement traité comme « plat ».
- **Signal** : à la fin de chaque bougie, la stratégie dit quelle position elle *souhaite*
  pour la bougie suivante. Le robot exécute à l'ouverture de la bougie suivante : il
  n'utilise jamais une information qu'il n'avait pas encore (pas de « triche » sur le futur).
- **Stop (stop-loss)** : prix auquel on coupe la position si le marché va contre nous.
  C'est la ceinture de sécurité : il limite la perte d'un trade. Toutes les stratégies
  livrées fournissent un stop basé sur l'**ATR**, et le portent sur **chaque bougie** où
  elles sont en position (pas seulement à l'entrée) : un robot qui démarre en cours de
  tendance a donc toujours le stop de la stratégie. Si une stratégie n'en donnait pas (ou
  le donnait du mauvais côté du prix), le robot placerait lui-même un stop de secours
  (voir *Points communs*) : **aucune position ne s'ouvre sans stop**.
- **Range (marché sans tendance)** : le prix oscille entre un plancher et un plafond sans
  direction durable ; on dit aussi « marché en sur-place » ou « latéral ». C'est le régime
  où les stratégies de tendance perdent et où `bollinger_reversion` gagne.
- **ATR (Average True Range)** : la taille moyenne des mouvements d'une bougie, c'est-à-dire
  « combien le prix bouge d'habitude ». Un stop à `3 × ATR` est placé à trois mouvements
  habituels du prix : assez loin pour ne pas être touché par le bruit, assez près pour
  limiter la casse. Le gestionnaire de risque utilise aussi la distance au stop pour
  calculer la taille de la position (`risk_per_trade`).
- **Moyenne mobile (SMA / EMA)** : moyenne des `N` dernières clôtures ; elle lisse le bruit
  et montre la direction de fond. L'EMA (exponentielle) donne plus de poids aux bougies
  récentes et réagit donc plus vite que la SMA (simple).
- **Période de chauffe (warm-up)** : les premières bougies pendant lesquelles les
  indicateurs ne sont pas encore calculables. La stratégie reste plate (`0`).
- **Suivi de tendance** vs **retour à la moyenne** : les deux grandes familles. La
  première parie que ce qui monte va continuer à monter ; la seconde parie que ce qui a
  trop baissé va rebondir. Elles gagnent dans des marchés opposés.

## Deux familles, deux philosophies

| Famille | Stratégies | Parie sur | Gagne quand | Perd quand |
|---|---|---|---|---|
| Suivi de tendance | `sma_cross`, `ema_trend`, `donchian_breakout` | la continuation du mouvement | le marché part dans une direction durable | le marché fait du sur-place (faux départs à répétition) |
| Retour à la moyenne | `bollinger_reversion` | le rebond après un excès | le marché oscille autour d'un niveau (range) | le marché s'effondre sans rebondir (« attraper un couteau qui tombe ») |
| Retour à la moyenne **filtré par la tendance** | `rsi_reversion` | le rebond d'un repli **dans un marché haussier** | le marché monte avec des corrections régulières (« acheter les creux ») | le marché s'effondre (le filtre réagit en retard) ; dans un range pur elle ne trade presque pas (voir sa section) |

Les stratégies de tendance ont typiquement un faible taux de réussite (30-45 % de trades
gagnants) mais des gains moyens bien plus grands que les pertes. Les stratégies de
retour à la moyenne ont l'inverse : beaucoup de petits gains, et de temps en temps une
perte plus importante. Ni l'un ni l'autre n'est « meilleur » dans l'absolu.

---

## `sma_cross` : croisement de moyennes mobiles

**Ce qu'elle parie.** Quand la moyenne courte (20 bougies) passe au-dessus de la moyenne
longue (50 bougies), le prix récent est plus haut que le prix moyen passé : une tendance
haussière s'installe. On suit le mouvement.

**Entrée / sortie.**
- Position **longue** tant que `SMA(fast) > SMA(slow)`.
- Position **courte** tant que `SMA(fast) < SMA(slow)` (ou plate si le short est désactivé).
- En cas d'égalité parfaite, on garde la position précédente (pas de va-et-vient).
- Il n'y a donc jamais de « sortie à plat » : on est toujours d'un côté ou de l'autre.
  C'est la plus simple des stratégies, et la plus exposée aux faux signaux.

**Stop.** Calculé sur la bougie d'entrée : `close − atr_mult × ATR` (pour un long), puis
**porté tel quel sur chaque bougie** tant que la position est tenue (il n'est pas déplacé) ;
la sortie normale est le croisement inverse. Si le stop est touché, le moteur / le robot
ferment la position et n'en rouvrent pas une du même côté tant que les moyennes ne se sont
pas recroisées (politique `wait`, voir *Points communs*).

**Marché favorable / défavorable.** Excellente dans une tendance longue et propre (par
exemple un marché haussier de plusieurs mois). Mauvaise dans un marché en range (sans
tendance) : les moyennes se croisent sans cesse et chaque croisement coûte des frais et une
petite perte.

| Paramètre | Défaut | Signification |
|---|---|---|
| `fast` | 20 | longueur de la moyenne courte (bougies) |
| `slow` | 50 | longueur de la moyenne longue ; doit être `> fast` |
| `atr_window` | 14 | longueur de l'ATR |
| `atr_mult` | 3.0 | distance du stop, en multiples d'ATR |

---

## `ema_trend` : tendance EMA filtrée par l'ADX, stop suiveur « chandelier »

**Ce qu'elle parie.** Même idée que `sma_cross`, avec deux garde-fous :
1. un **filtre de force** : l'ADX mesure si le marché est en tendance (ADX élevé) ou
   en sur-place (ADX bas). On n'entre que si `ADX > adx_threshold` : cela évite de
   prendre les petits croisements d'un marché qui hésite ;
2. un **stop suiveur** : une fois en position, le stop monte avec le prix pour
   protéger les gains acquis (stop « chandelier » : le plus haut atteint par
   `close − atr_mult × ATR` depuis l'entrée ; il ne redescend jamais).

**Entrée / sortie.**
- **Entrée longue** : `EMA(fast) > EMA(slow)` **et** `ADX > adx_threshold`.
- **Entrée courte** : symétrique (`EMA(fast) < EMA(slow)` et `ADX > adx_threshold`).
- **Sortie** si l'une des deux conditions se produit :
  - les EMA se recroisent (la tendance s'inverse). Si l'ADX est au-dessus du seuil à ce
    moment-là, on se retourne directement dans l'autre sens ;
  - le **plus bas** de la bougie touche le stop suiveur (`low <= stop` ; `high >= stop` pour
    un short), c'est-à-dire exactement la condition à laquelle le moteur de backtest et le
    robot exécutent ce stop. On reste plat une bougie, puis on réévalue : si la tendance et
    l'ADX le justifient, on rentre avec un nouveau stop. La stratégie et le moteur voient
    ainsi la **même** sortie : il n'y a plus de ré-entrée immédiate avec un stop déjà
    touché, qui faisait autrefois enchaîner les stop-outs.
- **Choix de conception** : l'ADX ne sert qu'à l'*entrée*. Une position ouverte est
  conservée tant que les EMA restent dans le bon ordre, même si l'ADX redescend sous le
  seuil. Une tendance qui ralentit n'est pas une tendance qui s'inverse ; sortir à chaque
  baisse de l'ADX ferait rater la suite du mouvement et multiplierait les frais.

**Stop.** À l'entrée : `close − atr_mult × ATR`. Ensuite, à chaque bougie en position,
le stop vaut `max(stop précédent, close − atr_mult × ATR)` (symétrique pour un short).
Le moteur met à jour son stop à chaque bougie (un stop qui serait passé du mauvais côté du
prix est ignoré).

**Marché favorable / défavorable.** Marchés en tendance nette (ADX > 25). Moins de
faux signaux que `sma_cross` dans les ranges, mais entre plus tard dans la tendance (le
filtre ADX coûte quelques bougies) et peut être sortie par le stop suiveur lors d'un
repli brutal avant que la tendance ne reprenne.

| Paramètre | Défaut | Signification |
|---|---|---|
| `fast` | 20 | longueur de l'EMA courte |
| `slow` | 50 | longueur de l'EMA longue ; doit être `> fast` |
| `adx_window` | 14 | longueur de l'ADX |
| `adx_threshold` | 20 | force minimale de la tendance pour entrer (0-100 ; 20-25 est l'usage) |
| `atr_window` | 14 | longueur de l'ATR |
| `atr_mult` | 3.0 | distance du stop suiveur, en multiples d'ATR |

---

## `rsi_reversion` : achat des replis sur RSI (achat uniquement)

**Ce qu'elle parie.** Le RSI (0-100) mesure la vitesse des hausses par rapport aux
baisses récentes. Sous 30, le prix a baissé « trop vite » : on parie sur un rebond.
Mais seulement si le marché de fond est haussier (clôture au-dessus de la moyenne à
200 bougies) : acheter un repli dans un marché qui monte est bien plus sûr que dans un
marché qui s'effondre.

**Entrée / sortie.**
- **Entrée** (si plat) : `RSI < oversold` **et** `close > SMA(trend_window)`.
- **Sortie** : `RSI > exit_level` (le rebond a eu lieu) **ou** la position a été tenue
  `max_hold` bougies (le rebond n'est pas venu : on ne s'entête pas).
- Jamais de vente à découvert.

**Stop.** Calculé à l'entrée : `close − atr_mult × ATR` (plus serré que pour les stratégies
de tendance : un rebond qui ne vient pas se reconnaît vite), puis porté tel quel sur chaque
bougie en position.

**Marché favorable / défavorable.** Marché haussier avec des corrections régulières
(« acheter les creux »). Mauvaise quand la tendance de fond casse : les premiers replis
après un sommet ressemblent à des creux à acheter, et le filtre de tendance ne réagit
qu'avec retard. Dans un marché sans replis (hausse rectiligne), elle ne fait tout
simplement rien. **Dans un range pur non plus**, et c'est voulu : quand le prix oscille
autour d'un niveau, le RSI passe sous 30 précisément quand la clôture est nettement *sous*
la moyenne à 200 bougies, donc le filtre `close > SMA(trend_window)` rejette presque toutes
les entrées (0 à 2 trades sur 3 000 bougies dans nos essais, là où `bollinger_reversion`
en fait 30 à 40). Si `demo` ou `compare` lui attribuent très peu de trades, elle n'est pas
cassée : c'est son filtre qui la tient à l'écart. Pour un marché en range, utilisez
`bollinger_reversion` ; abaisser `trend_window` ne suffit pas (le mécanisme est le même avec
une moyenne à 50).

| Paramètre | Défaut | Signification |
|---|---|---|
| `rsi_window` | 14 | longueur du RSI |
| `oversold` | 30 | niveau de RSI sous lequel on achète |
| `exit_level` | 55 | niveau de RSI au-dessus duquel on revend ; doit être `> oversold` |
| `trend_window` | 200 | longueur de la moyenne de tendance (filtre « marché haussier ») |
| `atr_window` | 14 | longueur de l'ATR |
| `atr_mult` | 2.0 | distance du stop, en multiples d'ATR |
| `max_hold` | 20 | nombre maximal de bougies en position |

---

## `donchian_breakout` : cassure de canal (méthode des « Turtles »)

**Ce qu'elle parie.** Si le prix clôture au-dessus du plus haut des 20 dernières bougies,
quelque chose de nouveau se passe (nouvelle information, nouveaux acheteurs) : on
achète la cassure et on laisse courir. C'est la stratégie rendue célèbre par
l'expérience des « Turtle Traders » dans les années 1980.

**Entrée / sortie.**
- **Entrée longue** (si plat) : `close > plus haut des entry_window bougies précédentes`.
- **Entrée courte** (si plat) : `close < plus bas des entry_window bougies précédentes`.
- **Sortie d'un long** : `close < plus bas des exit_window bougies précédentes`
  (fenêtre plus courte : on sort plus vite qu'on n'entre). Si cette clôture casse aussi le
  canal d'entrée vers le bas, on se retourne directement en short. Symétrique pour un short.
- Les canaux sont calculés sur les bougies *précédentes* : la bougie courante ne peut pas
  déclencher sa propre cassure.

**Stop.** Calculé à l'entrée : `close − atr_mult × ATR` (pour un long), puis porté tel quel
sur chaque bougie en position.

**Marché favorable / défavorable.** Marchés qui font de longs mouvements directionnels
(les cryptos en ont souvent). Très mauvaise dans un range (sans tendance) étroit : chaque sortie du
canal est un faux départ, et la sortie sur le plus bas à 10 bougies arrive après une
perte. Attendez-vous à beaucoup de petits trades perdants compensés par quelques gros
gagnants ; c'est normal pour cette stratégie, mais psychologiquement difficile.

| Paramètre | Défaut | Signification |
|---|---|---|
| `entry_window` | 20 | longueur du canal d'entrée (bougies) |
| `exit_window` | 10 | longueur du canal de sortie, en général plus court |
| `atr_window` | 14 | longueur de l'ATR |
| `atr_mult` | 2.0 | distance du stop, en multiples d'ATR |

---

## `bollinger_reversion` : retour à la moyenne sur bandes de Bollinger (achat uniquement)

**Ce qu'elle parie.** Les bandes de Bollinger entourent le prix à `n_std` écarts-types
(une mesure de la dispersion habituelle) autour de sa moyenne à 20 bougies. Une clôture
*sous* la bande basse est statistiquement rare : si le RSI confirme l'excès de baisse,
on achète en pariant sur un retour vers la moyenne.

**Entrée / sortie.**
- **Entrée** (si plat) : `close < bande basse` **et** `RSI < rsi_max`.
- **Sortie** : `close >= bande du milieu` (la moyenne a été rejointe).
- Jamais de vente à découvert. Pas de limite de durée : seul le stop (ou un garde-fou du
  gestionnaire de risque : kill switch, limite journalière) coupe un trade qui tourne mal.

**Stop.** Calculé à l'entrée : `close − atr_mult × ATR`, puis porté tel quel sur chaque
bougie en position. Après un stop touché, la stratégie peut encore dire « achète » à la
bougie suivante si le prix reste sous la bande basse ; le robot attend qu'elle soit repassée
à plat avant de racheter (politique `wait`) : dans un krach, cela évite d'enchaîner les
stop-outs à chaque bougie.

**Marché favorable / défavorable.** Marchés qui oscillent (range, consolidation). Dans
un krach, le prix peut *rester* sous la bande basse pendant des jours : la stratégie
achète, se fait couper par le stop, rachète plus bas, etc. Le RSI limite un peu ces
cas, pas totalement.

| Paramètre | Défaut | Signification |
|---|---|---|
| `window` | 20 | longueur de la moyenne et de l'écart-type |
| `n_std` | 2.0 | largeur des bandes, en écarts-types |
| `rsi_window` | 14 | longueur du RSI |
| `rsi_max` | 35 | RSI maximal pour entrer (confirmation de l'excès de baisse) |
| `atr_window` | 14 | longueur de l'ATR |
| `atr_mult` | 2.0 | distance du stop, en multiples d'ATR |

---

## Points communs à toutes les stratégies

- **Aucune ne regarde dans le futur.** Le signal de la bougie `t` est calculé avec les
  données jusqu'à `t` incluse et exécuté à l'ouverture de `t + 1`. Les tests vérifient
  que les signaux sur les `k` premières bougies sont identiques, qu'on fournisse `k`
  bougies ou toute la série.
- **Signaux stables.** Une position est conservée jusqu'à une condition de sortie
  explicite ; aucune stratégie ne bascule entre 1 et 0 à chaque bougie sur du bruit.
- **Un stop à chaque bougie en position.** Chaque stratégie exporte son `stop_loss` sur
  **toutes** les bougies où son signal n'est pas nul (stop d'entrée constant pour
  `sma_cross`, `donchian_breakout`, `rsi_reversion`, `bollinger_reversion` ; stop suiveur pour
  `ema_trend`). Un moteur ou un robot qui entre en cours de position dispose donc toujours
  du stop de la stratégie.
- **Jamais d'entrée sans stop, taille basée sur le risque.** Avant chaque entrée, le
  gestionnaire de risque choisit le stop dans cet ordre : (1) le stop de la stratégie s'il
  est du bon côté du prix d'exécution ; (2) `stop_loss_pct` s'il est configuré dans le
  moteur de backtest ; (3) un stop de secours à `default_stop_atr_mult × ATR(14)` (3 × ATR
  par défaut) ; (4) un stop de dernier recours à `default_stop_pct` (5 %) du prix, avec un
  avertissement. La taille de la position vaut `capital × risk_per_trade ÷ distance au
  stop`, plafonnée par `max_position_fraction`.
- **Ré-entrée après une sortie forcée (`wait`).** Après un stop, un objectif atteint ou la
  limite de perte journalière, le moteur et le robot n'ouvrent pas à nouveau du même côté
  tant que le signal brut de la stratégie n'est pas repassé par 0 (ou ne s'est pas retourné)
  sur une bougie ultérieure. `sma_cross` est le cas le plus visible : son signal reste `1`
  tant que la moyenne courte est au-dessus de la longue, donc après un stop elle ne rentre
  qu'au prochain croisement. L'ancien comportement (racheter dès la bougie suivante) reste
  disponible avec `reentry_after_stop = "immediate"` dans le moteur de backtest.
- **Période de chauffe.** `warmup_bars()` = plus longue fenêtre utilisée + une marge
  (doublée pour les EMA et l'ADX, qui convergent lentement). Avant cela, signal `0`.
- **Paramètres.** `--param nom=valeur` en ligne de commande, ou `[strategy.params]` dans
  `config.toml`. Un nom inconnu ou une valeur absurde (`fast >= slow`, fenêtre nulle,
  multiplicateur négatif…) provoque une erreur explicite.
- **Grille d'optimisation.** Chaque stratégie expose une petite grille (`param_grid`,
  au plus quelques dizaines de combinaisons) utilisée par `tradingbot optimize`
  (walk-forward). Volontairement petite : voir la section suivante.

---

## Aucune stratégie ne gagne toujours

C'est la section la plus importante de ce document.

### 1. Chaque stratégie dépend du régime de marché

Un marché alterne des **régimes** : tendance haussière, tendance baissière, range calme,
range nerveux, krach. Les stratégies de tendance gagnent dans les deux premiers et
perdent dans les ranges ; les stratégies de retour à la moyenne font l'inverse ; toutes
souffrent dans un krach. Personne ne sait à l'avance quel régime vient ensuite.
Conséquences pratiques :

- Un backtest brillant sur 2020-2021 (hausse quasi continue des cryptos) ne dit rien sur
  2022 (baisse) ni sur un marché plat.
- Testez toujours sur **plusieurs années** et plusieurs actifs, et regardez le pire
  passage (`max_drawdown`), pas seulement le rendement total.
- Une stratégie peut passer six mois à perdre un peu tous les jours, puis tout rattraper
  en trois semaines. Si vous l'arrêtez au mauvais moment, vous ne gardez que les pertes.

### 2. Le sur-apprentissage (overfitting) : le piège numéro un

Avec assez de paramètres, on peut toujours trouver une combinaison qui aurait gagné
**dans le passé**. Un `fast = 17, slow = 63, atr_mult = 2.7` qui bat tout le monde sur
l'historique a presque sûrement été ajusté au bruit de cet historique, pas à une vraie
régularité du marché. Il perdra demain.

Comment s'en protéger (ce que fait `tradingbot optimize`) :

- **Walk-forward** : on optimise sur une période (*in-sample*), puis on vérifie sur la
  période **suivante** jamais vue (*out-of-sample*), et on avance. Seul le résultat
  out-of-sample compte. Si la performance in-sample est bien meilleure que l'out-of-sample
  (ratio OOS / IS **très inférieur à 1**, nul ou négatif), méfiez-vous : un ratio proche de
  1 est robuste, un ratio bas signe le sur-apprentissage (même lecture dans le README, le
  guide du débutant et le compte rendu de `optimize`).
- **Peu de paramètres, petites grilles**, valeurs « rondes » : si la stratégie ne marche
  qu'avec `slow = 63` et pas avec `50` ou `100`, elle ne marche pas.
- **Se méfier de tout ce qui est trop beau** : un Sharpe > 3 ou une courbe d'équité sans
  creux sur plusieurs années sont presque toujours un bug (look-ahead) ou un sur-apprentissage.

### 3. Les coûts mangent les petites stratégies

Chaque ordre coûte des frais (`fee_rate`, ~0,1 % sur un exchange spot) et du
glissement (`slippage` : on est exécuté un peu plus mal que le prix affiché). Un
aller-retour coûte donc environ 0,2-0,3 % du montant. Une stratégie qui fait 500 trades
par an doit gagner plus de 100-150 % brut pour simplement couvrir ses coûts.

- Les stratégies lentes (bougies journalières, peu de trades) résistent mieux aux coûts
  que les stratégies rapides (bougies de 5 minutes).
- Le backtest inclut toujours frais et glissement ; ne les mettez jamais à zéro « pour voir ».
- Le `min_trade_notional` évite de passer des ordres si petits que les frais fixes
  dépasseraient le gain espéré.

### 4. Le risque d'abord, le rendement ensuite

Le robot ne gagne pas parce qu'il devine le marché, mais parce qu'il **perd peu** quand
il a tort (stop + taille de position proportionnelle au risque) et **laisse courir**
quand il a raison. Les réglages de `[risk]` (`risk_per_trade`, `max_drawdown`,
`daily_loss_limit`) comptent plus que le choix de la stratégie. Commencez en mode
*paper* (argent fictif), pendant des semaines, avant d'envisager le mode *live* ; et
n'y mettez jamais une somme dont la perte totale changerait votre vie.
