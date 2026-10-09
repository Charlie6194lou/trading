"""tradingbot — robot de trading pédagogique.

Modules :
- ``data``       : chargement / téléchargement / génération de bougies OHLCV
- ``indicators`` : indicateurs techniques (SMA, EMA, RSI, ATR, ...)
- ``strategies`` : stratégies qui transforment des bougies en signaux
- ``risk``       : taille des positions et garde-fous (drawdown, perte journalière)
- ``backtest``   : simulation historique, métriques, walk-forward
- ``execution``  : brokers (paper = fictif, ccxt = réel)
- ``live``       : boucle temps réel
- ``cli``        : interface en ligne de commande
"""

__version__ = "0.1.0"
