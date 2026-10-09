"""Indicateurs techniques : fonctions pures et vectorisées (pandas / numpy).

Toutes les fonctions renvoient une ``pd.Series`` (ou un ``pd.DataFrame``) alignée sur
l'index de l'entrée, avec ``NaN`` pendant la période de chauffe. Aucun indicateur ne
regarde dans le futur : la valeur à la bougie ``t`` ne dépend que des données ``<= t``
(pas de ``shift(-1)``, pas de ``center=True``, pas de normalisation sur toute la série).

Conventions :

- ``series`` : une ``pd.Series`` de prix (en général la clôture).
- ``df`` : un DataFrame OHLCV avec au moins les colonnes ``high``, ``low``, ``close``.
- Le lissage « Wilder » (RSI, ATR, ADX) est amorcé par la moyenne simple des ``window``
  premières valeurs, puis poursuivi par ``ewm(alpha=1/window, adjust=False)``, exactement
  comme dans la définition originale de J. Welles Wilder (et comme TA-Lib).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "adx",
    "atr",
    "bollinger",
    "crossover",
    "crossunder",
    "donchian",
    "ema",
    "macd",
    "returns",
    "rolling_volatility",
    "rsi",
    "sma",
    "true_range",
    "wilder_smooth",
]


# --- utilitaires internes ----------------------------------------------------------


def _check_window(window: int, name: str = "window") -> int:
    """Valide une taille de fenêtre (entier >= 1) et la renvoie."""
    if isinstance(window, bool) or not isinstance(window, int | np.integer):
        raise TypeError(f"`{name}` doit être un entier, reçu {type(window).__name__}")
    if window < 1:
        raise ValueError(f"`{name}` doit être >= 1, reçu {window}")
    return int(window)


def _as_float_series(series: pd.Series, name: str = "series") -> pd.Series:
    """Convertit l'entrée en ``pd.Series`` float64 (nouvelle série, l'entrée n'est pas modifiée)."""
    if not isinstance(series, pd.Series):
        raise TypeError(f"`{name}` doit être une pd.Series, reçu {type(series).__name__}")
    return series.astype("float64")


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> None:
    """Lève ``ValueError`` si une colonne OHLCV requise manque."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"`df` doit être un pd.DataFrame, reçu {type(df).__name__}")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes OHLCV manquantes pour cet indicateur : {missing}")


def _hlc(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Renvoie (high, low, close) en float64."""
    _require_columns(df, ("high", "low", "close"))
    return (
        df["high"].astype("float64"),
        df["low"].astype("float64"),
        df["close"].astype("float64"),
    )


def _align_pair(a: pd.Series, b: pd.Series | float | int) -> tuple[pd.Series, pd.Series]:
    """Aligne ``b`` sur l'index de ``a`` (un scalaire est diffusé sur tout l'index)."""
    a = _as_float_series(a, "a")
    if isinstance(b, pd.Series):
        b = b.astype("float64")
        if not b.index.equals(a.index):
            b = b.reindex(a.index)
    else:
        b = pd.Series(float(b), index=a.index, dtype="float64")
    return a, b


# --- briques de base ----------------------------------------------------------------


def wilder_smooth(series: pd.Series, window: int) -> pd.Series:
    """Lissage de Wilder (« RMA ») : amorçage par la moyenne simple des ``window`` premières
    valeurs valides, puis ``y[t] = y[t-1] + (x[t] - y[t-1]) / window``.

    Équivalent à ``ewm(alpha=1/window, adjust=False)`` une fois amorcé. ``NaN`` tant que
    ``window`` valeurs consécutives ne sont pas disponibles. Les ``NaN`` de tête de
    ``series`` (ex. issus d'un ``diff``) décalent simplement l'amorçage.
    """
    window = _check_window(window)
    x = _as_float_series(series)
    values = x.to_numpy(dtype="float64", copy=True)
    seed = x.rolling(window, min_periods=window).mean().to_numpy(dtype="float64")
    valid = np.flatnonzero(~np.isnan(seed))
    if valid.size == 0:
        return pd.Series(np.full(len(x), np.nan), index=x.index, name=x.name)
    start = int(valid[0])
    # Avant l'amorçage : NaN ; à l'amorçage : la moyenne simple ; ensuite les valeurs brutes,
    # que le lissage exponentiel (adjust=False) enchaîne exactement comme la récurrence.
    values[:start] = np.nan
    values[start] = seed[start]
    seeded = pd.Series(values, index=x.index, name=x.name)
    return seeded.ewm(alpha=1.0 / window, adjust=False).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    """Moyenne mobile simple sur ``window`` bougies (``NaN`` pour les ``window-1`` premières)."""
    window = _check_window(window)
    s = _as_float_series(series)
    out = s.rolling(window, min_periods=window).mean()
    return out.rename(f"sma_{window}")


def ema(series: pd.Series, span: int) -> pd.Series:
    """Moyenne mobile exponentielle : ``ewm(span=span, adjust=False)``.

    Récurrence : ``ema[0] = x[0]`` puis ``ema[t] = alpha * x[t] + (1 - alpha) * ema[t-1]``
    avec ``alpha = 2 / (span + 1)``. Définie dès la première valeur valide (pas de ``NaN``
    de chauffe au-delà des ``NaN`` de tête de l'entrée) : c'est à la stratégie de déclarer
    un ``warmup_bars()`` suffisant pour que l'EMA soit stabilisée.
    """
    span = _check_window(span, "span")
    s = _as_float_series(series)
    out = s.ewm(span=span, adjust=False).mean()
    return out.rename(f"ema_{span}")


def returns(series: pd.Series, log: bool = False) -> pd.Series:
    """Rendements d'une bougie à la suivante : ``p[t]/p[t-1] - 1`` ou, si ``log``,
    ``ln(p[t]/p[t-1])``. ``NaN`` à la première bougie (et là où le prix n'est pas > 0
    pour les rendements log).
    """
    s = _as_float_series(series)
    if log:
        out = np.log(s.where(s > 0)).diff()
        return out.rename("log_returns")
    out = s / s.shift(1) - 1.0
    return out.rename("returns")


def true_range(df: pd.DataFrame) -> pd.Series:
    """True range : ``max(high - low, |high - close[t-1]|, |low - close[t-1]|)``.

    ``NaN`` à la première bougie (pas de clôture précédente).
    """
    high, low, close = _hlc(df)
    prev_close = close.shift(1)
    ranges = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    )
    tr = ranges.max(axis=1, skipna=True)
    # Sans clôture précédente, le true range n'est pas défini.
    tr = tr.mask(prev_close.isna())
    return tr.rename("true_range")


# --- indicateurs --------------------------------------------------------------------


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """RSI de Wilder, borné dans ``[0, 100]``.

    Gains / pertes = variations positives / négatives du prix, lissés par la méthode de
    Wilder (moyenne simple des ``window`` premières variations, puis
    ``ewm(alpha=1/window, adjust=False)``). ``RSI = 100 * gain_moy / (gain_moy + perte_moy)``
    (identique à ``100 - 100/(1 + RS)``). Vaut ``50`` lorsque gain et perte moyens sont
    tous deux nuls (prix parfaitement plat). ``NaN`` pour les ``window`` premières bougies.
    """
    window = _check_window(window)
    s = _as_float_series(series)
    delta = s.diff()
    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)
    avg_gain = wilder_smooth(gains, window)
    avg_loss = wilder_smooth(losses, window)
    denom = avg_gain + avg_loss
    out = 100.0 * avg_gain / denom.where(denom != 0.0)
    # Marché parfaitement plat : ni gain ni perte -> 50 (NaN de chauffe conservés).
    out = out.mask(denom == 0.0, 50.0)
    return out.clip(lower=0.0, upper=100.0).rename(f"rsi_{window}")


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range : true range lissé par la méthode de Wilder.

    Le true range de la première bougie est indéfini (pas de clôture précédente), donc
    l'ATR est ``NaN`` pour les ``window`` premières bougies et vaut, à la bougie
    ``window``, la moyenne simple des true ranges des bougies ``1..window``.
    """
    window = _check_window(window)
    tr = true_range(df)
    return wilder_smooth(tr, window).rename(f"atr_{window}")


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD : ``macd = ema(fast) - ema(slow)``, ``signal = ema(macd, signal)``,
    ``hist = macd - signal``. Colonnes : ``macd``, ``signal``, ``hist``.
    """
    fast = _check_window(fast, "fast")
    slow = _check_window(slow, "slow")
    signal = _check_window(signal, "signal")
    if fast >= slow:
        raise ValueError(f"`fast` ({fast}) doit être strictement inférieur à `slow` ({slow})")
    s = _as_float_series(series)
    macd_line = ema(s, fast) - ema(s, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "hist": hist},
        index=s.index,
    )


def bollinger(series: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    """Bandes de Bollinger. Colonnes : ``middle`` (SMA), ``upper``, ``lower``,
    ``pct_b = (prix - lower) / (upper - lower)`` et ``bandwidth = (upper - lower) / middle``.

    L'écart-type est celui de la population (``ddof=0``), comme dans la définition de
    John Bollinger. Si les bandes sont de largeur nulle (prix constant sur la fenêtre),
    ``pct_b`` vaut ``0.5`` ; ``bandwidth`` est ``NaN`` si ``middle`` est nul.
    """
    window = _check_window(window)
    if n_std < 0:
        raise ValueError(f"`n_std` doit être >= 0, reçu {n_std}")
    s = _as_float_series(series)
    middle = s.rolling(window, min_periods=window).mean()
    std = s.rolling(window, min_periods=window).std(ddof=0)
    upper = middle + n_std * std
    lower = middle - n_std * std
    width = upper - lower
    pct_b = (s - lower) / width.where(width != 0.0)
    pct_b = pct_b.mask((width == 0.0) & middle.notna(), 0.5)
    bandwidth = width / middle.where(middle != 0.0)
    return pd.DataFrame(
        {
            "middle": middle,
            "upper": upper,
            "lower": lower,
            "pct_b": pct_b,
            "bandwidth": bandwidth,
        },
        index=s.index,
    )


def donchian(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Canal de Donchian calculé sur les ``window`` bougies **précédentes** (``shift(1)``) :
    ``upper = max(high)``, ``lower = min(low)``, ``middle = (upper + lower) / 2``.

    La bougie courante est exclue afin qu'elle ne puisse pas déclencher sa propre
    cassure (un ``close > upper`` est alors une vraie cassure du canal passé).
    ``NaN`` pour les ``window`` premières bougies.
    """
    window = _check_window(window)
    _require_columns(df, ("high", "low"))
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    upper = high.rolling(window, min_periods=window).max().shift(1)
    lower = low.rolling(window, min_periods=window).min().shift(1)
    middle = (upper + lower) / 2.0
    return pd.DataFrame({"upper": upper, "lower": lower, "middle": middle}, index=df.index)


def adx(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """ADX de Wilder. Colonnes : ``adx``, ``plus_di``, ``minus_di`` (toutes dans ``[0, 100]``).

    - ``+DM = high[t] - high[t-1]`` si c'est le plus grand des deux mouvements et > 0, sinon 0 ;
      ``-DM = low[t-1] - low[t]`` symétriquement.
    - ``+DI = 100 * lissage(+DM) / lissage(TR)`` et ``-DI`` de même (lissage de Wilder).
    - ``DX = 100 * |+DI - -DI| / (+DI + -DI)`` (0 si les deux DI sont nuls) et
      ``ADX = lissage(DX)``.

    ``+DI``/``-DI`` sont définis à partir de la bougie ``window`` et ``ADX`` à partir de la
    bougie ``2*window - 1``.
    """
    window = _check_window(window)
    high, low, close = _hlc(df)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0.0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0.0), 0.0)
    # Première bougie : pas de mouvement directionnel défini (comme pour le true range).
    plus_dm = plus_dm.mask(up_move.isna())
    minus_dm = minus_dm.mask(down_move.isna())

    tr = true_range(df)
    sm_tr = wilder_smooth(tr, window)
    sm_plus = wilder_smooth(plus_dm, window)
    sm_minus = wilder_smooth(minus_dm, window)

    plus_di = 100.0 * sm_plus / sm_tr.where(sm_tr != 0.0)
    minus_di = 100.0 * sm_minus / sm_tr.where(sm_tr != 0.0)
    # True range nul sur toute la fenêtre (prix figé) : aucune direction -> 0.
    flat = (sm_tr == 0.0) & sm_tr.notna()
    plus_di = plus_di.mask(flat, 0.0)
    minus_di = minus_di.mask(flat, 0.0)

    di_sum = plus_di + minus_di
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum.where(di_sum != 0.0)
    dx = dx.mask((di_sum == 0.0) & di_sum.notna(), 0.0)
    adx_line = wilder_smooth(dx, window)

    out = pd.DataFrame(
        {"adx": adx_line, "plus_di": plus_di, "minus_di": minus_di},
        index=df.index,
    )
    return out.clip(lower=0.0, upper=100.0)


def rolling_volatility(
    series: pd.Series, window: int = 20, periods_per_year: float | None = None
) -> pd.Series:
    """Écart-type glissant (``ddof=1``) des rendements log sur ``window`` rendements,
    multiplié par ``sqrt(periods_per_year)`` si fourni (volatilité annualisée).

    ``NaN`` pour les ``window`` premières bougies (il faut ``window`` rendements).
    """
    window = _check_window(window)
    if periods_per_year is not None and periods_per_year <= 0:
        raise ValueError(f"`periods_per_year` doit être > 0, reçu {periods_per_year}")
    log_ret = returns(series, log=True)
    vol = log_ret.rolling(window, min_periods=window).std(ddof=1)
    if periods_per_year is not None:
        vol = vol * np.sqrt(float(periods_per_year))
    return vol.rename(f"volatility_{window}")


def crossover(a: pd.Series, b: pd.Series | float | int) -> pd.Series:
    """``True`` à la bougie où ``a`` passe au-dessus de ``b`` : ``a[t] > b[t]`` et
    ``a[t-1] <= b[t-1]``. ``b`` peut être un scalaire (ex. ``crossover(rsi, 30)``).
    Toute comparaison impliquant un ``NaN`` vaut ``False``.
    """
    a, b = _align_pair(a, b)
    above_now = a > b
    not_above_before = a.shift(1) <= b.shift(1)
    return (above_now & not_above_before).astype(bool).rename("crossover")


def crossunder(a: pd.Series, b: pd.Series | float | int) -> pd.Series:
    """``True`` à la bougie où ``a`` passe en dessous de ``b`` : ``a[t] < b[t]`` et
    ``a[t-1] >= b[t-1]``. Symétrique de :func:`crossover`.
    """
    a, b = _align_pair(a, b)
    below_now = a < b
    not_below_before = a.shift(1) >= b.shift(1)
    return (below_now & not_below_before).astype(bool).rename("crossunder")
