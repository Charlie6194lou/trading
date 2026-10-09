"""Tests des cinq stratégies : registre, paramètres, format des signaux, chauffe,
causalité (aucun look-ahead), stops ATR, comportement en marché haussier / baissier /
plat, et valeurs calculées à la main sur de petites séries.

Aucun accès réseau. Seules dépendances projet : ``tradingbot.indicators`` (pour recalculer
indépendamment les conditions d'entrée / sortie) et ``tradingbot.strategies``.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
import pandas as pd
import pytest
from conftest import make_ohlcv

from tradingbot import indicators as ind
from tradingbot.strategies import STRATEGIES, Strategy, get_strategy, list_strategies
from tradingbot.strategies.base import SIGNAL_COLUMNS

NAMES = ["sma_cross", "ema_trend", "rsi_reversion", "donchian_breakout", "bollinger_reversion"]
TREND_FOLLOWING = ["sma_cross", "ema_trend", "donchian_breakout"]
LONG_ONLY = ["rsi_reversion", "bollinger_reversion"]

#: période de chauffe attendue avec les paramètres par défaut (plus longue fenêtre + marge 5,
#: fenêtre doublée pour les EMA / l'ADX qui convergent lentement)
EXPECTED_WARMUP = {
    "sma_cross": 55,  # max(slow=50, atr_window+1=15) + 5
    "ema_trend": 105,  # max(2*slow=100, 2*adx_window=28, atr_window+1=15) + 5
    "rsi_reversion": 205,  # max(trend_window=200, rsi_window+1, atr_window+1) + 5
    "donchian_breakout": 25,  # max(entry=20, exit=10, atr=14) + 5
    "bollinger_reversion": 25,  # max(window=20, rsi_window+1, atr_window+1) + 5
}


# --- helpers ------------------------------------------------------------------------


def _bars(closes: list[float], wiggle: float = 1.0) -> pd.DataFrame:
    """Bougies à partir de clôtures : ``high = close + wiggle``, ``low = close - wiggle``,
    ``open`` = clôture précédente. Avec ``wiggle = 1`` le true range vaut ``max(2, |Δclose| + 1)``.
    """
    c = np.asarray(closes, dtype="float64")
    idx = pd.date_range("2024-01-01", periods=len(c), freq="1h", tz="UTC")
    open_ = np.concatenate([[c[0]], c[:-1]])
    return pd.DataFrame(
        {"open": open_, "high": c + wiggle, "low": c - wiggle, "close": c, "volume": 1.0},
        index=idx,
    )


def _flat_ohlcv(n: int = 300) -> pd.DataFrame:
    """Prix parfaitement plat (100) avec une petite amplitude fixe : aucun indicateur ne
    doit déclencher quoi que ce soit."""
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1.0}, index=idx
    )


def _dip_ohlcv(n: int = 900, seed: int = 5) -> pd.DataFrame:
    """Tendance haussière en dents de scie : 30 bougies à +0,4 % puis 8 bougies à -1,2 %
    (+ un peu de bruit). Les replis font passer le RSI sous 30 alors que le prix reste
    au-dessus de sa moyenne à 200 bougies : terrain de jeu des stratégies de retour à la
    moyenne, qui n'entrent presque jamais sur une marche aléatoire pure."""
    rng = np.random.default_rng(seed)
    pattern: list[float] = []
    while len(pattern) < n:
        pattern += [0.004] * 30 + [-0.012] * 8
    rets = np.asarray(pattern[:n]) + rng.normal(0.0, 0.002, size=n)
    close = 100.0 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[100.0], close[:-1]])
    wig = np.abs(rng.normal(0.0, 0.001, size=n)) * close
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + wig,
            "low": np.minimum(open_, close) - wig,
            "close": close,
            "volume": 1.0,
        },
        index=idx,
    )


def _datasets() -> dict[str, pd.DataFrame]:
    return {
        "ohlcv": make_ohlcv(),
        "trending": make_ohlcv(n=800, seed=7, drift=0.002, vol=0.005),
        "falling": make_ohlcv(n=800, seed=7, drift=-0.002, vol=0.005),
        "daily": make_ohlcv(n=1095, seed=3, freq="1D", vol=0.02),
        "dips": _dip_ohlcv(),
    }


def _prev(sig: np.ndarray) -> np.ndarray:
    return np.concatenate(([0], sig[:-1]))


def _entries(sig: np.ndarray) -> np.ndarray:
    """Bougies d'entrée : signal non nul et différent du signal précédent."""
    return (sig != 0) & (sig != _prev(sig))


def _runs(sig: np.ndarray) -> list[tuple[int, int, int]]:
    """Segments de position : liste de ``(début, fin exclue, valeur)`` pour chaque suite
    de signaux non nuls identiques."""
    runs: list[tuple[int, int, int]] = []
    t = 0
    n = len(sig)
    while t < n:
        if sig[t] == 0:
            t += 1
            continue
        u = t
        while u < n and sig[u] == sig[t]:
            u += 1
        runs.append((t, u, int(sig[t])))
        t = u
    return runs


def _transitions(sig: np.ndarray) -> int:
    return int((np.diff(sig) != 0).sum())


# --- registre et paramètres ---------------------------------------------------------


def test_registry_contains_all_five() -> None:
    assert set(NAMES) <= set(STRATEGIES)
    for name in NAMES:
        cls = STRATEGIES[name]
        assert issubclass(cls, Strategy)
        assert cls.name == name
        inst = get_strategy(name)
        assert isinstance(inst, cls)
        assert inst.name == name


def test_list_strategies_sorted_with_french_descriptions() -> None:
    listed = list_strategies()
    names = [n for n, _ in listed]
    assert names == sorted(names)
    assert set(NAMES) <= set(names)
    for name, desc in listed:
        if name in NAMES:
            assert isinstance(desc, str) and len(desc) > 30, name
            assert desc.endswith("."), name
            # Une phrase (pas de retour à la ligne, pas de point au milieu).
            assert "\n" not in desc and desc.count(". ") == 0, name


def test_get_strategy_unknown_name_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        get_strategy("inexistante")


@pytest.mark.parametrize("name", NAMES)
def test_param_grid_keys_subset_of_defaults(name: str) -> None:
    cls = STRATEGIES[name]
    assert set(cls.param_grid) <= set(cls.default_params)
    assert cls.param_grid, "la grille ne doit pas être vide"


@pytest.mark.parametrize("name", NAMES)
def test_param_grid_small_and_every_combination_valid(name: str) -> None:
    cls = STRATEGIES[name]
    sizes = [len(v) for v in cls.param_grid.values()]
    assert all(2 <= s <= 4 for s in sizes), sizes
    assert math.prod(sizes) <= 36
    keys = list(cls.param_grid)
    for combo in itertools.product(*(cls.param_grid[k] for k in keys)):
        params = dict(zip(keys, combo, strict=True))
        inst = cls(**params)  # ne doit pas lever
        for k, v in params.items():
            assert inst.params[k] == v


@pytest.mark.parametrize("name", NAMES)
def test_unknown_param_raises_value_error(name: str) -> None:
    with pytest.raises(ValueError, match="inconnus"):
        get_strategy(name, parametre_bidon=3)
    with pytest.raises(ValueError):
        STRATEGIES[name](fast_window=1)


@pytest.mark.parametrize("name", NAMES)
def test_params_merge_with_defaults(name: str) -> None:
    cls = STRATEGIES[name]
    key = next(iter(cls.param_grid))
    value = next(v for v in cls.param_grid[key] if v != cls.default_params[key])
    inst = get_strategy(name, **{key: value})
    assert inst.params[key] == value
    for k, v in cls.default_params.items():
        if k != key:
            assert inst.params[k] == v
    assert set(inst.params) == set(cls.default_params)
    assert cls.__name__ in repr(inst)


def test_integral_float_params_are_coerced_to_int() -> None:
    inst = get_strategy("sma_cross", fast=10.0, slow=30)
    assert inst.params["fast"] == 10 and isinstance(inst.params["fast"], int)
    with pytest.raises(ValueError, match="entier"):
        get_strategy("sma_cross", fast=10.5)
    with pytest.raises(ValueError):
        get_strategy("sma_cross", fast="20")
    with pytest.raises(ValueError):
        get_strategy("sma_cross", fast=True)


@pytest.mark.parametrize(
    ("name", "bad"),
    [
        ("sma_cross", {"fast": 50, "slow": 50}),
        ("sma_cross", {"fast": 60}),
        ("sma_cross", {"fast": 0}),
        ("sma_cross", {"atr_mult": 0.0}),
        ("sma_cross", {"atr_window": -1}),
        ("ema_trend", {"fast": 50}),
        ("ema_trend", {"adx_threshold": -1.0}),
        ("ema_trend", {"adx_threshold": 100.0}),
        ("ema_trend", {"atr_mult": -3.0}),
        ("rsi_reversion", {"oversold": 60.0}),
        ("rsi_reversion", {"oversold": 0.0}),
        ("rsi_reversion", {"exit_level": 101.0}),
        ("rsi_reversion", {"max_hold": 0}),
        ("rsi_reversion", {"trend_window": 0}),
        ("donchian_breakout", {"entry_window": 0}),
        ("donchian_breakout", {"atr_mult": -2.0}),
        ("donchian_breakout", {"exit_window": 2.5}),
        ("bollinger_reversion", {"n_std": 0.0}),
        ("bollinger_reversion", {"rsi_max": 0.0}),
        ("bollinger_reversion", {"rsi_max": 100.0}),
        ("bollinger_reversion", {"window": 1}),
        ("bollinger_reversion", {"atr_mult": float("nan")}),
    ],
)
def test_invalid_param_values_raise_value_error(name: str, bad: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        get_strategy(name, **bad)


@pytest.mark.parametrize("name", NAMES)
def test_warmup_bars_defaults_and_scaling(name: str) -> None:
    assert get_strategy(name).warmup_bars() == EXPECTED_WARMUP[name]


def test_warmup_bars_follow_longest_window() -> None:
    assert get_strategy("sma_cross", slow=100).warmup_bars() == 105
    assert get_strategy("sma_cross", slow=10, fast=5, atr_window=30).warmup_bars() == 36
    assert get_strategy("ema_trend", slow=100).warmup_bars() == 205
    assert get_strategy("rsi_reversion", trend_window=50).warmup_bars() == 55
    assert get_strategy("donchian_breakout", entry_window=55).warmup_bars() == 60
    assert get_strategy("bollinger_reversion", window=30).warmup_bars() == 35


# --- format de sortie, chauffe, causalité --------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_output_format(name: str, ohlcv: pd.DataFrame) -> None:
    before = ohlcv.copy()
    strat = get_strategy(name)
    out = strat.generate_signals(ohlcv)
    pd.testing.assert_frame_equal(ohlcv, before)  # l'entrée n'est pas modifiée

    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == list(SIGNAL_COLUMNS)
    assert out.index.equals(ohlcv.index)
    assert len(out) == len(ohlcv)
    assert out["signal"].dtype == np.int64
    assert out["stop_loss"].dtype == np.float64
    assert out["take_profit"].dtype == np.float64
    assert out["signal"].isin([-1, 0, 1]).all()
    assert out["take_profit"].isna().all()
    # Passe la validation du contrat sans être modifié par elle.
    pd.testing.assert_frame_equal(Strategy.validate_signals(ohlcv, out), out)


@pytest.mark.parametrize("name", NAMES)
def test_short_input_is_all_flat(name: str, ohlcv: pd.DataFrame) -> None:
    strat = get_strategy(name)
    for k in (1, 5, strat.warmup_bars()):
        out = strat.generate_signals(ohlcv.iloc[:k])
        assert len(out) == k
        assert (out["signal"] == 0).all()
        assert out["stop_loss"].isna().all()


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("dataset", ["ohlcv", "dips", "daily"])
def test_signal_zero_and_no_stop_during_warmup(name: str, dataset: str) -> None:
    df = _datasets()[dataset]
    strat = get_strategy(name)
    out = strat.generate_signals(df)
    w = strat.warmup_bars()
    assert (out["signal"].iloc[:w] == 0).all()
    assert out["stop_loss"].iloc[:w].isna().all()
    # Les stratégies ne restent pas inertes pour autant (sauf rsi_reversion sur une marche
    # aléatoire sans replis, ce qui est son comportement attendu).
    if not (name == "rsi_reversion" and dataset != "dips"):
        assert (out["signal"] != 0).any()


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("dataset", ["ohlcv", "dips"])
def test_causality_prefix_signals_equal_full_signals(name: str, dataset: str) -> None:
    df = _datasets()[dataset]
    strat = get_strategy(name)
    full = strat.generate_signals(df)
    w = strat.warmup_bars()
    n = len(df)
    for k in (w + 1, w + 40, w + 137, n // 2, n - 1, n):
        part = strat.generate_signals(df.iloc[:k])
        pd.testing.assert_frame_equal(part, full.iloc[:k], check_exact=True)


@pytest.mark.parametrize("name", NAMES)
def test_no_look_ahead_future_shock_leaves_past_unchanged(name: str) -> None:
    """Un choc violent sur les 60 dernières bougies ne change aucun signal antérieur."""
    df = _datasets()["dips" if name in LONG_ONLY else "ohlcv"]
    n = len(df)
    shocked = df.copy()
    cols = ["open", "high", "low", "close"]
    shocked.loc[shocked.index[n - 60 :], cols] = shocked.loc[shocked.index[n - 60 :], cols] * 1.5
    strat = get_strategy(name)
    base = strat.generate_signals(df)
    after = strat.generate_signals(shocked)
    pd.testing.assert_frame_equal(base.iloc[: n - 60], after.iloc[: n - 60], check_exact=True)


# --- stops ATR --------------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_entry_bars_carry_atr_stop(name: str) -> None:
    strat = get_strategy(name)
    p = strat.params
    total_entries = 0
    for dataset, df in _datasets().items():
        out = strat.generate_signals(df)
        sig = out["signal"].to_numpy()
        stop = out["stop_loss"].to_numpy()
        close = df["close"].to_numpy()
        atr = ind.atr(df, p["atr_window"]).to_numpy()
        entry = _entries(sig)
        total_entries += int(entry.sum())

        expected = close - sig * p["atr_mult"] * atr
        assert np.isfinite(stop[entry]).all(), dataset
        np.testing.assert_allclose(stop[entry], expected[entry], rtol=1e-12)
        long_entry = entry & (sig == 1)
        short_entry = entry & (sig == -1)
        assert (stop[long_entry] < close[long_entry]).all(), dataset
        assert (stop[short_entry] > close[short_entry]).all(), dataset
        assert np.isnan(stop[sig == 0]).all(), dataset
        if name != "ema_trend":
            assert np.isnan(stop[~entry]).all(), dataset
    assert total_entries > 0


@pytest.mark.parametrize("name", TREND_FOLLOWING)
def test_short_entries_have_stop_above_close(name: str) -> None:
    df = _datasets()["falling"]
    out = get_strategy(name).generate_signals(df)
    sig = out["signal"].to_numpy()
    entry = _entries(sig) & (sig == -1)
    assert entry.sum() >= 1
    assert (out["stop_loss"].to_numpy()[entry] > df["close"].to_numpy()[entry]).all()


@pytest.mark.parametrize("dataset", ["trending", "ohlcv", "daily"])
def test_ema_trend_chandelier_stop_is_running_extreme(dataset: str) -> None:
    df = _datasets()[dataset]
    strat = get_strategy("ema_trend")
    out = strat.generate_signals(df)
    sig = out["signal"].to_numpy()
    stop = out["stop_loss"].to_numpy()
    close = df["close"].to_numpy()
    atr = ind.atr(df, strat.params["atr_window"]).to_numpy()
    base = close - sig * strat.params["atr_mult"] * atr

    runs = _runs(sig)
    assert runs
    assert np.isnan(stop[sig == 0]).all()
    for start, end, side in runs:
        seg = stop[start:end]
        assert np.isfinite(seg).all()
        if side == 1:
            np.testing.assert_allclose(seg, np.maximum.accumulate(base[start:end]), rtol=1e-12)
            assert (np.diff(seg) >= 0).all()
            assert seg[0] < close[start]
        else:
            np.testing.assert_allclose(seg, np.minimum.accumulate(base[start:end]), rtol=1e-12)
            assert (np.diff(seg) <= 0).all()
            assert seg[0] > close[start]


def test_ema_trend_crash_triggers_chandelier_exit_before_ema_flip() -> None:
    df = _datasets()["trending"].iloc[:400].copy()
    last = df.index[-1]
    prev_close = float(df["close"].iloc[-2])
    crash = prev_close * 0.90
    df.loc[last, ["open", "high", "low", "close"]] = [
        prev_close,
        prev_close * 1.001,
        crash * 0.99,
        crash,
    ]
    strat = get_strategy("ema_trend")
    out = strat.generate_signals(df)
    sig = out["signal"].to_numpy()
    assert sig[-2] == 1
    assert sig[-1] == 0
    assert math.isnan(out["stop_loss"].iloc[-1])
    ema_fast = ind.ema(df["close"], 20).iloc[-1]
    ema_slow = ind.ema(df["close"], 50).iloc[-1]
    assert ema_fast > ema_slow  # la sortie vient bien du stop, pas d'un recroisement
    assert crash < out["stop_loss"].iloc[-2]


@pytest.mark.parametrize("dataset", ["ohlcv", "daily", "trending"])
def test_ema_trend_entries_exits_and_adx_rules(dataset: str) -> None:
    df = _datasets()[dataset]
    strat = get_strategy("ema_trend")
    p = strat.params
    out = strat.generate_signals(df)
    sig = out["signal"].to_numpy()
    stop = out["stop_loss"].to_numpy()
    close = df["close"].to_numpy()
    fast = ind.ema(df["close"], p["fast"]).to_numpy()
    slow = ind.ema(df["close"], p["slow"]).to_numpy()
    adx = ind.adx(df, p["adx_window"])["adx"].to_numpy()
    prev = _prev(sig)
    w = strat.warmup_bars()

    entry = _entries(sig)
    assert entry.sum() >= 1
    assert (adx[entry] > p["adx_threshold"]).all()
    assert (np.sign(fast - slow)[entry] == sig[entry]).all()

    kept_with_low_adx = 0
    for t in range(w, len(sig)):
        if prev[t] != 0 and sig[t] != prev[t]:
            flipped = np.sign(fast[t] - slow[t]) == -prev[t]
            breached = (prev[t] == 1 and close[t] < stop[t - 1]) or (
                prev[t] == -1 and close[t] > stop[t - 1]
            )
            assert flipped or breached, t
            if breached and not flipped:
                assert sig[t] == 0  # plat une bougie après une sortie sur stop
        if (
            prev[t] != 0
            and np.sign(fast[t] - slow[t]) == prev[t]
            and not (
                (prev[t] == 1 and close[t] < stop[t - 1])
                or (prev[t] == -1 and close[t] > stop[t - 1])
            )
        ):
            assert sig[t] == prev[t], t  # la position est conservée...
            if adx[t] <= p["adx_threshold"]:
                kept_with_low_adx += 1  # ...même si l'ADX retombe sous le seuil
    assert kept_with_low_adx > 0


# --- comportement par régime de marché ----------------------------------------------


@pytest.mark.parametrize("name", TREND_FOLLOWING)
def test_trend_strategies_mostly_long_in_uptrend(name: str, trending_ohlcv: pd.DataFrame) -> None:
    strat = get_strategy(name)
    sig = strat.generate_signals(trending_ohlcv)["signal"].to_numpy()[strat.warmup_bars() :]
    assert (sig == 1).mean() > 0.5
    assert (sig == -1).mean() < 0.1


@pytest.mark.parametrize("name", TREND_FOLLOWING)
def test_trend_strategies_mostly_short_in_downtrend(name: str) -> None:
    df = _datasets()["falling"]
    strat = get_strategy(name)
    sig = strat.generate_signals(df)["signal"].to_numpy()[strat.warmup_bars() :]
    assert (sig == -1).mean() > 0.5
    assert (sig == 1).mean() < 0.1


@pytest.mark.parametrize("name", LONG_ONLY)
def test_long_only_strategies_never_short(name: str) -> None:
    strat = get_strategy(name)
    for dataset, df in _datasets().items():
        sig = strat.generate_signals(df)["signal"]
        assert sig.min() >= 0, dataset


@pytest.mark.parametrize("name", LONG_ONLY)
def test_mean_reversion_strategies_buy_the_dips(name: str) -> None:
    df = _datasets()["dips"]
    strat = get_strategy(name)
    sig = strat.generate_signals(df)["signal"].to_numpy()
    assert _entries(sig).sum() >= 5
    assert 0.05 < sig[strat.warmup_bars() :].mean() < 0.6


@pytest.mark.parametrize("name", NAMES)
def test_flat_series_produces_no_signal(name: str) -> None:
    out = get_strategy(name).generate_signals(_flat_ohlcv())
    assert (out["signal"] == 0).all()
    assert out["stop_loss"].isna().all()


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("dataset", ["random_walk", "ohlcv", "daily"])
def test_no_flicker_few_transitions_on_noise(name: str, dataset: str) -> None:
    df = (
        make_ohlcv(n=600, seed=11, drift=0.0, vol=0.01)
        if dataset == "random_walk"
        else _datasets()[dataset]
    )
    sig = get_strategy(name).generate_signals(df)["signal"].to_numpy()
    assert _transitions(sig) <= len(df) / 10


@pytest.mark.parametrize("name", NAMES)
def test_positions_are_held_at_least_one_bar_and_not_mostly_one_bar(name: str) -> None:
    df = _datasets()["dips" if name in LONG_ONLY else "ohlcv"]
    sig = get_strategy(name).generate_signals(df)["signal"].to_numpy()
    runs = _runs(sig)
    assert runs
    lengths = np.array([end - start for start, end, _ in runs])
    assert np.median(lengths) >= 3


# --- sorties uniquement sur condition explicite (pas de va-et-vient) --------------------


@pytest.mark.parametrize("dataset", ["ohlcv", "daily"])
def test_sma_cross_transitions_are_true_crossings(dataset: str) -> None:
    df = _datasets()[dataset]
    strat = get_strategy("sma_cross")
    out = strat.generate_signals(df)
    sig = out["signal"].to_numpy()
    fast = ind.sma(df["close"], strat.params["fast"]).to_numpy()
    slow = ind.sma(df["close"], strat.params["slow"]).to_numpy()
    w = strat.warmup_bars()
    assert (sig[w:] != 0).all()  # toujours d'un côté ou de l'autre
    prev = _prev(sig)
    for t in range(w, len(sig)):
        if sig[t] != prev[t]:
            assert np.sign(fast[t] - slow[t]) == sig[t], t
        assert np.sign(fast[t] - slow[t]) in (0, sig[t]), t


def test_sma_cross_holds_position_on_exact_equality() -> None:
    # À t=11 : sma2 = (104+106)/2 = 104 et sma3 = (104+106+102)/3 = 104 exactement.
    df = _bars([100] * 8 + [102, 104, 106, 102, 100])
    strat = get_strategy("sma_cross", fast=2, slow=3, atr_window=2, atr_mult=1.0)
    sma2 = ind.sma(df["close"], 2)
    sma3 = ind.sma(df["close"], 3)
    assert sma2.iloc[11] == sma3.iloc[11]
    out = strat.generate_signals(df)
    assert out["signal"].tolist() == [0] * 8 + [1, 1, 1, 1, -1]
    assert out["stop_loss"].notna().tolist() == [False] * 8 + [True, False, False, False, True]


@pytest.mark.parametrize("dataset", ["ohlcv", "daily"])
def test_donchian_transitions_follow_channel_rules(dataset: str) -> None:
    df = _datasets()[dataset]
    strat = get_strategy("donchian_breakout")
    p = strat.params
    out = strat.generate_signals(df)
    sig = out["signal"].to_numpy()
    close = df["close"].to_numpy()
    entry_ch = ind.donchian(df, p["entry_window"])
    exit_ch = ind.donchian(df, p["exit_window"])
    up_in, lo_in = entry_ch["upper"].to_numpy(), entry_ch["lower"].to_numpy()
    up_out, lo_out = exit_ch["upper"].to_numpy(), exit_ch["lower"].to_numpy()
    prev = _prev(sig)
    w = strat.warmup_bars()
    checked = 0
    for t in range(w, len(sig)):
        if sig[t] == prev[t]:
            if prev[t] == 1:
                assert close[t] >= lo_out[t], t
            elif prev[t] == -1:
                assert close[t] <= up_out[t], t
            else:
                assert lo_in[t] <= close[t] <= up_in[t], t
            continue
        checked += 1
        if prev[t] == 1:
            assert close[t] < lo_out[t], t
        if prev[t] == -1:
            assert close[t] > up_out[t], t
        if sig[t] == 1:
            assert close[t] > up_in[t], t
        if sig[t] == -1:
            assert close[t] < lo_in[t], t
    assert checked > 0


@pytest.mark.parametrize("max_hold", [5, 20])
def test_rsi_reversion_transitions_follow_rules(max_hold: int) -> None:
    df = _datasets()["dips"]
    strat = get_strategy("rsi_reversion", max_hold=max_hold)
    p = strat.params
    out = strat.generate_signals(df)
    sig = out["signal"].to_numpy()
    close = df["close"].to_numpy()
    rsi = ind.rsi(df["close"], p["rsi_window"]).to_numpy()
    trend = ind.sma(df["close"], p["trend_window"]).to_numpy()
    runs = _runs(sig)
    assert len(runs) >= 5
    for start, end, side in runs:
        assert side == 1
        assert rsi[start] < p["oversold"] and close[start] > trend[start]
        assert end - start <= max_hold
        if end < len(sig):
            exit_t = end
            assert rsi[exit_t] > p["exit_level"] or exit_t - start >= max_hold, (start, end)
            assert sig[exit_t] == 0
        for t in range(start + 1, end):
            assert rsi[t] <= p["exit_level"], t
    if max_hold == 5:
        assert all(end - start == max_hold for start, end, _ in runs[:-1])
    # Hors position : jamais de condition d'entrée ignorée.
    prev = _prev(sig)
    for t in range(strat.warmup_bars(), len(sig)):
        if prev[t] == 0 and sig[t] == 0:
            just_exited = t > 0 and sig[t - 1] == 0 and prev[t - 1] == 1
            if not just_exited:
                assert not (rsi[t] < p["oversold"] and close[t] > trend[t]), t


def test_bollinger_transitions_follow_rules() -> None:
    df = _datasets()["dips"]
    strat = get_strategy("bollinger_reversion")
    p = strat.params
    out = strat.generate_signals(df)
    sig = out["signal"].to_numpy()
    close = df["close"].to_numpy()
    bands = ind.bollinger(df["close"], p["window"], p["n_std"])
    lower, middle = bands["lower"].to_numpy(), bands["middle"].to_numpy()
    rsi = ind.rsi(df["close"], p["rsi_window"]).to_numpy()
    runs = _runs(sig)
    assert len(runs) >= 5
    for start, end, side in runs:
        assert side == 1
        assert close[start] < lower[start] and rsi[start] < p["rsi_max"]
        for t in range(start + 1, end):
            assert close[t] < middle[t], t
        if end < len(sig):
            assert close[end] >= middle[end], end
    prev = _prev(sig)
    for t in range(strat.warmup_bars(), len(sig)):
        if prev[t] == 0 and sig[t] == 0 and not (sig[t - 1] == 0 and prev[t - 1] == 1):
            assert not (close[t] < lower[t] and rsi[t] < p["rsi_max"]), t


# --- valeurs calculées à la main ---------------------------------------------------------


def test_sma_cross_hand_computed() -> None:
    """fast=2, slow=3, ATR(2), stop à 1 ATR. Chauffe = max(3, 3) + 5 = 8 bougies.

    Clôtures 100 ×8 puis 102, 104, 106, 105, 103, 101, 100, 103.
    t=8  : sma2=101   > sma3=100.67 → long ; TR=3, ATR=2+(3-2)/2=2.5 → stop 99.5
    t=12 : sma2=104   < sma3=104.67 → short ; ATR : 2.75, 2.875, 2.4375, 2.71875 → stop 105.71875
    t=15 : sma2=101.5 > sma3=101.33 → long ; ATR : 2.859375, 2.4296875, 3.21484375 → stop 99.78515625
    """
    df = _bars([100] * 8 + [102, 104, 106, 105, 103, 101, 100, 103])
    strat = get_strategy("sma_cross", fast=2, slow=3, atr_window=2, atr_mult=1.0)
    assert strat.warmup_bars() == 8
    out = strat.generate_signals(df)
    assert out["signal"].tolist() == [0] * 8 + [1, 1, 1, 1, -1, -1, -1, 1]
    nan = math.nan
    expected = [nan] * 8 + [99.5, nan, nan, nan, 105.71875, nan, nan, 99.78515625]
    np.testing.assert_array_equal(out["stop_loss"].to_numpy(), expected)


def test_donchian_hand_computed() -> None:
    """entry=3, exit=2, ATR(2), stop à 1 ATR ; high = close+1, low = close-1. Chauffe = 3+5 = 8.

    Clôtures 100 ×9 puis 103, 104, 102, 100, 99, 104, 104.
    t=9  : upper3 = max(high[6..8]) = 101 < 103 → long, ATR = 2+(4-2)/2 = 3 → stop 100
    t=10 : lower2 = min(low[8..9]) = 99 ; 104 ≥ 99 → on garde
    t=11 : lower2 = min(low[9..10]) = 102 ; 102 n'est pas < 102 → on garde
    t=12 : lower2 = min(low[10..11]) = 101 > 100 → sortie ; lower3 = 101 > 100 → short,
           ATR : 2.5, 2.75, 2.875 → stop 102.875
    t=13 : upper2 = max(high[11..12]) = 103 ; 99 → on garde le short
    t=14 : upper2 = max(high[12..13]) = 101 < 104 → sortie ; upper3 = 103 < 104 → long,
           ATR : 2.4375, 4.21875 → stop 99.78125
    t=15 : lower2 = min(low[13..14]) = 98 ; 104 → on garde
    """
    df = _bars([100] * 9 + [103, 104, 102, 100, 99, 104, 104])
    strat = get_strategy(
        "donchian_breakout", entry_window=3, exit_window=2, atr_window=2, atr_mult=1.0
    )
    assert strat.warmup_bars() == 8
    out = strat.generate_signals(df)
    assert out["signal"].tolist() == [0] * 9 + [1, 1, 1, -1, -1, 1, 1]
    nan = math.nan
    expected = [nan] * 9 + [100.0, nan, nan, 102.875, nan, 99.78125, nan]
    np.testing.assert_array_equal(out["stop_loss"].to_numpy(), expected)


def test_bollinger_hand_computed() -> None:
    """window=3, n_std=0.5, RSI(2) < 50, ATR(2), stop à 1 ATR. Chauffe = 3+5 = 8.

    Clôtures 100 ×9 puis 97, 96, 96, 98, 98.
    t=8  : bandes de largeur nulle (lower = 100), 100 n'est pas < 100 → rien
    t=9  : moyenne(100,100,97)=99, écart-type(ddof=0)=√2 → lower = 99-0.5√2 ≈ 98.29 > 97 ;
           RSI(2) = 0 (que des baisses) < 50 → entrée ; ATR = 2+(4-2)/2 = 3 → stop 94
    t=10 : middle = 97.67 > 96 → on garde ; t=11 : middle = 96.33 > 96 → on garde
    t=12 : middle = 96.67 ≤ 98 → sortie ; t=13 : lower ≈ 96.86 < 98 → rien
    """
    df = _bars([100] * 9 + [97, 96, 96, 98, 98])
    strat = get_strategy(
        "bollinger_reversion",
        window=3,
        n_std=0.5,
        rsi_window=2,
        rsi_max=50.0,
        atr_window=2,
        atr_mult=1.0,
    )
    assert strat.warmup_bars() == 8
    out = strat.generate_signals(df)
    assert out["signal"].tolist() == [0] * 9 + [1, 1, 1, 0, 0]
    nan = math.nan
    np.testing.assert_array_equal(
        out["stop_loss"].to_numpy(), [nan] * 9 + [94.0, nan, nan, nan, nan]
    )


def _rsi_hand_frame() -> pd.DataFrame:
    """15 bougies à 90, un saut à 110 puis 109, 108, 107, 106, 105, 104, 104.5, 104, 106.

    RSI(2) de Wilder (gains / pertes lissés avec alpha = 1/2) :
    t=15 : 100 ; t=16 : 90.9 ; t=17 : 76.9 ; t=18 : 58.8 ; t=19 : 40 ; t=20 : 24.39 ;
    t=21 : 13.7 ; t=22 : 40 ; t=23 : 24.85 ; t=24 : 81.35.
    SMA(10) : t=20 : (4×90 + 110+109+108+107+106+105)/10 = 100.5 < 105 ;
              t=23 : (90 + 110+…+104)/10 = 104.75 > 104 (filtre de tendance refusé).
    ATR(2) : 2 jusqu'à t=14, puis 11.5, 6.75, 4.375, 3.1875, 2.59375, 2.296875 (t=20).
    """
    return _bars([90] * 15 + [110, 109, 108, 107, 106, 105, 104, 104.5, 104, 106])


def test_rsi_reversion_hand_computed_max_hold_exit() -> None:
    df = _rsi_hand_frame()
    strat = get_strategy(
        "rsi_reversion",
        rsi_window=2,
        oversold=30.0,
        exit_level=55.0,
        trend_window=10,
        atr_window=2,
        atr_mult=1.0,
        max_hold=2,
    )
    assert strat.warmup_bars() == 15
    out = strat.generate_signals(df)
    # Entrée t=20 (RSI 24.39 < 30 et 105 > 100.5), tenue 2 bougies (t=20, 21), sortie t=22 par
    # max_hold (RSI 40 < 55). t=23 : RSI 24.85 mais 104 < SMA 104.75 → pas d'entrée.
    assert out["signal"].tolist() == [0] * 20 + [1, 1, 0, 0, 0]
    nan = math.nan
    np.testing.assert_array_equal(
        out["stop_loss"].to_numpy(), [nan] * 20 + [105 - 2.296875, nan, nan, nan, nan]
    )


def test_rsi_reversion_hand_computed_rsi_exit() -> None:
    df = _rsi_hand_frame()
    strat = get_strategy(
        "rsi_reversion",
        rsi_window=2,
        oversold=30.0,
        exit_level=55.0,
        trend_window=10,
        atr_window=2,
        atr_mult=1.0,
        max_hold=10,
    )
    out = strat.generate_signals(df)
    # Même entrée t=20 ; RSI 13.7, 40, 24.85 ≤ 55 → on garde ; t=24 : RSI 81.35 > 55 → sortie.
    assert out["signal"].tolist() == [0] * 20 + [1, 1, 1, 1, 0]
    assert out["stop_loss"].iloc[20] == 105 - 2.296875
    assert out["stop_loss"].drop(out.index[20]).isna().all()


def _ema_trend_reference(df: pd.DataFrame, p: dict[str, Any], warmup: int) -> pd.DataFrame:
    """Ré-implémentation indépendante (boucle scalaire) des règles documentées d'ema_trend."""
    fast = ind.ema(df["close"], p["fast"]).tolist()
    slow = ind.ema(df["close"], p["slow"]).tolist()
    adx = ind.adx(df, p["adx_window"])["adx"].tolist()
    atr = ind.atr(df, p["atr_window"]).tolist()
    close = df["close"].astype(float).tolist()
    n = len(df)
    signal = [0] * n
    stop = [math.nan] * n
    pos = 0
    trail = math.nan
    for t in range(n):
        if t < warmup or any(math.isnan(x) for x in (adx[t], atr[t], fast[t], slow[t])):
            pos = 0
            continue
        trend = 0 if fast[t] == slow[t] else (1 if fast[t] > slow[t] else -1)
        if pos == 1 and trend != -1 and close[t] >= trail:
            trail = max(trail, close[t] - p["atr_mult"] * atr[t])
            signal[t], stop[t] = 1, trail
            continue
        if pos == -1 and trend != 1 and close[t] <= trail:
            trail = min(trail, close[t] + p["atr_mult"] * atr[t])
            signal[t], stop[t] = -1, trail
            continue
        stopped_out = (pos == 1 and trend != -1) or (pos == -1 and trend != 1)
        pos = 0
        if stopped_out:
            continue
        if trend != 0 and adx[t] > p["adx_threshold"]:
            pos = trend
            trail = close[t] - pos * p["atr_mult"] * atr[t]
            signal[t], stop[t] = pos, trail
    return pd.DataFrame(
        {"signal": np.array(signal, dtype=np.int64), "stop_loss": stop, "take_profit": math.nan},
        index=df.index,
    )


@pytest.mark.parametrize("dataset", ["ohlcv", "daily", "trending", "dips"])
def test_ema_trend_matches_independent_reference(dataset: str) -> None:
    df = _datasets()[dataset]
    strat = get_strategy("ema_trend")
    out = strat.generate_signals(df)
    ref = _ema_trend_reference(df, strat.params, strat.warmup_bars())
    pd.testing.assert_frame_equal(out, ref, check_exact=True)
    assert (out["signal"] != 0).any()
