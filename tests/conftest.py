"""Fixtures partagées. Aucun test ne doit toucher au réseau."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_ohlcv(
    n: int = 600,
    seed: int = 42,
    start: str = "2024-01-01",
    freq: str = "1h",
    start_price: float = 100.0,
    drift: float = 0.0,
    vol: float = 0.01,
) -> pd.DataFrame:
    """Marche aléatoire géométrique → bougies OHLCV cohérentes (low <= open/close <= high).

    Index : DatetimeIndex UTC, trié, sans doublon. Colonnes : open, high, low, close, volume.
    """
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, size=n)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start_price], close[:-1]])
    wiggle = np.abs(rng.normal(0, vol / 2, size=n)) * close
    high = np.maximum(open_, close) + wiggle
    low = np.minimum(open_, close) - wiggle
    volume = rng.uniform(100, 1000, size=n)
    index = pd.date_range(start=start, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    """600 bougies horaires aléatoires mais reproductibles."""
    return make_ohlcv()


@pytest.fixture
def trending_ohlcv() -> pd.DataFrame:
    """Marché haussier régulier : utile pour vérifier qu'une stratégie de tendance gagne."""
    return make_ohlcv(n=800, seed=7, drift=0.002, vol=0.005)


@pytest.fixture
def daily_ohlcv() -> pd.DataFrame:
    """3 ans de bougies journalières."""
    return make_ohlcv(n=1095, seed=3, freq="1D", vol=0.02)
