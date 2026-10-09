"""Tests de `tradingbot.data.loader` et `tradingbot.data.synthetic`.

Aucun accès réseau : l'exchange ccxt est remplacé par un faux (`FakeExchange`) qui
sert des bougies en mémoire et enregistre les appels reçus. Les valeurs attendues
sont calculées à la main.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from conftest import make_ohlcv

from tradingbot.data import loader
from tradingbot.data.loader import (
    OHLCV_COLUMNS,
    cache_path,
    ensure_ohlcv,
    fetch_ohlcv,
    infer_periods_per_year,
    load_csv,
    load_or_fetch,
    resample_ohlcv,
    save_csv,
    timeframe_to_timedelta,
)
from tradingbot.data.synthetic import default_regimes, generate_ohlcv

TF_MS = 3_600_000  # 1h en millisecondes
START_MS = 1_704_067_200_000  # 2024-01-01T00:00:00Z
T0 = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")


# --- helpers ------------------------------------------------------------------------


def _bars(
    rows: list[tuple[float, float, float, float, float]],
    start: str = "2024-01-01",
    freq: str = "1h",
) -> pd.DataFrame:
    """DataFrame canonique à partir de tuples (open, high, low, close, volume)."""
    idx = pd.date_range(start, periods=len(rows), freq=freq, tz="UTC", name="timestamp")
    return pd.DataFrame(
        {
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "volume": [r[4] for r in rows],
        },
        index=idx,
        dtype="float64",
    )


def _assert_canonical(df: pd.DataFrame) -> None:
    """Vérifie le format canonique décrit dans docs/ARCHITECTURE.md."""
    assert list(df.columns) == list(OHLCV_COLUMNS)
    assert all(df[c].dtype == np.float64 for c in OHLCV_COLUMNS)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz is not None
    assert str(df.index.tz) == "UTC"
    assert df.index.name == "timestamp"
    assert df.index.is_monotonic_increasing
    assert df.index.is_unique
    assert not df[list(OHLCV_COLUMNS)].isna().any().any()


class FakeExchange:
    """Imite la partie de ccxt utilisée par `fetch_ohlcv`.

    Sert `n` bougies horaires consécutives à partir de `start_ms`, paginées par
    `since` / `limit` comme un vrai exchange, et mémorise chaque appel dans `calls`.
    Par défaut, « maintenant » tombe au milieu de la dernière bougie (non clôturée).
    """

    rateLimit = 50
    timeframes = {"1m": "1m", "1h": "1h", "4h": "4h", "1d": "1d"}

    def __init__(
        self,
        n: int = 2500,
        start_ms: int = START_MS,
        tf_ms: int = TF_MS,
        now_ms: int | None = None,
        extra_before_since: int = 0,
    ) -> None:
        self.candles = [
            [start_ms + i * tf_ms, 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 10.0 + i]
            for i in range(n)
        ]
        self.calls: list[tuple[str, str, int | None, int | None]] = []
        self.now_ms = now_ms if now_ms is not None else self.candles[-1][0] + tf_ms // 2
        self.extra_before_since = extra_before_since

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: int | None = None,
        limit: int | None = None,
    ) -> list[list[float]]:
        self.calls.append((symbol, timeframe, since, limit))
        if since is None:
            rows = self.candles[-limit:] if limit else self.candles
        else:
            rows = [c for c in self.candles if c[0] >= since]
            if (
                self.extra_before_since
            ):  # certains exchanges renvoient aussi des bougies antérieures
                before = [c for c in self.candles if c[0] < since][-self.extra_before_since :]
                rows = before + rows
            if limit:
                rows = rows[:limit]
        return [list(r) for r in rows]

    def parse8601(self, text: str) -> int | None:
        # comme ccxt : None pour une date sans heure
        if "T" not in text and " " not in text:
            return None
        ts = pd.Timestamp(text)
        ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
        return int(ts.value // 1_000_000)

    def milliseconds(self) -> int:
        return self.now_ms


def _no_sleep(_seconds: float) -> None:
    return None


# --- ensure_ohlcv -------------------------------------------------------------------


def test_ensure_ohlcv_yahoo_style_columns() -> None:
    """Colonnes « Yahoo » : Date/Open/High/Low/Close/Adj Close/Volume, Adj Close ignorée."""
    raw = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "Open": [10, 11, 12],
            "High": [12, 13, 14],
            "Low": [9, 10, 11],
            "Close": [11, 12, 13],
            "Adj Close": [1, 1, 1],
            "Volume": [100, 200, 300],
        }
    )
    df = ensure_ohlcv(raw)
    _assert_canonical(df)
    assert len(df) == 3
    assert df.index[0] == pd.Timestamp("2024-01-01", tz="UTC")
    assert df.index[-1] == pd.Timestamp("2024-01-03", tz="UTC")
    assert df["open"].tolist() == [10.0, 11.0, 12.0]
    assert df["high"].tolist() == [12.0, 13.0, 14.0]
    assert df["low"].tolist() == [9.0, 10.0, 11.0]
    assert df["close"].tolist() == [11.0, 12.0, 13.0]
    assert df["volume"].tolist() == [100.0, 200.0, 300.0]
    assert "Adj Close" not in df.columns and "adj_close" not in df.columns


def test_ensure_ohlcv_uppercase_and_vol_alias() -> None:
    raw = pd.DataFrame(
        {
            "TIMESTAMP": ["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"],
            "OPEN": [1.0, 2.0],
            "HIGH": [2.0, 3.0],
            "LOW": [0.5, 1.5],
            "CLOSE": [1.5, 2.5],
            "VOL": [7.0, 8.0],
        }
    )
    df = ensure_ohlcv(raw)
    _assert_canonical(df)
    assert df["volume"].tolist() == [7.0, 8.0]
    assert df.index[1] == T0 + pd.Timedelta(hours=1)


@pytest.mark.parametrize("time_col", ["Datetime", "Time", "date", "timestamp", "time"])
def test_ensure_ohlcv_time_column_aliases(time_col: str) -> None:
    raw = pd.DataFrame(
        {
            "open": [1.0, 2.0],
            time_col: ["2024-01-01 00:00:00", "2024-01-01 01:00:00"],
            "high": [2.0, 3.0],
            "low": [0.5, 1.5],
            "close": [1.5, 2.5],
        }
    )
    df = ensure_ohlcv(raw)
    _assert_canonical(df)
    assert list(df.index) == [T0, T0 + pd.Timedelta(hours=1)]
    assert time_col not in df.columns


def test_ensure_ohlcv_epoch_milliseconds_and_seconds() -> None:
    ms = pd.DataFrame(
        {
            "timestamp": [START_MS, START_MS + TF_MS],
            "open": [1, 2],
            "high": [2, 3],
            "low": [0, 1],
            "close": [1, 2],
            "volume": [1, 1],
        }
    )
    seconds = ms.assign(timestamp=ms["timestamp"] // 1000)
    df_ms = ensure_ohlcv(ms)
    df_s = ensure_ohlcv(seconds)
    _assert_canonical(df_ms)
    assert list(df_ms.index) == [T0, T0 + pd.Timedelta(hours=1)]
    assert df_s.index.equals(df_ms.index)


def test_ensure_ohlcv_epoch_as_strings() -> None:
    """Un CSV d'epoch ms lu sans conversion donne des chaînes de chiffres."""
    raw = pd.DataFrame(
        {
            "time": [str(START_MS), str(START_MS + TF_MS)],
            "open": ["1.5", "2.5"],
            "high": ["2", "3"],
            "low": ["1", "2"],
            "close": ["1.75", "2.75"],
        }
    )
    df = ensure_ohlcv(raw)
    _assert_canonical(df)
    assert list(df.index) == [T0, T0 + pd.Timedelta(hours=1)]
    assert df["open"].tolist() == [1.5, 2.5]


def test_ensure_ohlcv_naive_index_is_localised_to_utc() -> None:
    raw = pd.DataFrame(
        {"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [1.0]},
        index=pd.DatetimeIndex(["2024-01-01 12:00:00"]),
    )
    df = ensure_ohlcv(raw)
    _assert_canonical(df)
    assert df.index[0] == pd.Timestamp("2024-01-01 12:00:00", tz="UTC")


def test_ensure_ohlcv_tz_aware_index_is_converted_to_utc() -> None:
    raw = pd.DataFrame(
        {"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [1.0]},
        index=pd.DatetimeIndex(["2024-07-01 14:00:00"], tz="Europe/Paris"),  # UTC+2 en été
    )
    df = ensure_ohlcv(raw)
    assert df.index[0] == pd.Timestamp("2024-07-01 12:00:00", tz="UTC")


def test_ensure_ohlcv_iso_strings_with_offset() -> None:
    raw = pd.DataFrame(
        {
            "date": ["2024-01-01T02:00:00+02:00", "2024-01-01T01:00:00Z"],
            "open": [1.0, 2.0],
            "high": [2.0, 3.0],
            "low": [0.5, 1.5],
            "close": [1.5, 2.5],
        }
    )
    df = ensure_ohlcv(raw)
    assert list(df.index) == [T0, T0 + pd.Timedelta(hours=1)]
    assert df["open"].tolist() == [1.0, 2.0]


def test_ensure_ohlcv_sorts_and_drops_duplicates_keeping_last() -> None:
    raw = pd.DataFrame(
        {
            "timestamp": [
                "2024-01-01 02:00",
                "2024-01-01 00:00",
                "2024-01-01 01:00",
                "2024-01-01 00:00",  # doublon : cette ligne (la dernière) doit gagner
            ],
            "open": [30.0, 10.0, 20.0, 11.0],
            "high": [31.0, 11.0, 21.0, 12.0],
            "low": [29.0, 9.0, 19.0, 10.0],
            "close": [30.5, 10.5, 20.5, 11.5],
            "volume": [3.0, 1.0, 2.0, 1.5],
        }
    )
    df = ensure_ohlcv(raw)
    _assert_canonical(df)
    assert len(df) == 3
    assert list(df.index) == [T0, T0 + pd.Timedelta(hours=1), T0 + pd.Timedelta(hours=2)]
    assert df["open"].tolist() == [11.0, 20.0, 30.0]
    assert df["volume"].tolist() == [1.5, 2.0, 3.0]


def test_ensure_ohlcv_drops_rows_with_nan_prices_and_fills_volume() -> None:
    raw = _bars(
        [
            (1.0, 2.0, 0.5, 1.5, 1.0),
            (np.nan, 2.0, 0.5, 1.5, 1.0),  # open manquant -> ligne supprimée
            (3.0, 4.0, 2.5, 3.5, np.nan),  # volume manquant -> 0.0
        ]
    )
    df = ensure_ohlcv(raw)
    assert len(df) == 2
    assert list(df.index) == [T0, T0 + pd.Timedelta(hours=2)]
    assert df["volume"].tolist() == [1.0, 0.0]


def test_ensure_ohlcv_volume_defaults_to_zero_when_absent() -> None:
    raw = pd.DataFrame(
        {"open": [1.0, 2.0], "high": [2.0, 3.0], "low": [0.5, 1.5], "close": [1.5, 2.5]},
        index=pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC"),
    )
    df = ensure_ohlcv(raw)
    _assert_canonical(df)
    assert df["volume"].tolist() == [0.0, 0.0]


@pytest.mark.parametrize("missing", ["open", "high", "low", "close"])
def test_ensure_ohlcv_missing_required_column_raises(missing: str) -> None:
    raw = make_ohlcv(n=5).drop(columns=[missing])
    with pytest.raises(ValueError, match="manquantes") as excinfo:
        ensure_ohlcv(raw)
    assert missing in str(excinfo.value)


def test_ensure_ohlcv_empty_raises() -> None:
    empty = make_ohlcv(n=5).iloc[0:0]
    with pytest.raises(ValueError, match="vide"):
        ensure_ohlcv(empty)


def test_ensure_ohlcv_all_rows_invalid_raises() -> None:
    raw = make_ohlcv(n=3)
    raw = raw.assign(close=np.nan)
    with pytest.raises(ValueError, match="vide"):
        ensure_ohlcv(raw)


def test_ensure_ohlcv_no_time_axis_raises() -> None:
    raw = pd.DataFrame({"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5]})
    with pytest.raises(ValueError, match="dates"):
        ensure_ohlcv(raw)


def test_ensure_ohlcv_ambiguous_columns_raise() -> None:
    raw = pd.DataFrame(
        {
            "timestamp": ["2024-01-01"],
            "open": [1.0],
            "high": [2.0],
            "low": [0.5],
            "close": [1.5],
            "Close": [9.0],
        }
    )
    with pytest.raises(ValueError, match="ambigu"):
        ensure_ohlcv(raw)


def test_ensure_ohlcv_not_a_dataframe_raises() -> None:
    with pytest.raises(TypeError):
        ensure_ohlcv([1, 2, 3])  # type: ignore[arg-type]


def test_ensure_ohlcv_unnamed_first_column_with_dates() -> None:
    """CSV sauvé avec un index sans nom : pandas nomme la colonne 'Unnamed: 0'."""
    raw = pd.DataFrame(
        {
            "Unnamed: 0": ["2024-01-01T00:00:00+00:00", "2024-01-01T01:00:00+00:00"],
            "open": [1.0, 2.0],
            "high": [2.0, 3.0],
            "low": [0.5, 1.5],
            "close": [1.5, 2.5],
            "volume": [1.0, 1.0],
        }
    )
    df = ensure_ohlcv(raw)
    _assert_canonical(df)
    assert list(df.index) == [T0, T0 + pd.Timedelta(hours=1)]


def test_ensure_ohlcv_string_index_from_csv_index_col() -> None:
    """Index de chaînes (lecture CSV avec index_col=0 sans parse_dates)."""
    raw = pd.DataFrame(
        {"open": [1.0, 2.0], "high": [2.0, 3.0], "low": [0.5, 1.5], "close": [1.5, 2.5]},
        index=pd.Index(["2024-01-01 00:00", "2024-01-01 01:00"]),
    )
    df = ensure_ohlcv(raw)
    assert list(df.index) == [T0, T0 + pd.Timedelta(hours=1)]


def test_ensure_ohlcv_extra_columns_are_dropped_and_order_canonical() -> None:
    raw = make_ohlcv(n=4)
    raw = raw.assign(rsi=1.0)
    raw = raw.loc[:, ["volume", "rsi", "close", "low", "high", "open"]]
    df = ensure_ohlcv(raw)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


def test_ensure_ohlcv_is_identity_on_canonical_frame(ohlcv: pd.DataFrame) -> None:
    out = ensure_ohlcv(ohlcv)
    pd.testing.assert_frame_equal(out, ohlcv, check_names=False)
    assert out.index.name == "timestamp"
    assert out is not ohlcv


def test_ensure_ohlcv_does_not_modify_input() -> None:
    raw = pd.DataFrame(
        {
            "Date": ["2024-01-02", "2024-01-01"],
            "Open": [2, 1],
            "High": [3, 2],
            "Low": [1, 0],
            "Close": [2, 1],
        }
    )
    snapshot = raw.copy()
    ensure_ohlcv(raw)
    pd.testing.assert_frame_equal(raw, snapshot)


def test_ensure_ohlcv_infers_regular_freq() -> None:
    raw = pd.DataFrame(
        {
            "timestamp": [START_MS + i * TF_MS for i in range(5)],
            "open": [1.0] * 5,
            "high": [2.0] * 5,
            "low": [0.5] * 5,
            "close": [1.5] * 5,
        }
    )
    assert ensure_ohlcv(raw).index.freq == pd.tseries.frequencies.to_offset("h")
    gappy = raw.drop(index=2)
    assert ensure_ohlcv(gappy).index.freq is None


# --- CSV ----------------------------------------------------------------------------


def test_csv_round_trip_exact(tmp_path: Path, ohlcv: pd.DataFrame) -> None:
    path = tmp_path / "sub" / "btc.csv"
    save_csv(ohlcv, path)
    assert path.is_file()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "timestamp,open,high,low,close,volume"
    assert lines[1].startswith("2024-01-01T00:00:00+00:00,")
    assert lines[2].startswith("2024-01-01T01:00:00+00:00,")
    assert len(lines) == len(ohlcv) + 1

    back = load_csv(path)
    _assert_canonical(back)
    pd.testing.assert_frame_equal(back, ohlcv, check_names=False)
    assert back.index.name == "timestamp"


def test_csv_round_trip_synthetic_daily(tmp_path: Path) -> None:
    df = generate_ohlcv(n=40, seed=5, freq="1d", start="2023-03-01")
    path = tmp_path / "daily.csv"
    save_csv(df, path)
    back = load_csv(path)
    pd.testing.assert_frame_equal(back, df)


def test_save_csv_sub_second_timestamps_round_trip(tmp_path: Path) -> None:
    idx = pd.DatetimeIndex(
        ["2024-01-01 00:00:00.250", "2024-01-01 00:00:01.500", "2024-01-01 00:00:02.750"],
        tz="UTC",
        name="timestamp",
    )
    df = pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0],
            "high": [2.0, 3.0, 4.0],
            "low": [0.5, 1.5, 2.5],
            "close": [1.5, 2.5, 3.5],
            "volume": [1.0, 1.0, 1.0],
        },
        index=idx,
    )
    path = tmp_path / "ms.csv"
    save_csv(df, path)
    assert ".250000+00:00" in path.read_text(encoding="utf-8").splitlines()[1]
    back = load_csv(path)
    assert list(back.index) == list(df.index)


def test_load_csv_semicolon_epoch_and_yahoo_columns(tmp_path: Path) -> None:
    path = tmp_path / "export.csv"
    path.write_text(
        "Date;Open;High;Low;Close;Adj Close;Volume\n"
        f"{START_MS + TF_MS};2;3;1;2.5;0;20\n"
        f"{START_MS};1;2;0.5;1.5;0;10\n",
        encoding="utf-8",
    )
    df = load_csv(path)
    _assert_canonical(df)
    assert list(df.index) == [T0, T0 + pd.Timedelta(hours=1)]
    assert df["open"].tolist() == [1.0, 2.0]
    assert df["volume"].tolist() == [10.0, 20.0]


def test_load_csv_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="introuvable"):
        load_csv(tmp_path / "nope.csv")


def test_load_csv_header_only_raises(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("timestamp,open,high,low,close,volume\n", encoding="utf-8")
    with pytest.raises(ValueError, match="aucune bougie"):
        load_csv(path)


def test_save_csv_accepts_messy_frame(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        {
            "Date": ["2024-01-02", "2024-01-01"],
            "Open": [2, 1],
            "High": [3, 2],
            "Low": [1, 0],
            "Close": [2, 1],
        }
    )
    path = tmp_path / "messy.csv"
    save_csv(raw, path)
    back = load_csv(path)
    assert back.index[0] == pd.Timestamp("2024-01-01", tz="UTC")
    assert back["volume"].tolist() == [0.0, 0.0]


# --- timeframe_to_timedelta / infer_periods_per_year --------------------------------


@pytest.mark.parametrize(
    ("tf", "seconds"),
    [
        ("1m", 60),
        ("3m", 180),
        ("5m", 300),
        ("15m", 900),
        ("30m", 1800),
        ("1h", 3600),
        ("2h", 7200),
        ("4h", 14400),
        ("6h", 21600),
        ("12h", 43200),
        ("1d", 86400),
        ("3d", 259200),
        ("1w", 604800),
    ],
)
def test_timeframe_to_timedelta_values(tf: str, seconds: int) -> None:
    delta = timeframe_to_timedelta(tf)
    assert isinstance(delta, pd.Timedelta)
    assert delta == pd.Timedelta(seconds=seconds)


@pytest.mark.parametrize("tf", ["1M", "h", "0m", "", "1x", "1.5h", "1 mois", "m1", "-1h"])
def test_timeframe_to_timedelta_invalid_raises(tf: str) -> None:
    with pytest.raises(ValueError, match="Timeframe invalide"):
        timeframe_to_timedelta(tf)


def test_timeframe_to_timedelta_non_string_raises() -> None:
    with pytest.raises(TypeError):
        timeframe_to_timedelta(60)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("freq", "expected"),
    [("1min", 525_600.0), ("1h", 8_760.0), ("4h", 2_190.0), ("1D", 365.0)],
)
def test_infer_periods_per_year(freq: str, expected: float) -> None:
    idx = pd.date_range("2024-01-01", periods=50, freq=freq, tz="UTC")
    assert infer_periods_per_year(idx) == pytest.approx(expected)


def test_infer_periods_per_year_short_index_defaults_to_365() -> None:
    assert infer_periods_per_year(pd.DatetimeIndex(["2024-01-01"], tz="UTC")) == 365.0
    assert infer_periods_per_year(pd.DatetimeIndex([], tz="UTC")) == 365.0


def test_infer_periods_per_year_is_robust_to_gaps() -> None:
    idx = pd.date_range("2024-01-01", periods=50, freq="1h", tz="UTC")
    gappy = idx.delete([5, 6, 7, 20])  # quelques trous : la médiane reste 1h
    assert infer_periods_per_year(gappy) == pytest.approx(8_760.0)


def test_infer_periods_per_year_accepts_dataframe(daily_ohlcv: pd.DataFrame) -> None:
    assert infer_periods_per_year(daily_ohlcv) == pytest.approx(365.0)


# --- resample_ohlcv -----------------------------------------------------------------


def test_resample_1h_to_4h_hand_checked() -> None:
    hourly = _bars(
        [
            (10.0, 12.0, 9.0, 11.0, 1.0),
            (11.0, 15.0, 10.0, 14.0, 2.0),
            (14.0, 14.5, 8.0, 9.0, 3.0),
            (9.0, 10.0, 7.0, 9.5, 4.0),
            (9.5, 11.0, 9.0, 10.0, 5.0),
            (10.0, 13.0, 9.8, 12.0, 6.0),
            (12.0, 12.2, 11.0, 11.5, 7.0),
            (11.5, 16.0, 11.4, 15.0, 8.0),
        ]
    )
    out = resample_ohlcv(hourly, "4h")
    _assert_canonical(out)
    assert list(out.index) == [T0, T0 + pd.Timedelta(hours=4)]
    assert out["open"].tolist() == [10.0, 9.5]
    assert out["high"].tolist() == [15.0, 16.0]
    assert out["low"].tolist() == [7.0, 9.0]
    assert out["close"].tolist() == [9.5, 15.0]
    assert out["volume"].tolist() == [10.0, 26.0]
    assert infer_periods_per_year(out) == pytest.approx(2_190.0)


def test_resample_bins_are_anchored_to_midnight_utc() -> None:
    """Des bougies qui commencent à 02:00 tombent dans l'intervalle [00:00, 04:00)."""
    hourly = _bars(
        [
            (1.0, 2.0, 0.5, 1.5, 1.0),
            (1.5, 3.0, 1.0, 2.5, 1.0),
            (2.5, 4.0, 2.0, 3.5, 1.0),
            (3.5, 5.0, 3.0, 4.5, 1.0),
        ],
        start="2024-01-01 02:00",
    )
    out = resample_ohlcv(hourly, "4h")
    assert list(out.index) == [T0, T0 + pd.Timedelta(hours=4)]
    assert out["open"].tolist() == [1.0, 2.5]
    assert out["close"].tolist() == [2.5, 4.5]
    assert out["volume"].tolist() == [2.0, 2.0]


def test_resample_drops_empty_bins() -> None:
    hourly = _bars([(1.0, 2.0, 0.5, 1.5, 1.0)] * 12)
    hourly = hourly.drop(index=hourly.index[4:8])  # plus rien entre 04:00 et 08:00
    out = resample_ohlcv(hourly, "4h")
    assert list(out.index) == [T0, T0 + pd.Timedelta(hours=8)]
    assert out["volume"].tolist() == [4.0, 4.0]


def test_resample_daily_to_weekly_starts_monday() -> None:
    daily = _bars([(1.0, 2.0, 0.5, 1.5, 1.0)] * 10, start="2024-01-03", freq="1D")  # un mercredi
    out = resample_ohlcv(daily, "1w")
    assert list(out.index) == [
        pd.Timestamp("2024-01-01", tz="UTC"),  # lundi
        pd.Timestamp("2024-01-08", tz="UTC"),
    ]
    assert out["volume"].tolist() == [5.0, 5.0]  # mer..dim puis lun..ven


def test_resample_1h_to_1d_volume_and_extremes(ohlcv: pd.DataFrame) -> None:
    out = resample_ohlcv(ohlcv, "1d")
    first_day = ohlcv.iloc[:24]
    assert out.index[0] == T0
    assert out["open"].iloc[0] == first_day["open"].iloc[0]
    assert out["close"].iloc[0] == first_day["close"].iloc[-1]
    assert out["high"].iloc[0] == first_day["high"].max()
    assert out["low"].iloc[0] == first_day["low"].min()
    assert out["volume"].iloc[0] == pytest.approx(first_day["volume"].sum())
    assert len(out) == 25  # 600 h = 25 jours


def test_resample_accepts_pandas_alias() -> None:
    """Le timeframe ccxt ("1d") et l'alias pandas ("D") donnent le même résultat."""
    hourly = _bars(
        [(float(i), float(i) + 1.0, float(i) - 0.5, float(i) + 0.5, 1.0) for i in range(48)]
    )
    out = resample_ohlcv(hourly, "1d")
    out_alias = resample_ohlcv(hourly, "D")
    pd.testing.assert_frame_equal(out, out_alias)
    assert list(out.index) == [T0, T0 + pd.Timedelta(days=1)]
    assert out["open"].tolist() == [0.0, 24.0]
    assert out["high"].tolist() == [24.0, 48.0]
    assert out["low"].tolist() == [-0.5, 23.5]
    assert out["close"].tolist() == [23.5, 47.5]
    assert out["volume"].tolist() == [24.0, 24.0]


# --- fetch_ohlcv --------------------------------------------------------------------


def test_fetch_ohlcv_paginates_2500_candles_in_three_calls() -> None:
    fx = FakeExchange(n=2500)
    sleeps: list[float] = []
    df = fetch_ohlcv("fake", "BTC/USDT", "1h", since="2024-01-01", exchange=fx, sleep=sleeps.append)

    _assert_canonical(df)
    assert len(fx.calls) == 3
    assert [c[2] for c in fx.calls] == [
        START_MS,
        START_MS + 999 * TF_MS + 1,
        START_MS + 1999 * TF_MS + 1,
    ]
    assert all(c[3] == 1000 for c in fx.calls)
    assert all(c[0] == "BTC/USDT" and c[1] == "1h" for c in fx.calls)
    # 2500 bougies servies, la dernière (ouverte) est retirée
    assert len(df) == 2499
    assert df.index[0] == T0
    assert df.index[-1] == T0 + pd.Timedelta(hours=2498)
    assert df["close"].iloc[-1] == 100.5 + 2498
    assert df["open"].iloc[0] == 100.0
    # une pause rateLimit/1000 entre chaque appel, pas après le dernier
    assert sleeps == [0.05, 0.05]


def test_fetch_ohlcv_keeps_last_candle_when_closed_or_when_asked() -> None:
    closed = FakeExchange(n=2500, now_ms=START_MS + 2500 * TF_MS)  # la dernière vient de clôturer
    df = fetch_ohlcv("fake", "BTC/USDT", "1h", since=START_MS, exchange=closed, sleep=_no_sleep)
    assert len(df) == 2500

    open_bar = FakeExchange(n=2500)
    df2 = fetch_ohlcv(
        "fake",
        "BTC/USDT",
        "1h",
        since=START_MS,
        exchange=open_bar,
        sleep=_no_sleep,
        drop_last_incomplete=False,
    )
    assert len(df2) == 2500

    # now_ms injecté prime sur exchange.milliseconds()
    df3 = fetch_ohlcv(
        "fake",
        "BTC/USDT",
        "1h",
        since=START_MS,
        exchange=FakeExchange(n=2500),
        sleep=_no_sleep,
        now_ms=START_MS + 2500 * TF_MS,
    )
    assert len(df3) == 2500


def test_fetch_ohlcv_respects_limit_cap() -> None:
    fx = FakeExchange(n=2500)
    df = fetch_ohlcv(
        "fake", "BTC/USDT", "1h", since=START_MS, limit=1500, exchange=fx, sleep=_no_sleep
    )
    assert len(df) == 1500
    assert [(c[2], c[3]) for c in fx.calls] == [(START_MS, 1000), (START_MS + 999 * TF_MS + 1, 500)]
    assert df.index[-1] == T0 + pd.Timedelta(hours=1499)


def test_fetch_ohlcv_without_since_returns_most_recent_closed_candles() -> None:
    fx = FakeExchange(n=2500)
    df = fetch_ohlcv("fake", "BTC/USDT", "1h", limit=500, exchange=fx, sleep=_no_sleep)
    assert fx.calls == [("BTC/USDT", "1h", None, 500)]
    assert len(df) == 499  # 500 dernières moins la bougie en cours
    assert df.index[-1] == T0 + pd.Timedelta(hours=2498)


def test_fetch_ohlcv_until_is_inclusive() -> None:
    fx = FakeExchange(n=2500)
    until = START_MS + 10 * TF_MS
    df = fetch_ohlcv(
        "fake", "BTC/USDT", "1h", since=START_MS, until=until, exchange=fx, sleep=_no_sleep
    )
    assert len(df) == 11
    assert df.index[-1] == T0 + pd.Timedelta(hours=10)
    assert len(fx.calls) == 1


def test_fetch_ohlcv_until_stops_pagination_early() -> None:
    fx = FakeExchange(n=2500)
    until = START_MS + 1200 * TF_MS
    df = fetch_ohlcv(
        "fake", "BTC/USDT", "1h", since=START_MS, until=until, exchange=fx, sleep=_no_sleep
    )
    assert len(df) == 1201
    assert len(fx.calls) == 2


@pytest.mark.parametrize(
    "since",
    [
        START_MS,
        "2024-01-01",
        "2024-01-01T00:00:00Z",
        "2024-01-01 00:00:00",
        datetime(2024, 1, 1),
        datetime(2024, 1, 1, tzinfo=UTC),
        pd.Timestamp("2024-01-01", tz="UTC"),
        pd.Timestamp("2024-01-01 01:00", tz="Europe/Paris"),
        np.int64(START_MS),
        float(START_MS),
    ],
)
def test_fetch_ohlcv_since_accepts_many_types(since: Any) -> None:
    fx = FakeExchange(n=20, now_ms=START_MS + 100 * TF_MS)
    df = fetch_ohlcv("fake", "BTC/USDT", "1h", since=since, exchange=fx, sleep=_no_sleep)
    assert fx.calls[0][2] == START_MS
    assert len(df) == 20


def test_fetch_ohlcv_filters_candles_older_than_since_and_dedups() -> None:
    fx = FakeExchange(n=2500, extra_before_since=3)
    df = fetch_ohlcv(
        "fake", "BTC/USDT", "1h", since=START_MS + 5 * TF_MS, exchange=fx, sleep=_no_sleep
    )
    assert df.index[0] == T0 + pd.Timedelta(hours=5)
    assert df.index.is_unique
    assert len(df) == 2500 - 5 - 1


def test_fetch_ohlcv_stops_when_exchange_does_not_progress() -> None:
    class StuckExchange(FakeExchange):
        def fetch_ohlcv(
            self, symbol: str, timeframe: str, since: int | None = None, limit: int | None = None
        ):
            self.calls.append((symbol, timeframe, since, limit))
            return [list(r) for r in self.candles[:limit]]  # toujours la même page

    fx = StuckExchange(n=1000, now_ms=START_MS + 5000 * TF_MS)
    df = fetch_ohlcv("fake", "BTC/USDT", "1h", since=START_MS, exchange=fx, sleep=_no_sleep)
    assert len(df) == 1000
    assert len(fx.calls) == 2


def test_fetch_ohlcv_empty_response_gives_empty_canonical_frame() -> None:
    fx = FakeExchange(n=0, now_ms=START_MS)
    df = fetch_ohlcv("fake", "BTC/USDT", "1h", since=START_MS, exchange=fx, sleep=_no_sleep)
    assert len(df) == 0
    assert list(df.columns) == list(OHLCV_COLUMNS)
    assert isinstance(df.index, pd.DatetimeIndex) and str(df.index.tz) == "UTC"
    assert df.index.name == "timestamp"


def test_fetch_ohlcv_rejects_unsupported_timeframe_and_bad_args() -> None:
    fx = FakeExchange(n=10)
    with pytest.raises(ValueError, match="non supporté"):
        fetch_ohlcv("fake", "BTC/USDT", "2h", exchange=fx, sleep=_no_sleep)
    with pytest.raises(ValueError, match="Timeframe invalide"):
        fetch_ohlcv("fake", "BTC/USDT", "1M", exchange=fx, sleep=_no_sleep)
    with pytest.raises(ValueError, match="limit"):
        fetch_ohlcv("fake", "BTC/USDT", "1h", limit=0, exchange=fx, sleep=_no_sleep)
    with pytest.raises(ValueError, match="postérieur"):
        fetch_ohlcv("fake", "BTC/USDT", "1h", since=START_MS + 10, until=START_MS, exchange=fx)
    with pytest.raises(ValueError, match="Date invalide"):
        fetch_ohlcv("fake", "BTC/USDT", "1h", since="pas une date", exchange=fx)
    assert fx.calls == []


def test_fetch_ohlcv_unknown_exchange_id_raises_without_network() -> None:
    with pytest.raises(ValueError, match="inconnu"):
        fetch_ohlcv("exchange_qui_n_existe_pas", "BTC/USDT", "1h")


def test_fetch_ohlcv_rows_with_more_than_six_fields() -> None:
    class VerboseExchange(FakeExchange):
        def fetch_ohlcv(
            self, symbol: str, timeframe: str, since: int | None = None, limit: int | None = None
        ):
            rows = super().fetch_ohlcv(symbol, timeframe, since, limit)
            return [r + [1, 2, 3] for r in rows]  # champs supplémentaires (certains exchanges)

    fx = VerboseExchange(n=5, now_ms=START_MS + 100 * TF_MS)
    df = fetch_ohlcv("fake", "BTC/USDT", "1h", since=START_MS, exchange=fx, sleep=_no_sleep)
    assert list(df.columns) == list(OHLCV_COLUMNS)
    assert df["volume"].tolist() == [10.0, 11.0, 12.0, 13.0, 14.0]


# --- cache --------------------------------------------------------------------------


def test_cache_path_sanitises_symbol(tmp_path: Path) -> None:
    assert cache_path("data/cache", "binance", "BTC/USDT", "1h") == Path(
        "data/cache/binance_BTC-USDT_1h.csv"
    )
    assert (
        cache_path(tmp_path, "bybit", "BTC/USDT:USDT", "4h")
        == tmp_path / "bybit_BTC-USDT-USDT_4h.csv"
    )
    assert cache_path(tmp_path, "kraken", "ETH/EUR", "1d").name == "kraken_ETH-EUR_1d.csv"


def test_load_or_fetch_uses_cache_then_refreshes(tmp_path: Path) -> None:
    first = FakeExchange(n=50, now_ms=START_MS + 1000 * TF_MS)
    df1 = load_or_fetch(
        "fake",
        "BTC/USDT",
        "1h",
        since=START_MS,
        cache_dir=tmp_path,
        exchange=first,
        sleep=_no_sleep,
    )
    assert len(first.calls) == 1
    assert len(df1) == 50
    path = cache_path(tmp_path, "fake", "BTC/USDT", "1h")
    assert path.is_file()

    second = FakeExchange(n=10, now_ms=START_MS + 1000 * TF_MS)
    df2 = load_or_fetch(
        "fake",
        "BTC/USDT",
        "1h",
        since=START_MS,
        cache_dir=tmp_path,
        exchange=second,
        sleep=_no_sleep,
    )
    assert second.calls == []  # servi depuis le cache
    pd.testing.assert_frame_equal(df2, df1)

    third = FakeExchange(n=10, now_ms=START_MS + 1000 * TF_MS)
    df3 = load_or_fetch(
        "fake",
        "BTC/USDT",
        "1h",
        since=START_MS,
        cache_dir=tmp_path,
        refresh=True,
        exchange=third,
        sleep=_no_sleep,
    )
    assert len(third.calls) == 1
    assert len(df3) == 10
    pd.testing.assert_frame_equal(load_csv(path), df3)


def test_load_or_fetch_does_not_cache_empty_result(tmp_path: Path) -> None:
    fx = FakeExchange(n=0, now_ms=START_MS)
    df = load_or_fetch("fake", "BTC/USDT", "1h", cache_dir=tmp_path, exchange=fx, sleep=_no_sleep)
    assert len(df) == 0
    assert not cache_path(tmp_path, "fake", "BTC/USDT", "1h").exists()


# --- synthetic ----------------------------------------------------------------------


def test_generate_ohlcv_shape_and_coherence() -> None:
    df = generate_ohlcv(n=500, seed=0, start_price=250.0)
    _assert_canonical(df)
    assert len(df) == 500
    assert df.index[0] == pd.Timestamp("2022-01-01", tz="UTC")
    assert df.index[1] - df.index[0] == pd.Timedelta(hours=1)
    assert df["open"].iloc[0] == 250.0  # prix de départ exact
    assert (df["low"] <= np.minimum(df["open"], df["close"])).all()
    assert (df["high"] >= np.maximum(df["open"], df["close"])).all()
    assert (df["volume"] > 0).all()
    assert (df[["open", "high", "low", "close"]] > 0).all().all()
    # pas de gap : l'ouverture est la clôture précédente
    np.testing.assert_array_equal(df["open"].to_numpy()[1:], df["close"].to_numpy()[:-1])


def test_generate_ohlcv_seed_reproducible() -> None:
    a = generate_ohlcv(n=300, seed=123)
    b = generate_ohlcv(n=300, seed=123)
    c = generate_ohlcv(n=300, seed=124)
    pd.testing.assert_frame_equal(a, b)
    assert not a["close"].equals(c["close"])


def test_generate_ohlcv_matches_conftest_contract(ohlcv: pd.DataFrame) -> None:
    df = generate_ohlcv(n=len(ohlcv), seed=42, start="2024-01-01")
    assert df.index.equals(ohlcv.index)
    assert list(df.columns) == list(ohlcv.columns)
    assert df.index.dtype == ohlcv.index.dtype
    pd.testing.assert_frame_equal(ensure_ohlcv(df), df)


@pytest.mark.parametrize("n", [1, 2, 3, 10, 2000, 2001, 2002])
def test_default_regimes_lengths_sum_to_n(n: int) -> None:
    regimes = default_regimes(n, 0.01)
    assert len(regimes) == 3
    assert sum(r[0] for r in regimes) == n
    assert [r[0] for r in regimes] == [n // 3, n // 3, n - 2 * (n // 3)]
    assert [r[1] for r in regimes] == [0.0004, -0.0003, 0.0]
    assert [r[2] for r in regimes] == pytest.approx([0.01, 0.015, 0.007])
    assert len(generate_ohlcv(n=n, seed=1)) == n


def test_generate_ohlcv_custom_regimes_without_noise_are_exact() -> None:
    """vol = 0 : le prix suit exactement exp(drift) et les mèches sont nulles."""
    df = generate_ohlcv(n=6, seed=0, start_price=100.0, regimes=[(4, 0.01, 0.0), (2, -0.02, 0.0)])
    expected_close = [100.0 * np.exp(0.01 * k) for k in range(1, 5)]
    expected_close += [expected_close[-1] * np.exp(-0.02), expected_close[-1] * np.exp(-0.04)]
    np.testing.assert_allclose(df["close"].to_numpy(), expected_close, rtol=1e-12)
    np.testing.assert_allclose(
        df["high"].to_numpy(), np.maximum(df["open"], df["close"]), rtol=1e-12
    )
    np.testing.assert_allclose(
        df["low"].to_numpy(), np.minimum(df["open"], df["close"]), rtol=1e-12
    )
    assert (df["volume"] > 0).all()


def test_generate_ohlcv_regimes_drive_the_trend() -> None:
    df = generate_ohlcv(n=3000, seed=0, regimes=[(1500, 0.002, 0.005), (1500, -0.002, 0.005)])
    log_ret = np.log(df["close"]).diff().dropna()
    assert log_ret.iloc[:1499].mean() == pytest.approx(0.002, abs=0.0005)
    assert log_ret.iloc[1500:].mean() == pytest.approx(-0.002, abs=0.0005)
    assert df["close"].iloc[1499] > df["close"].iloc[0]
    assert df["close"].iloc[-1] < df["close"].iloc[1499]


def test_generate_ohlcv_regimes_sum_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="somme"):
        generate_ohlcv(n=100, regimes=[(50, 0.0, 0.01), (40, 0.0, 0.01)])
    with pytest.raises(ValueError, match="Régime"):
        generate_ohlcv(n=100, regimes=[(100, 0.0, -0.01)])


@pytest.mark.parametrize(
    "kwargs",
    [{"n": 0}, {"n": -5}, {"start_price": 0.0}, {"start_price": -1.0}, {"vol": -0.1}],
)
def test_generate_ohlcv_invalid_arguments(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        generate_ohlcv(**kwargs)


def test_generate_ohlcv_freq_and_start_variants() -> None:
    daily = generate_ohlcv(n=10, seed=0, freq="1d", start="2020-06-15")
    assert daily.index[0] == pd.Timestamp("2020-06-15", tz="UTC")
    assert daily.index[-1] == pd.Timestamp("2020-06-24", tz="UTC")
    assert infer_periods_per_year(daily) == pytest.approx(365.0)

    minutes = generate_ohlcv(n=5, seed=0, freq="15m")
    assert minutes.index[1] - minutes.index[0] == pd.Timedelta(minutes=15)

    paris = generate_ohlcv(n=2, seed=0, start=pd.Timestamp("2024-01-01 01:00", tz="Europe/Paris"))
    assert paris.index[0] == T0

    alias = generate_ohlcv(n=3, seed=0, freq="D")
    assert alias.index[1] - alias.index[0] == pd.Timedelta(days=1)


def test_generate_ohlcv_single_bar() -> None:
    df = generate_ohlcv(n=1, seed=0, start_price=42.0)
    assert len(df) == 1
    assert df["open"].iloc[0] == 42.0
    assert df["low"].iloc[0] <= min(df["open"].iloc[0], df["close"].iloc[0])


def test_generate_ohlcv_no_look_ahead_prefix_property() -> None:
    """Les 100 premières bougies ne dépendent pas de n (le générateur avance dans l'ordre)
    tant que les régimes sont identiques."""
    regimes = [(100, 0.001, 0.01), (100, -0.001, 0.01)]
    short = generate_ohlcv(n=200, seed=9, regimes=regimes)
    longer = generate_ohlcv(
        n=300, seed=9, regimes=[(100, 0.001, 0.01), (100, -0.001, 0.01), (100, 0.0, 0.01)]
    )
    np.testing.assert_allclose(short["close"].to_numpy(), longer["close"].to_numpy()[:200])


# --- module hygiene -----------------------------------------------------------------


def test_loader_has_no_print_and_uses_module_logger() -> None:
    source = Path(loader.__file__).read_text(encoding="utf-8")
    assert "print(" not in source
    assert "logging.getLogger(__name__)" in source
    assert "shift(-1)" not in source and "center=True" not in source
