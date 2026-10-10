"""Tests du gestionnaire de risque : formules de taille, plafonds, kill switch, limite
journalière, état sérialisable."""

from __future__ import annotations

import json
import logging
import math
from datetime import UTC, datetime, timedelta, timezone

import pandas as pd
import pytest

from tradingbot.risk import RiskConfig, RiskManager, _to_utc


def manager(**overrides: object) -> RiskManager:
    kwargs: dict[str, object] = {
        "risk_per_trade": 0.01,
        "max_position_fraction": 0.95,
        "max_drawdown": 0.20,
        "daily_loss_limit": 0.05,
        "allow_short": False,
        "min_trade_notional": 10.0,
    }
    kwargs.update(overrides)
    return RiskManager(RiskConfig(**kwargs))  # type: ignore[arg-type]


T0 = pd.Timestamp("2024-03-10 00:00", tz="UTC")


def at(hours: float) -> pd.Timestamp:
    return T0 + pd.Timedelta(hours=hours)


# ============================================================================ sizing
def test_position_size_with_stop_is_risk_divided_by_distance() -> None:
    rm = manager()
    # 10 000 × 1 % = 100 de risque ; distance 5 → 20 unités (notionnel 2 000 < 9 500).
    assert rm.position_size(10_000, 100, stop_price=95) == pytest.approx(20.0)
    # Stop au-dessus (short) : même distance, même taille.
    assert rm.position_size(10_000, 100, stop_price=105) == pytest.approx(20.0)
    # Distance 2 → 50 unités.
    assert rm.position_size(10_000, 100, stop_price=98) == pytest.approx(50.0)


def test_position_size_without_stop_is_max_fraction() -> None:
    rm = manager()
    assert rm.position_size(10_000, 100) == pytest.approx(95.0)
    assert rm.position_size(10_000, 200) == pytest.approx(47.5)
    # Stop égal au prix : distance nulle → comportement « sans stop ».
    assert rm.position_size(10_000, 100, stop_price=100) == pytest.approx(95.0)


def test_position_size_capped_by_max_position_fraction() -> None:
    rm = manager()
    # Stop très proche : 100 / 0,1 = 1 000 unités → plafonné à 9 500 / 100 = 95.
    assert rm.position_size(10_000, 100, stop_price=99.9) == pytest.approx(95.0)
    rm2 = manager(max_position_fraction=0.5)
    assert rm2.position_size(10_000, 100, stop_price=99.9) == pytest.approx(50.0)
    assert rm2.position_size(10_000, 100) == pytest.approx(50.0)


def test_position_size_min_notional() -> None:
    rm = manager()
    # 100 × 1 % = 1 de risque ; distance 5 → 0,2 unité, notionnel 20 ≥ 10 : OK.
    assert rm.position_size(100, 100, stop_price=95) == pytest.approx(0.2)
    # Même calcul avec un minimum de 50 : notionnel 20 < 50 → 0.
    assert manager(min_trade_notional=50.0).position_size(100, 100, stop_price=95) == 0.0
    # Sans stop : 5 × 0,95 = 4,75 < 10 → 0.
    assert rm.position_size(5, 100) == 0.0


def test_position_size_vol_targeting() -> None:
    rm = manager(vol_target=0.20)
    base = rm.position_size(10_000, 100, stop_price=95)
    assert base == pytest.approx(20.0)
    # Vol réalisée 0,40 → facteur 0,5.
    assert rm.position_size(10_000, 100, stop_price=95, realized_vol=0.40) == pytest.approx(10.0)
    # Vol réalisée 0,10 → facteur min(1, 2) = 1 : jamais de levier.
    assert rm.position_size(10_000, 100, stop_price=95, realized_vol=0.10) == pytest.approx(20.0)
    # Vol inconnue ou nulle → pas de pondération.
    assert rm.position_size(10_000, 100, stop_price=95, realized_vol=None) == pytest.approx(20.0)
    assert rm.position_size(10_000, 100, stop_price=95, realized_vol=0.0) == pytest.approx(20.0)
    assert rm.position_size(10_000, 100, stop_price=95, realized_vol=math.nan) == pytest.approx(
        20.0
    )
    # Sans vol_target configurée, realized_vol est ignorée.
    assert manager().position_size(10_000, 100, stop_price=95, realized_vol=5.0) == pytest.approx(
        20.0
    )
    # Pondération qui fait passer sous le notionnel minimal → 0.
    assert manager(vol_target=0.01).position_size(100, 100, stop_price=95, realized_vol=1.0) == 0.0


@pytest.mark.parametrize(
    ("equity", "price", "stop"),
    [
        (0.0, 100.0, None),
        (-10.0, 100.0, None),
        (10_000.0, 0.0, None),
        (10_000.0, -5.0, None),
        (math.nan, 100.0, None),
        (10_000.0, math.nan, None),
        (10_000.0, math.inf, None),
        (math.inf, 100.0, None),
    ],
)
def test_position_size_bad_inputs_return_zero(
    equity: float, price: float, stop: float | None
) -> None:
    size = manager().position_size(equity, price, stop)
    assert size == 0.0
    assert not math.isnan(size)


def test_position_size_nan_stop_is_ignored() -> None:
    rm = manager()
    assert rm.position_size(10_000, 100, stop_price=math.nan) == pytest.approx(95.0)
    assert rm.position_size(10_000, 100, stop_price=math.inf) == pytest.approx(95.0)


def test_position_size_never_negative() -> None:
    rm = manager()
    for stop in (50.0, 150.0, 99.999, 0.0):
        assert rm.position_size(10_000, 100, stop_price=stop) >= 0.0


def test_invalid_config_rejected() -> None:
    with pytest.raises(ValueError):
        RiskManager(RiskConfig(risk_per_trade=0.0))
    with pytest.raises(ValueError):
        RiskManager(RiskConfig(max_position_fraction=-1))
    with pytest.raises(ValueError):
        RiskManager(RiskConfig(default_stop_atr_mult=0.0))
    with pytest.raises(ValueError):
        RiskManager(RiskConfig(default_stop_pct=1.0))
    with pytest.raises(ValueError):
        RiskManager(RiskConfig(default_stop_pct=0.0))
    with pytest.raises(ValueError):
        RiskManager(RiskConfig(max_drawdown=-0.1))
    with pytest.raises(ValueError):
        RiskManager(RiskConfig(vol_target=0.0))
    assert RiskManager().config == RiskConfig()  # config par défaut acceptée


@pytest.mark.parametrize("fraction", [1.01, 2.0, 95.0, math.inf])
def test_max_position_fraction_above_one_is_rejected_in_french(fraction: float) -> None:
    """F8 : au-delà de 1 (tout le capital) le sizing demande du levier et chaque ordre
    échoue sur « fonds insuffisants » ; on refuse dès la configuration."""
    with pytest.raises(ValueError, match="max_position_fraction.*entre 0.*et 1"):
        RiskManager(RiskConfig(max_position_fraction=fraction))


def test_max_position_fraction_one_is_accepted() -> None:
    rm = manager(max_position_fraction=1.0)
    assert rm.position_size(10_000, 100) == pytest.approx(100.0)


def test_default_stop_fields_defaults() -> None:
    cfg = RiskConfig()
    assert cfg.default_stop_atr_mult == 3.0
    assert cfg.default_stop_pct == 0.05
    assert "default_stop_atr_mult" in RiskManager(cfg).state()["config"]


# ============================================================================ stop protecteur
def test_protective_stop_uses_strategy_stop_when_on_the_right_side() -> None:
    rm = manager()
    assert rm.protective_stop(1, 100.0, strategy_stop=95.0, atr=2.0) == 95.0
    assert rm.protective_stop(-1, 100.0, strategy_stop=105.0, atr=2.0) == 105.0
    # stop_loss_pct fourni mais le stop de stratégie reste prioritaire
    assert rm.protective_stop(1, 100.0, strategy_stop=95.0, atr=2.0, stop_loss_pct=0.02) == 95.0


def test_protective_stop_wrong_side_or_nan_falls_through_to_pct_then_atr() -> None:
    rm = manager()
    # Stop long au-dessus du prix : ignoré → stop_loss_pct 2 % → 98.
    assert rm.protective_stop(1, 100.0, strategy_stop=105.0, atr=2.0, stop_loss_pct=0.02) == 98.0
    # Stop court sous le prix : ignoré → 102.
    assert rm.protective_stop(-1, 100.0, strategy_stop=95.0, atr=2.0, stop_loss_pct=0.02) == 102.0
    # Stop égal au prix : pas du bon côté strict → ATR : 100 − 3×2 = 94.
    assert rm.protective_stop(1, 100.0, strategy_stop=100.0, atr=2.0) == 94.0
    # Pas de stop du tout, pas de pct (None / 0) → ATR.
    assert rm.protective_stop(1, 100.0, atr=2.0) == 94.0
    assert rm.protective_stop(1, 100.0, strategy_stop=math.nan, atr=2.0, stop_loss_pct=0.0) == 94.0
    assert rm.protective_stop(-1, 100.0, atr=2.0) == 106.0
    rm2 = manager(default_stop_atr_mult=2.5)
    assert rm2.protective_stop(1, 100.0, atr=2.0) == 95.0


def test_protective_stop_last_resort_pct_when_atr_missing_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="tradingbot.risk")
    rm = manager()
    assert rm.protective_stop(1, 100.0) == pytest.approx(95.0)
    assert rm.protective_stop(1, 100.0, atr=math.nan) == pytest.approx(95.0)
    assert rm.protective_stop(1, 100.0, atr=0.0) == pytest.approx(95.0)
    assert rm.protective_stop(-1, 100.0) == pytest.approx(105.0)
    rm2 = manager(default_stop_pct=0.10)
    assert rm2.protective_stop(1, 200.0) == pytest.approx(180.0)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 5
    assert all("dernier recours" in r.getMessage() for r in warnings)


def test_protective_stop_rejects_invalid_side_or_price() -> None:
    rm = manager()
    with pytest.raises(ValueError, match="side"):
        rm.protective_stop(0, 100.0)
    with pytest.raises(ValueError, match="price"):
        rm.protective_stop(1, 0.0)
    with pytest.raises(ValueError, match="price"):
        rm.protective_stop(1, math.nan)


# ============================================================================ kill switch
def test_kill_switch_latches_until_reset() -> None:
    rm = manager(max_drawdown=0.20)
    rm.update(at(0), 10_000)
    assert rm.peak_equity == 10_000
    assert not rm.kill_switch_active()
    rm.update(at(1), 9_000)  # -10 %
    assert rm.drawdown() == pytest.approx(0.10)
    assert not rm.kill_switch_active()
    rm.update(at(2), 8_000)  # exactement -20 % : pas strictement supérieur
    assert not rm.kill_switch_active()
    rm.update(at(3), 7_900)  # -21 %
    assert rm.kill_switch_active()
    # Reste actif même si l'équité remonte au-dessus du plus haut.
    rm.update(at(4), 12_000)
    assert rm.kill_switch_active()
    assert rm.peak_equity == 12_000
    assert rm.can_open(1, at(4), 12_000) == (False, "kill switch actif : drawdown maximal dépassé")
    rm.reset()
    assert not rm.kill_switch_active()
    assert rm.peak_equity is None
    assert rm.drawdown() == 0.0
    assert rm.can_open(1, at(5), 12_000)[0]


def test_kill_switch_disabled_when_max_drawdown_none_or_zero() -> None:
    for value in (None, 0.0):
        rm = manager(max_drawdown=value, daily_loss_limit=None)
        rm.update(at(0), 10_000)
        rm.update(at(1), 1_000)
        assert not rm.kill_switch_active()
        assert rm.can_open(1, at(1), 1_000) == (True, "ok")


def test_peak_tracks_highest_equity() -> None:
    rm = manager(max_drawdown=0.5, daily_loss_limit=None)
    for i, eq in enumerate([10_000, 11_000, 10_500, 12_000, 11_000]):
        rm.update(at(i), eq)
    assert rm.peak_equity == 12_000
    assert rm.drawdown() == pytest.approx(1 - 11_000 / 12_000)


def test_update_ignores_invalid_equity() -> None:
    rm = manager()
    rm.update(at(0), 10_000)
    rm.update(at(1), math.nan)
    assert rm.peak_equity == 10_000
    assert rm.state()["last_equity"] == 10_000


# ============================================================================ limite journalière
def test_daily_limit_hit_and_reset_across_utc_days() -> None:
    rm = manager(daily_loss_limit=0.05, max_drawdown=None)
    rm.update(at(0), 10_000)  # 2024-03-10 00:00 UTC
    assert rm.day_start_equity == 10_000
    assert not rm.daily_limit_hit(at(0), 10_000)
    rm.update(at(5), 9_600)  # -4 %
    assert not rm.daily_limit_hit(at(5), 9_600)
    assert rm.can_open(1, at(5), 9_600) == (True, "ok")
    rm.update(at(6), 9_400)  # -6 % > 5 %
    assert rm.daily_limit_hit(at(6), 9_400)
    ok, reason = rm.can_open(1, at(6), 9_400)
    assert not ok and "journal" in reason
    # Le verrou reste posé même si l'équité remonte dans la journée.
    rm.update(at(7), 9_800)
    assert rm.daily_limit_hit(at(7), 9_800)
    assert not rm.can_open(1, at(7), 9_800)[0]
    # Jour UTC suivant : verrou levé, nouvelle référence = dernière équité du jour précédent.
    rm.update(at(25), 9_800)  # 2024-03-11 01:00 UTC
    assert rm.day_start_equity == 9_800
    assert not rm.daily_limit_hit(at(25), 9_800)
    assert rm.can_open(1, at(25), 9_800) == (True, "ok")
    assert rm.state()["day"] == "2024-03-11"
    # Un horodatage d'un autre jour que celui suivi n'est jamais « bloqué ».
    assert not rm.daily_limit_hit(at(50), 1.0)


def test_daily_limit_without_update_is_pure_check() -> None:
    rm = manager(daily_loss_limit=0.05)
    rm.update(at(0), 10_000)
    # Pas d'update à 9 000, mais la perte instantanée dépasse la limite → True sans verrou.
    assert rm.daily_limit_hit(at(1), 9_000)
    assert not rm.state()["daily_locked"]
    assert not rm.daily_limit_hit(at(1), 9_900)


def test_daily_limit_can_be_deactivated_by_the_caller() -> None:
    """M1 : sur des bougies >= 1 jour l'appelant désactive la limite ; elle ne se verrouille
    jamais et can_open ne la vérifie plus. Le choix n'est ni sérialisé ni effacé par reset."""
    rm = manager(daily_loss_limit=0.05, max_drawdown=None)
    assert rm.daily_limit_active
    rm.set_daily_limit_active(False, "bougies >= 1 jour")
    assert not rm.daily_limit_active
    assert rm.daily_limit_inactive_reason == "bougies >= 1 jour"
    rm.update(at(0), 10_000)
    rm.update(at(1), 9_000)  # -10 % : aurait verrouillé
    assert not rm.daily_limit_hit(at(1), 9_000)
    assert rm.can_open(1, at(1), 9_000) == (True, "ok")
    assert rm.state()["daily_locked"] is False
    assert "daily_limit_active" not in rm.state()
    rm.reset()
    assert not rm.daily_limit_active
    # Réactivation : la limite fonctionne à nouveau normalement.
    rm.set_daily_limit_active(True)
    assert rm.daily_limit_active and rm.daily_limit_inactive_reason == ""
    rm.update(at(2), 10_000)
    rm.update(at(3), 9_000)
    assert rm.daily_limit_hit(at(3), 9_000)
    assert not rm.can_open(1, at(3), 9_000)[0]
    # Désactiver lève un verrou déjà posé.
    rm.set_daily_limit_active(False, "test")
    assert not rm.daily_limit_hit(at(3), 9_000)
    assert "daily_limit_active=False" in repr(rm)


def test_daily_reference_after_multi_day_gap_is_current_equity() -> None:
    """F12 : après un redémarrage plusieurs jours plus tard, la dernière équité connue ne
    décrit plus le début du jour : on repart de l'équité courante."""
    rm = manager(daily_loss_limit=0.05, max_drawdown=None)
    rm.update(pd.Timestamp("2024-01-01 10:00", tz="UTC"), 10_000)
    rm.update(pd.Timestamp("2024-01-01 23:00", tz="UTC"), 9_800)
    rm2 = manager(daily_loss_limit=0.05, max_drawdown=None)
    rm2.load_state(rm.state())
    rm2.update(pd.Timestamp("2024-01-04 09:00", tz="UTC"), 9_400)
    assert rm2.day_start_equity == 9_400  # et non 9 800 (vieux de 3 jours)
    rm2.update(pd.Timestamp("2024-01-04 10:00", tz="UTC"), 9_300)  # -1,1 % sur la journée
    assert not rm2.daily_limit_hit(pd.Timestamp("2024-01-04 10:00", tz="UTC"), 9_300)
    # Sens inverse : gain pendant l'arrêt → la référence trop basse n'affaiblit plus la limite.
    rm3 = manager(daily_loss_limit=0.05, max_drawdown=None)
    rm3.load_state(rm.state())
    rm3.update(pd.Timestamp("2024-01-04 09:00", tz="UTC"), 10_800)
    assert rm3.day_start_equity == 10_800
    rm3.update(pd.Timestamp("2024-01-04 12:00", tz="UTC"), 10_200)  # -5,6 % sur la journée
    assert rm3.daily_limit_hit(pd.Timestamp("2024-01-04 12:00", tz="UTC"), 10_200)
    # Sans trou (23:00 → 00:00), la référence reste la dernière équité de la veille.
    rm4 = manager(daily_loss_limit=0.05, max_drawdown=None)
    rm4.update(pd.Timestamp("2024-01-01 23:00", tz="UTC"), 9_800)
    rm4.update(pd.Timestamp("2024-01-02 00:00", tz="UTC"), 9_700)
    assert rm4.day_start_equity == 9_800
    # Bougies journalières (écart exactement 1 jour) : la veille reste la référence.
    rm5 = manager(daily_loss_limit=0.05, max_drawdown=None)
    rm5.update(pd.Timestamp("2024-01-01", tz="UTC"), 9_800)
    rm5.update(pd.Timestamp("2024-01-02", tz="UTC"), 9_700)
    assert rm5.day_start_equity == 9_800
    # Premier update après load_state sans horodatage mémorisé : équité courante.
    rm6 = manager(daily_loss_limit=0.05, max_drawdown=None)
    rm6.load_state({"last_equity": 5_000.0, "day": "2024-01-01"})
    rm6.update(pd.Timestamp("2024-01-02 08:00", tz="UTC"), 9_000)
    assert rm6.day_start_equity == 9_000


def test_daily_limit_disabled() -> None:
    rm = manager(daily_loss_limit=None, max_drawdown=None)
    rm.update(at(0), 10_000)
    rm.update(at(1), 5_000)
    assert not rm.daily_limit_hit(at(1), 5_000)
    assert rm.can_open(1, at(1), 5_000)[0]


def test_reset_daily_explicit() -> None:
    rm = manager(daily_loss_limit=0.05, max_drawdown=None)
    rm.update(at(0), 10_000)
    rm.update(at(1), 9_000)
    assert rm.daily_limit_hit(at(1), 9_000)
    rm.reset_daily(at(2))  # même jour, forcé : référence = dernière équité (9 000)
    assert rm.day_start_equity == 9_000
    assert not rm.daily_limit_hit(at(2), 9_000)
    rm.reset_daily(at(3), equity=20_000)
    assert rm.day_start_equity == 20_000


def test_day_boundary_uses_utc_not_local_offset() -> None:
    rm = manager(daily_loss_limit=0.05, max_drawdown=None)
    plus2 = timezone(timedelta(hours=2))
    # 01:00 en UTC+2 = 23:00 UTC la veille.
    rm.update(datetime(2024, 3, 11, 1, 0, tzinfo=plus2), 10_000)
    assert rm.state()["day"] == "2024-03-10"
    rm.update(datetime(2024, 3, 11, 1, 0, tzinfo=UTC), 9_000)
    assert rm.state()["day"] == "2024-03-11"
    assert rm.day_start_equity == 10_000


def test_timestamps_accepted_in_many_forms() -> None:
    assert _to_utc("2024-03-10T12:00:00Z") == datetime(2024, 3, 10, 12, tzinfo=UTC)
    assert _to_utc("2024-03-10 12:00") == datetime(2024, 3, 10, 12, tzinfo=UTC)  # naïf = UTC
    assert _to_utc(datetime(2024, 3, 10, 12)) == datetime(2024, 3, 10, 12, tzinfo=UTC)
    assert _to_utc(pd.Timestamp("2024-03-10 14:00", tz="Europe/Paris")) == datetime(
        2024, 3, 10, 13, tzinfo=UTC
    )
    rm = manager()
    rm.update("2024-03-10", 10_000)
    rm.update(datetime(2024, 3, 10, 5), 9_000)
    assert rm.state()["day"] == "2024-03-10"


# ============================================================================ can_open
def test_can_open_reasons() -> None:
    rm = manager(allow_short=False)
    rm.update(at(0), 10_000)
    assert rm.can_open(1, at(0), 10_000) == (True, "ok")
    ok, reason = rm.can_open(-1, at(0), 10_000)
    assert not ok and "découvert" in reason
    ok, reason = rm.can_open(0, at(0), 10_000)
    assert not ok and "plat" in reason
    ok, reason = rm.can_open(1, at(0), 0.0)
    assert not ok and "équité" in reason
    assert manager(allow_short=True).can_open(-1, at(0), 10_000) == (True, "ok")


def test_can_open_before_any_update() -> None:
    rm = manager()
    assert rm.can_open(1, at(0), 10_000) == (True, "ok")


# ============================================================================ état
def test_state_round_trip_is_json_serialisable() -> None:
    rm = manager(max_drawdown=0.1, daily_loss_limit=0.05)
    rm.update(at(0), 10_000)
    rm.update(at(3), 9_300)  # -7 % : kill switch (10 %) non, limite journalière oui
    assert rm.daily_limit_hit(at(3), 9_300)
    rm.update(at(4), 8_900)  # -11 % : kill switch
    state = rm.state()
    encoded = json.dumps(state)
    decoded = json.loads(encoded)
    assert decoded["kill_switch"] is True
    assert decoded["daily_locked"] is True
    assert decoded["peak_equity"] == 10_000
    assert decoded["day_start_equity"] == 10_000
    assert decoded["last_equity"] == 8_900
    assert decoded["day"] == "2024-03-10"
    assert decoded["last_timestamp"] == "2024-03-10T04:00:00+00:00"
    assert decoded["config"]["max_drawdown"] == 0.1

    other = manager(max_drawdown=0.1, daily_loss_limit=0.05)
    other.load_state(decoded)
    assert other.state() == state
    assert other.kill_switch_active()
    assert other.daily_limit_hit(at(5), 9_900)
    assert other.peak_equity == 10_000
    assert other.drawdown() == pytest.approx(0.11)
    # Le lendemain, la limite journalière se lève mais le kill switch reste.
    other.update(at(30), 9_000)
    assert not other.daily_limit_hit(at(30), 9_000)
    assert other.kill_switch_active()


def test_state_of_fresh_manager() -> None:
    state = manager().state()
    json.dumps(state)
    assert state["peak_equity"] is None
    assert state["kill_switch"] is False
    assert state["day"] is None
    assert state["last_timestamp"] is None
    fresh = manager()
    fresh.load_state(state)
    assert fresh.state() == state


def test_load_state_tolerates_missing_keys() -> None:
    rm = manager()
    rm.load_state({"peak_equity": 500.0})
    assert rm.peak_equity == 500.0
    assert not rm.kill_switch_active()
    assert "RiskManager(" in repr(rm)


def test_kill_switch_and_daily_limit_are_logged_at_info_not_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """En backtest ces événements sont simulés des centaines de fois : INFO, pas WARNING
    (le LiveTrader avertit et notifie lui-même en temps réel)."""
    caplog.set_level(logging.DEBUG, logger="tradingbot.risk")
    rm = manager()
    rm.update(at(0), 1000.0)
    rm.update(at(1), 940.0)  # -6 % sur la journée → limite journalière
    rm.update(at(2), 790.0)  # -21 % depuis le plus haut → kill switch
    records = [r for r in caplog.records if r.name == "tradingbot.risk"]
    messages = [r.getMessage() for r in records]
    assert any(m.startswith("Limite de perte journalière atteinte") for m in messages)
    assert any(m.startswith("Kill switch déclenché") for m in messages)
    assert {r.levelno for r in records} == {logging.INFO}
    assert rm.kill_switch_active() and rm.daily_limit_hit(at(2), 790.0)
