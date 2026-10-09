"""Tests du chargement de configuration : TOML, défauts, .env, précédence des secrets,
validation (mode, timeframe, types), masquage, conversion des `--param`."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest

from tradingbot.config import (
    ENV_API_KEY,
    ENV_API_SECRET,
    ENV_TELEGRAM_CHAT_ID,
    ENV_TELEGRAM_TOKEN,
    MASK,
    CostSettings,
    GeneralSettings,
    NotificationSettings,
    Settings,
    StrategySettings,
    load_dotenv,
    load_settings,
    parse_param_overrides,
    parse_scalar,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = REPO_ROOT / "config.example.toml"
LOGGER_NAME = "tradingbot.config"

NO_ENV: dict[str, str] = {}

FULL_CONFIG = """
[general]
mode = "live"
exchange = "Kraken"
symbol = "ETH/USD"
timeframe = "4h"
state_dir = "etat"

[strategy]
name = "sma_cross"

[strategy.params]
fast = 10
slow = 30
threshold = 0.5
use_filter = true
label = "x"

[risk]
initial_cash = 2500.0
risk_per_trade = 0.02
max_position_fraction = 0.5
max_drawdown = 0.1
daily_loss_limit = 0.03
allow_short = true
min_trade_notional = 25
vol_target = 0.3

[costs]
fee_rate = 0.002
slippage = 0.001

[notifications]
telegram = true
"""

FULL_ENV = {
    ENV_API_KEY: "key-123",
    ENV_API_SECRET: "secret-456",
    ENV_TELEGRAM_TOKEN: "tok:789",
    ENV_TELEGRAM_CHAT_ID: "42",
}


@pytest.fixture
def isolated_environ(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Remplace ``os.environ`` par un dict vide le temps du test (aucune fuite)."""
    fake: dict[str, str] = {}
    monkeypatch.setattr(os, "environ", fake)
    return fake


def write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def warnings_of(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


def bot_keys_in(environ: dict[str, str]) -> set[str]:
    """Variables du robot présentes dans ``environ`` (pytest y ajoute PYTEST_CURRENT_TEST)."""
    return set(environ) & {ENV_API_KEY, ENV_API_SECRET, ENV_TELEGRAM_TOKEN, ENV_TELEGRAM_CHAT_ID}


# ============================================================================ load_dotenv
def test_load_dotenv_parses_comments_quotes_export_and_inline_comments(
    tmp_path: Path, isolated_environ: dict[str, str], caplog: pytest.LogCaptureFixture
) -> None:
    lines = [
        "# commentaire",
        '  TRADINGBOT_API_KEY="abc=def"   # un = dans la valeur citée',
        "TRADINGBOT_API_SECRET='s3cr3t'",
        "export TELEGRAM_BOT_TOKEN=123:ABC",
        "TELEGRAM_CHAT_ID = 42 # commentaire en fin de ligne",
        "",
        "EMPTY=",
        "LIGNE SANS EGAL",
        "VALUE_WITH_HASH=a#b",
        "   ",
    ]
    dotenv = write(tmp_path / ".env", "\n".join(lines) + "\n")
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)

    parsed = load_dotenv(dotenv)

    assert parsed == {
        "TRADINGBOT_API_KEY": "abc=def",
        "TRADINGBOT_API_SECRET": "s3cr3t",
        "TELEGRAM_BOT_TOKEN": "123:ABC",
        "TELEGRAM_CHAT_ID": "42",
        "EMPTY": "",
        "VALUE_WITH_HASH": "a#b",
    }
    # la ligne 8 est malformée : avertissement (avec le numéro de ligne), pas d'exception
    msgs = warnings_of(caplog)
    assert len(msgs) == 1
    assert "Ligne 8" in msgs[0]
    # les valeurs lues sont exportées dans os.environ
    assert isolated_environ["TELEGRAM_BOT_TOKEN"] == "123:ABC"
    assert isolated_environ["EMPTY"] == ""


def test_load_dotenv_does_not_override_existing_environ(
    tmp_path: Path, isolated_environ: dict[str, str]
) -> None:
    isolated_environ["TRADINGBOT_API_KEY"] = "from-os"
    dotenv = write(tmp_path / ".env", "TRADINGBOT_API_KEY=from-file\nTELEGRAM_CHAT_ID=7\n")

    parsed = load_dotenv(dotenv)

    assert parsed == {"TRADINGBOT_API_KEY": "from-file", "TELEGRAM_CHAT_ID": "7"}
    assert isolated_environ["TRADINGBOT_API_KEY"] == "from-os"
    assert isolated_environ["TELEGRAM_CHAT_ID"] == "7"


def test_load_dotenv_missing_file_returns_empty_dict(
    tmp_path: Path, isolated_environ: dict[str, str]
) -> None:
    assert load_dotenv(tmp_path / "absent.env") == {}
    assert not bot_keys_in(isolated_environ)


def test_load_dotenv_accepts_str_path_and_utf8_bom(
    tmp_path: Path, isolated_environ: dict[str, str]
) -> None:
    path = tmp_path / ".env"
    path.write_bytes(b"\xef\xbb\xbfTRADINGBOT_API_KEY=k\n")
    assert load_dotenv(str(path)) == {"TRADINGBOT_API_KEY": "k"}


# ============================================================================ load_settings
def test_full_file_every_field(tmp_path: Path) -> None:
    cfg = write(tmp_path / "config.toml", FULL_CONFIG)
    s = load_settings(cfg, env=FULL_ENV)

    assert s.general == GeneralSettings(
        mode="live", exchange="kraken", symbol="ETH/USD", timeframe="4h", state_dir="etat"
    )
    assert s.strategy == StrategySettings(
        name="sma_cross",
        params={"fast": 10, "slow": 30, "threshold": 0.5, "use_filter": True, "label": "x"},
    )
    assert isinstance(s.strategy.params["fast"], int)
    assert s.initial_cash == 2500.0
    assert s.risk.risk_per_trade == 0.02
    assert s.risk.max_position_fraction == 0.5
    assert s.risk.max_drawdown == 0.1
    assert s.risk.daily_loss_limit == 0.03
    assert s.risk.allow_short is True
    # entier TOML converti en float pour un champ float
    assert s.risk.min_trade_notional == 25.0
    assert isinstance(s.risk.min_trade_notional, float)
    assert s.risk.vol_target == 0.3
    assert s.costs == CostSettings(fee_rate=0.002, slippage=0.001)
    assert s.notifications == NotificationSettings(telegram=True)
    assert s.api_key == "key-123"
    assert s.api_secret == "secret-456"
    assert s.telegram_token == "tok:789"
    assert s.telegram_chat_id == "42"
    assert s.is_live is True
    assert s.has_api_keys is True
    assert s.has_telegram is True


def test_full_file_risk_section_builds_a_real_riskconfig(tmp_path: Path) -> None:
    risk_mod = pytest.importorskip("tradingbot.risk")
    cfg = write(tmp_path / "config.toml", FULL_CONFIG)
    s = load_settings(cfg, env=NO_ENV)
    assert isinstance(s.risk, risk_mod.RiskConfig)
    # la config chargée est acceptée telle quelle par le gestionnaire de risque
    manager = risk_mod.RiskManager(s.risk)
    assert manager.config is s.risk


def test_missing_file_uses_example_defaults_and_logs_info(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    missing = tmp_path / "absent.toml"

    s = load_settings(missing, env=NO_ENV)

    infos = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert any("introuvable" in m and str(missing) in m for m in infos)
    assert not warnings_of(caplog)
    assert s.general == GeneralSettings(
        mode="paper", exchange="binance", symbol="BTC/USDT", timeframe="1h", state_dir="state"
    )
    assert s.strategy == StrategySettings(name="ema_trend", params={})
    assert s.initial_cash == 1000.0
    assert s.risk.risk_per_trade == 0.01
    assert s.risk.max_position_fraction == 0.95
    assert s.risk.max_drawdown == 0.20
    assert s.risk.daily_loss_limit == 0.05
    assert s.risk.allow_short is False
    assert s.costs == CostSettings(fee_rate=0.001, slippage=0.0005)
    assert s.notifications == NotificationSettings(telegram=False)
    assert s.api_key is None and s.api_secret is None
    assert s.telegram_token is None and s.telegram_chat_id is None
    assert s.is_live is False
    assert s.has_api_keys is False
    # les défauts du code, ceux du fichier absent et ceux de config.example.toml coïncident
    assert s == load_settings(EXAMPLE_CONFIG, env=NO_ENV)
    assert s == Settings()


def test_partial_file_keeps_defaults_for_the_rest(tmp_path: Path) -> None:
    cfg = write(
        tmp_path / "config.toml",
        '[general]\nsymbol = "ETH/USDT"\n\n[risk]\nmax_drawdown = 0.1\n',
    )
    s = load_settings(cfg, env=NO_ENV)
    assert s.general.symbol == "ETH/USDT"
    assert s.general.mode == "paper"
    assert s.general.timeframe == "1h"
    assert s.risk.max_drawdown == 0.1
    assert s.risk.risk_per_trade == 0.01
    assert s.initial_cash == 1000.0
    assert s.costs.fee_rate == 0.001
    assert s.strategy.name == "ema_trend"


def test_example_config_loads_without_any_warning(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    s = load_settings(EXAMPLE_CONFIG, env=NO_ENV)
    assert warnings_of(caplog) == []
    assert s.general.mode == "paper"
    assert s.initial_cash == 1000.0


# ---------------------------------------------------------------------------- validation
@pytest.mark.parametrize("mode", ["demo", "real", "", "papier"])
def test_invalid_mode_raises_french_valueerror(tmp_path: Path, mode: str) -> None:
    cfg = write(tmp_path / "config.toml", f'[general]\nmode = "{mode}"\n')
    with pytest.raises(ValueError, match=r"\[general\] mode invalide.*paper.*live"):
        load_settings(cfg, env=NO_ENV)


def test_mode_is_normalised_case_and_spaces(tmp_path: Path) -> None:
    cfg = write(tmp_path / "config.toml", '[general]\nmode = " Live "\n')
    s = load_settings(cfg, env=NO_ENV)
    assert s.general.mode == "live"
    assert s.is_live


@pytest.mark.parametrize("timeframe", ["1x", "h1", "", "0h", "1.5h", "1 h", "hour", "-1h"])
def test_invalid_timeframe_raises_french_valueerror(tmp_path: Path, timeframe: str) -> None:
    cfg = write(tmp_path / "config.toml", f'[general]\ntimeframe = "{timeframe}"\n')
    with pytest.raises(ValueError, match=r"\[general\] timeframe invalide"):
        load_settings(cfg, env=NO_ENV)


@pytest.mark.parametrize("timeframe", ["1m", "5m", "15m", "1h", "4h", "1d", "1w", "4H", "12h"])
def test_valid_timeframes(tmp_path: Path, timeframe: str) -> None:
    cfg = write(tmp_path / "config.toml", f'[general]\ntimeframe = "{timeframe}"\n')
    assert load_settings(cfg, env=NO_ENV).general.timeframe == timeframe


@pytest.mark.parametrize(
    ("toml", "key"),
    [
        ('[costs]\nfee_rate = "0.1"\n', "fee_rate"),
        ('[notifications]\ntelegram = "oui"\n', "telegram"),
        ("[notifications]\ntelegram = 1\n", "telegram"),
        ("[general]\nmode = 1\n", "mode"),
        ('[risk]\nmax_drawdown = "aucun"\n', "max_drawdown"),
        ("[risk]\nallow_short = 0\n", "allow_short"),
        ('[strategy]\nparams = "fast=10"\n', "params"),
        ('[risk]\ninitial_cash = "1000"\n', "initial_cash"),
    ],
)
def test_wrong_value_type_raises_valueerror_naming_the_key(
    tmp_path: Path, toml: str, key: str
) -> None:
    cfg = write(tmp_path / "config.toml", toml)
    with pytest.raises(ValueError, match=rf"{key} : valeur invalide"):
        load_settings(cfg, env=NO_ENV)


@pytest.mark.parametrize(
    "toml",
    [
        "[costs]\nfee_rate = 1.5\n",
        "[costs]\nslippage = -0.001\n",
        "[risk]\ninitial_cash = 0\n",
        "[risk]\ninitial_cash = -5\n",
        '[general]\nsymbol = ""\n',
        '[general]\nexchange = "  "\n',
        '[strategy]\nname = ""\n',
    ],
)
def test_out_of_range_or_empty_values_raise(tmp_path: Path, toml: str) -> None:
    cfg = write(tmp_path / "config.toml", toml)
    with pytest.raises(ValueError):
        load_settings(cfg, env=NO_ENV)


def test_section_that_is_not_a_table_raises(tmp_path: Path) -> None:
    cfg = write(tmp_path / "config.toml", 'general = "paper"\n')
    with pytest.raises(ValueError, match=r"section \[general\] doit être une table"):
        load_settings(cfg, env=NO_ENV)


def test_toml_syntax_error_raises_french_valueerror(tmp_path: Path) -> None:
    cfg = write(tmp_path / "config.toml", "[general\nmode = paper\n")
    with pytest.raises(ValueError, match="illisible"):
        load_settings(cfg, env=NO_ENV)


def test_unknown_section_and_key_warn_but_do_not_fail(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
    cfg = write(
        tmp_path / "config.toml",
        '[general]\nmode = "paper"\ncolour = "blue"\n\n[plugins]\nfoo = 1\n\n[risk]\nlever = 3\n',
    )
    s = load_settings(cfg, env=NO_ENV)
    msgs = warnings_of(caplog)
    assert len(msgs) == 3
    assert any("[plugins]" in m and "Section inconnue" in m for m in msgs)
    assert any("[general]" in m and "'colour'" in m for m in msgs)
    assert any("[risk]" in m and "'lever'" in m for m in msgs)
    assert s.general.mode == "paper"
    assert s.risk.risk_per_trade == 0.01


def test_strategy_params_keys_are_free_form_and_never_warn(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
    cfg = write(tmp_path / "config.toml", "[strategy.params]\nanything = 3\nother = [1, 2]\n")
    s = load_settings(cfg, env=NO_ENV)
    assert s.strategy.params == {"anything": 3, "other": [1, 2]}
    assert warnings_of(caplog) == []


def test_live_without_api_keys_warns(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
    cfg = write(tmp_path / "config.toml", '[general]\nmode = "live"\n')
    s = load_settings(cfg, env=NO_ENV)
    assert s.is_live and not s.has_api_keys
    assert any(ENV_API_KEY in m and "live" in m for m in warnings_of(caplog))


def test_telegram_enabled_without_token_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
    cfg = write(tmp_path / "config.toml", "[notifications]\ntelegram = true\n")
    s = load_settings(cfg, env={ENV_TELEGRAM_TOKEN: "tok"})  # chat_id manquant
    assert s.notifications.telegram and not s.has_telegram
    assert any(ENV_TELEGRAM_CHAT_ID in m for m in warnings_of(caplog))


# ---------------------------------------------------------------------------- secrets
def test_explicit_env_dict_takes_precedence_over_dotenv_file(
    tmp_path: Path, isolated_environ: dict[str, str]
) -> None:
    cfg = write(tmp_path / "config.toml", '[general]\nmode = "paper"\n')
    write(tmp_path / ".env", f"{ENV_API_KEY}=from-dotenv\n{ENV_API_SECRET}=dotenv-secret\n")

    s = load_settings(cfg, env={ENV_API_KEY: "explicit"})

    assert s.api_key == "explicit"
    assert s.api_secret is None  # le .env n'est pas consulté quand env est fourni
    assert not bot_keys_in(isolated_environ)  # ... et os.environ n'est pas touché
    assert load_settings(cfg, env=NO_ENV).api_key is None


def test_dotenv_next_to_config_then_os_environ_wins(
    tmp_path: Path, isolated_environ: dict[str, str]
) -> None:
    cfg_dir = tmp_path / "projet"
    cfg_dir.mkdir()
    cfg = write(cfg_dir / "config.toml", '[general]\nmode = "paper"\n')
    write(cfg_dir / ".env", f"{ENV_API_KEY}=from-dotenv\n{ENV_API_SECRET}=dotenv-secret\n")
    isolated_environ[ENV_API_SECRET] = "os-secret"

    s = load_settings(cfg)

    assert s.api_key == "from-dotenv"
    assert s.api_secret == "os-secret"
    assert isolated_environ[ENV_API_KEY] == "from-dotenv"
    assert isolated_environ[ENV_API_SECRET] == "os-secret"


def test_dotenv_falls_back_to_current_directory(
    tmp_path: Path, isolated_environ: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_dir = tmp_path / "conf"
    cfg_dir.mkdir()
    cfg = write(cfg_dir / "config.toml", '[general]\nmode = "paper"\n')
    write(tmp_path / ".env", f"{ENV_TELEGRAM_TOKEN}=cwd-token\n")
    monkeypatch.chdir(tmp_path)

    s = load_settings(cfg)

    assert s.telegram_token == "cwd-token"


def test_missing_config_without_any_dotenv_reads_os_environ_only(
    tmp_path: Path, isolated_environ: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    isolated_environ[ENV_API_KEY] = "os-key"
    s = load_settings()  # config.toml absent dans le répertoire courant
    assert s.api_key == "os-key"
    assert s.api_secret is None
    assert s.general.mode == "paper"


def test_blank_secret_is_none(tmp_path: Path) -> None:
    s = load_settings(tmp_path / "absent.toml", env={ENV_API_KEY: "   ", ENV_API_SECRET: ""})
    assert s.api_key is None
    assert s.api_secret is None
    assert s.has_api_keys is False


def test_secret_values_are_stripped(tmp_path: Path) -> None:
    s = load_settings(tmp_path / "absent.toml", env={ENV_API_KEY: "  k1 ", ENV_API_SECRET: "s1"})
    assert s.api_key == "k1"
    assert s.has_api_keys is True


# ---------------------------------------------------------------------------- masquage
def test_to_dict_masks_secrets_and_is_json_serialisable(tmp_path: Path) -> None:
    s = load_settings(tmp_path / "absent.toml", env=FULL_ENV)
    d = s.to_dict()

    assert d["api_key"] == MASK
    assert d["api_secret"] == MASK
    assert d["telegram_token"] == MASK
    assert d["telegram_chat_id"] == MASK
    assert d["general"] == {
        "mode": "paper",
        "exchange": "binance",
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "state_dir": "state",
    }
    assert d["risk"]["risk_per_trade"] == 0.01
    assert d["costs"] == {"fee_rate": 0.001, "slippage": 0.0005}
    assert d["notifications"] == {"telegram": False}
    assert d["initial_cash"] == 1000.0
    dumped = json.dumps(d)
    for secret in FULL_ENV.values():
        assert secret not in dumped
    # l'objet Settings lui-même n'est pas modifié
    assert s.api_key == "key-123"


def test_to_dict_keeps_none_for_unset_secrets() -> None:
    d = Settings().to_dict()
    assert d["api_key"] is None
    assert d["telegram_chat_id"] is None


def test_repr_never_leaks_secrets(tmp_path: Path) -> None:
    s = load_settings(tmp_path / "absent.toml", env=FULL_ENV)
    text = repr(s)
    assert MASK in text
    for secret in FULL_ENV.values():
        assert secret not in text


# ============================================================================ --param
def test_parse_param_overrides_converts_int_float_bool_str() -> None:
    out = parse_param_overrides(["fast=10", "slow=2.5", "x=true", "name=abc"])
    assert out == {"fast": 10, "slow": 2.5, "x": True, "name": "abc"}
    assert isinstance(out["fast"], int) and not isinstance(out["fast"], bool)
    assert isinstance(out["slow"], float)
    assert out["x"] is True
    assert isinstance(out["name"], str)


def test_parse_param_overrides_more_conversions() -> None:
    out = parse_param_overrides(
        [
            "y=False",
            "z=-3",
            "w=1e3",
            "v=.5",
            'q="quoted"',
            "r='single'",
            "n=none",
            " k = v ",
            "eq=a=b",
            "empty=",
        ]
    )
    assert out == {
        "y": False,
        "z": -3,
        "w": 1000.0,
        "v": 0.5,
        "q": "quoted",
        "r": "single",
        "n": None,
        "k": "v",
        "eq": "a=b",
        "empty": "",
    }
    assert isinstance(out["w"], float)


def test_parse_param_overrides_last_value_wins_and_empty_input() -> None:
    assert parse_param_overrides(["a=1", "a=2"]) == {"a": 2}
    assert parse_param_overrides([]) == {}
    assert parse_param_overrides(None) == {}


@pytest.mark.parametrize("item", ["fast", "=10", "fa st=10", "1x=2", ""])
def test_parse_param_overrides_rejects_malformed_items(item: str) -> None:
    with pytest.raises(ValueError, match="invalide"):
        parse_param_overrides([item])


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("TRUE", True),
        ("false", False),
        ("007", 7),
        ("+4", 4),
        ("3.", 3.0),
        ("1E-2", 0.01),
        ("nan", "nan"),  # pas converti en float : reste une chaîne
        ("inf", "inf"),
        ("null", None),
        ("BTC/USDT", "BTC/USDT"),
        ("'", "'"),  # un seul guillemet : inchangé
    ],
)
def test_parse_scalar(text: str, expected: object) -> None:
    result = parse_scalar(text)
    assert result == expected
    assert type(result) is type(expected)
