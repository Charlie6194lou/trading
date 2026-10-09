"""Chargement de la configuration du robot.

Trois sources, de la moins à la plus prioritaire pour les secrets :

1. ``config.toml`` (copie adaptée de ``config.example.toml``) pour tout ce qui n'est pas
   secret : mode, exchange, symbole, stratégie, risque, frais, notifications ;
2. un fichier ``.env`` (format ``CLE=valeur``) situé à côté du fichier de configuration,
   ou à défaut dans le répertoire courant ;
3. les variables d'environnement du système, prioritaires sur ``.env``.

Les clés API ne sont JAMAIS lues depuis ``config.toml`` : uniquement depuis ``.env`` ou
l'environnement, afin qu'elles ne soient pas publiées par erreur. Les valeurs absentes du
fichier prennent les mêmes défauts que ``config.example.toml`` ; si le fichier n'existe
pas, le robot démarre en mode ``paper`` (argent fictif) avec ces défauts.
"""

from __future__ import annotations

import logging
import os
import re
import tomllib
import types
import typing
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, get_args, get_origin, get_type_hints

if TYPE_CHECKING:
    from tradingbot.risk import RiskConfig

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "ENV_API_KEY",
    "ENV_API_SECRET",
    "ENV_TELEGRAM_CHAT_ID",
    "ENV_TELEGRAM_TOKEN",
    "MASK",
    "SECRET_FIELDS",
    "VALID_MODES",
    "CostSettings",
    "GeneralSettings",
    "NotificationSettings",
    "Settings",
    "StrategySettings",
    "load_dotenv",
    "load_settings",
    "parse_param_overrides",
    "parse_scalar",
]

#: chemin par défaut du fichier de configuration (relatif au répertoire courant)
DEFAULT_CONFIG_PATH = "config.toml"
#: modes de fonctionnement acceptés
VALID_MODES: tuple[str, ...] = ("paper", "live")

#: noms des variables d'environnement (voir ``.env.example``)
ENV_API_KEY = "TRADINGBOT_API_KEY"
ENV_API_SECRET = "TRADINGBOT_API_SECRET"
ENV_TELEGRAM_TOKEN = "TELEGRAM_BOT_TOKEN"
ENV_TELEGRAM_CHAT_ID = "TELEGRAM_CHAT_ID"

#: champs de :class:`Settings` masqués par :meth:`Settings.to_dict`
SECRET_FIELDS: tuple[str, ...] = ("api_key", "api_secret", "telegram_token", "telegram_chat_id")
#: valeur affichée à la place d'un secret
MASK = "***"

_KNOWN_SECTIONS: tuple[str, ...] = ("general", "strategy", "risk", "costs", "notifications")
# même grammaire que tradingbot.data.loader.timeframe_to_timedelta : "15m", "1h", "4H", "1d", "1w"
_TIMEFRAME_RE = re.compile(r"^(\d+)([mhdwHDW])$")
_DOTENV_LINE_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")
_INLINE_COMMENT_RE = re.compile(r"\s#")
_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?$")

_T = TypeVar("_T")


# --- dataclasses (miroir de config.example.toml) ------------------------------------


@dataclass
class GeneralSettings:
    """Section ``[general]`` : quoi trader, où, et avec quel argent (fictif ou réel)."""

    #: "paper" (argent fictif) ou "live" (argent réel)
    mode: str = "paper"
    #: identifiant ccxt de l'exchange (binance, kraken, coinbase, bybit...)
    exchange: str = "binance"
    #: paire au format ccxt, ex. "BTC/USDT"
    symbol: str = "BTC/USDT"
    #: durée d'une bougie : 1m, 5m, 15m, 1h, 4h, 1d...
    timeframe: str = "1h"
    #: répertoire où le bot mémorise ses positions / trades
    state_dir: str = "state"


@dataclass
class StrategySettings:
    """Section ``[strategy]`` : nom de la stratégie et ses paramètres optionnels."""

    #: identifiant court (voir ``tradingbot list-strategies``)
    name: str = "ema_trend"
    #: paramètres passés à la stratégie (table ``[strategy.params]``) ; vide = défauts
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class CostSettings:
    """Section ``[costs]`` : frais et glissement estimés, en fraction du notionnel."""

    #: frais par ordre (0.001 = 0,1 %)
    fee_rate: float = 0.001
    #: glissement défavorable par ordre (0.0005 = 0,05 %)
    slippage: float = 0.0005


@dataclass
class NotificationSettings:
    """Section ``[notifications]``."""

    #: envoyer les trades sur Telegram (jeton et chat dans ``.env``)
    telegram: bool = False


def _risk_config_cls() -> type[RiskConfig]:
    # Import paresseux : le module de risque est chargé seulement quand on en a besoin.
    from tradingbot.risk import RiskConfig

    return RiskConfig


def _default_risk() -> RiskConfig:
    return _risk_config_cls()()


@dataclass(repr=False)
class Settings:
    """Configuration complète du robot (miroir de ``config.example.toml`` + secrets).

    Les secrets (``api_key``, ``api_secret``, ``telegram_token``, ``telegram_chat_id``)
    viennent exclusivement de ``.env`` ou des variables d'environnement. ``repr()`` et
    :meth:`to_dict` les masquent toujours.
    """

    general: GeneralSettings = field(default_factory=GeneralSettings)
    strategy: StrategySettings = field(default_factory=StrategySettings)
    risk: RiskConfig = field(default_factory=_default_risk)
    costs: CostSettings = field(default_factory=CostSettings)
    notifications: NotificationSettings = field(default_factory=NotificationSettings)
    #: capital de départ (paper) ou de référence (live), clé ``initial_cash`` de ``[risk]``
    initial_cash: float = 1000.0
    api_key: str | None = None
    api_secret: str | None = None
    telegram_token: str | None = None
    telegram_chat_id: str | None = None

    @property
    def is_live(self) -> bool:
        """True si ``mode = "live"`` (argent réel)."""
        return self.general.mode == "live"

    @property
    def has_api_keys(self) -> bool:
        """True si la clé ET le secret API sont renseignés."""
        return bool(self.api_key and self.api_secret)

    @property
    def has_telegram(self) -> bool:
        """True si le jeton ET l'identifiant de chat Telegram sont renseignés."""
        return bool(self.telegram_token and self.telegram_chat_id)

    def to_dict(self) -> dict[str, Any]:
        """Dictionnaire sérialisable (JSON) avec les secrets remplacés par ``"***"``.

        Un secret absent reste ``None`` : on peut ainsi savoir s'il est configuré sans
        jamais l'afficher.
        """
        data = asdict(self)
        for name in SECRET_FIELDS:
            if data.get(name):
                data[name] = MASK
        return data

    def __repr__(self) -> str:
        return f"Settings({self.to_dict()!r})"


# --- .env -----------------------------------------------------------------------------


def _parse_dotenv_value(raw: str) -> str:
    """Valeur d'une ligne ``.env`` : guillemets retirés, commentaire en fin de ligne ignoré."""
    if raw and raw[0] in "\"'":
        closing = raw.find(raw[0], 1)
        if closing != -1:
            return raw[1:closing]
        return raw
    match = _INLINE_COMMENT_RE.search(raw)
    if match:
        raw = raw[: match.start()]
    return raw.strip()


def load_dotenv(path: str | Path = ".env") -> dict[str, str]:
    """Lit un fichier ``.env`` (lignes ``CLE=valeur``) sans dépendance externe.

    - lignes vides et commentaires (``#``) ignorés, préfixe ``export`` accepté ;
    - guillemets simples ou doubles retirés ; un commentaire après une valeur non
      citée est ignoré ;
    - les variables déjà présentes dans ``os.environ`` ne sont PAS écrasées, les autres
      y sont ajoutées ;
    - renvoie le dictionnaire des valeurs lues dans le fichier (vide s'il n'existe pas).
    """
    dotenv_path = Path(path)
    if not dotenv_path.is_file():
        logger.debug("Aucun fichier .env en %s", dotenv_path)
        return {}
    parsed: dict[str, str] = {}
    text = dotenv_path.read_text(encoding="utf-8-sig")
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _DOTENV_LINE_RE.match(line)
        if match is None:
            logger.warning(
                "Ligne %d de %s ignorée (format attendu : CLE=valeur)", lineno, dotenv_path
            )
            continue
        key, raw_value = match.group(1), match.group(2)
        parsed[key] = _parse_dotenv_value(raw_value.strip())
    for key, value in parsed.items():
        os.environ.setdefault(key, value)
    logger.debug("%d variable(s) lue(s) dans %s", len(parsed), dotenv_path)
    return parsed


def _find_dotenv(config_path: Path) -> Path:
    """``.env`` à côté du fichier de configuration, sinon dans le répertoire courant."""
    candidates = (config_path.resolve().parent / ".env", Path.cwd() / ".env")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


# --- TOML -> dataclasses ---------------------------------------------------------------


def _read_toml(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        logger.info(
            "Fichier de configuration %s introuvable : valeurs par défaut utilisées "
            "(mode paper, identiques à config.example.toml)",
            config_path,
        )
        return {}
    try:
        with config_path.open("rb") as fh:
            raw = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Fichier de configuration {config_path} illisible : {exc}") from exc
    logger.info("Configuration chargée depuis %s", config_path)
    return raw


def _table(raw: Mapping[str, Any], name: str) -> dict[str, Any]:
    """Copie de la section ``[name]`` (vide si absente) ; erreur si ce n'est pas une table."""
    table = raw.get(name)
    if table is None:
        return {}
    if not isinstance(table, Mapping):
        raise ValueError(
            f"La section [{name}] doit être une table TOML (clés sous [{name}]), "
            f"pas {type(table).__name__}"
        )
    return dict(table)


def _type_label(hint: Any) -> str:
    """Description en français du type attendu, pour les messages d'erreur."""
    origin = get_origin(hint)
    if origin in (types.UnionType, typing.Union):
        labels = [_type_label(a) for a in get_args(hint) if a is not type(None)]
        return " ou ".join(dict.fromkeys(labels))
    if hint is bool:
        return "true ou false"
    if hint is int:
        return "un entier"
    if hint is float:
        return "un nombre"
    if hint is str:
        return "une chaîne de caractères"
    if hint is dict or origin is dict:
        return "une table"
    return "une valeur"


def _coerce(section: str, key: str, value: Any, hint: Any) -> Any:
    """Vérifie / convertit ``value`` selon ``hint`` ; lève ``ValueError`` (en français)."""
    origin = get_origin(hint)
    if origin in (types.UnionType, typing.Union):
        if value is None:
            return None
        for alternative in get_args(hint):
            if alternative is type(None):
                continue
            try:
                return _coerce(section, key, value, alternative)
            except ValueError:
                continue
    elif hint is bool:
        if isinstance(value, bool):
            return value
    elif hint is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    elif hint is float:
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    elif hint is str:
        if isinstance(value, str):
            return value
    elif hint is dict or origin is dict:
        if isinstance(value, Mapping):
            return dict(value)
    else:
        return value
    raise ValueError(f"[{section}] {key} : valeur invalide {value!r}, attendu {_type_label(hint)}")


def _from_table(section: str, table: Mapping[str, Any], cls: type[_T]) -> _T:
    """Construit la dataclass ``cls`` depuis une table TOML ; clés inconnues = avertissement."""
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    kwargs: dict[str, Any] = {}
    for key, value in table.items():
        if key in known:
            kwargs[key] = _coerce(section, key, value, hints.get(key, Any))
        else:
            logger.warning(
                "Clé inconnue ignorée dans la section [%s] : %r (clés connues : %s)",
                section,
                key,
                ", ".join(sorted(known)),
            )
    return cls(**kwargs)


# --- validation ------------------------------------------------------------------------


def _validate_general(general: GeneralSettings) -> None:
    mode = general.mode.strip().lower()
    if mode not in VALID_MODES:
        raise ValueError(
            f"[general] mode invalide : {general.mode!r}. Valeurs possibles : "
            + " ou ".join(f'"{m}"' for m in VALID_MODES)
            + ' ("paper" = argent fictif, "live" = argent réel).'
        )
    general.mode = mode

    timeframe = general.timeframe.strip()
    match = _TIMEFRAME_RE.match(timeframe)
    if match is None or int(match.group(1)) <= 0:
        raise ValueError(
            f"[general] timeframe invalide : {general.timeframe!r}. Format attendu : un "
            'nombre entier positif suivi de m, h, d ou w (ex. "15m", "1h", "4h", "1d").'
        )
    general.timeframe = timeframe

    general.exchange = general.exchange.strip().lower()
    if not general.exchange:
        raise ValueError('[general] exchange ne doit pas être vide (ex. "binance")')

    general.symbol = general.symbol.strip()
    if not general.symbol:
        raise ValueError('[general] symbol ne doit pas être vide (ex. "BTC/USDT")')
    if "/" not in general.symbol:
        logger.warning(
            "[general] symbol %r ne contient pas de « / » : ccxt attend le format BASE/QUOTE "
            "(ex. BTC/USDT)",
            general.symbol,
        )

    general.state_dir = general.state_dir.strip()
    if not general.state_dir:
        raise ValueError("[general] state_dir ne doit pas être vide")


def _validate_strategy(strategy: StrategySettings) -> None:
    strategy.name = strategy.name.strip()
    if not strategy.name:
        raise ValueError(
            "[strategy] name ne doit pas être vide (voir `tradingbot list-strategies`)"
        )


def _validate_fraction(section: str, key: str, value: float) -> None:
    if not (0.0 <= value < 1.0):
        raise ValueError(
            f"[{section}] {key} doit être une fraction entre 0 et 1 (0.001 = 0,1 %), pas {value!r}"
        )


def _validate_initial_cash(value: float) -> None:
    if not (value > 0) or value == float("inf"):
        raise ValueError(
            f"[risk] initial_cash doit être un montant strictement positif, pas {value!r}"
        )


def _secret(env: Mapping[str, str], name: str) -> str | None:
    """Valeur de la variable ``name`` ; ``None`` si absente ou vide."""
    value = env.get(name)
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _warn_consistency(settings: Settings) -> None:
    if settings.is_live and not settings.has_api_keys:
        logger.warning(
            'mode = "live" mais %s / %s ne sont pas renseignés dans .env : la commande '
            "`live` refusera de démarrer",
            ENV_API_KEY,
            ENV_API_SECRET,
        )
    if settings.notifications.telegram and not settings.has_telegram:
        logger.warning(
            "[notifications] telegram = true mais %s / %s ne sont pas renseignés dans .env : "
            "les notifications Telegram seront désactivées",
            ENV_TELEGRAM_TOKEN,
            ENV_TELEGRAM_CHAT_ID,
        )


# --- API publique ------------------------------------------------------------------------


def load_settings(
    path: str | Path = DEFAULT_CONFIG_PATH, env: Mapping[str, str] | None = None
) -> Settings:
    """Charge ``config.toml`` puis les secrets, et renvoie un :class:`Settings` validé.

    - fichier absent : défauts identiques à ``config.example.toml`` (journalisé en info) ;
    - sections ou clés inconnues : avertissement, jamais d'erreur ;
    - ``mode`` hors de ``{"paper", "live"}`` ou ``timeframe`` mal formé : ``ValueError`` ;
    - ``env`` : dictionnaire de variables d'environnement à utiliser tel quel (tests, CLI).
      S'il vaut ``None``, on lit ``.env`` (à côté du fichier de configuration, sinon dans le
      répertoire courant) puis ``os.environ``, ce dernier étant prioritaire.
    """
    config_path = Path(path)
    raw = _read_toml(config_path)
    for section in raw:
        if section not in _KNOWN_SECTIONS:
            logger.warning(
                "Section inconnue ignorée dans %s : [%s] (sections connues : %s)",
                config_path,
                section,
                ", ".join(_KNOWN_SECTIONS),
            )

    general = _from_table("general", _table(raw, "general"), GeneralSettings)
    _validate_general(general)

    strategy = _from_table("strategy", _table(raw, "strategy"), StrategySettings)
    _validate_strategy(strategy)

    risk_table = _table(raw, "risk")
    initial_cash = _coerce(
        "risk", "initial_cash", risk_table.pop("initial_cash", Settings.initial_cash), float
    )
    _validate_initial_cash(initial_cash)
    risk = _from_table("risk", risk_table, _risk_config_cls())

    costs = _from_table("costs", _table(raw, "costs"), CostSettings)
    _validate_fraction("costs", "fee_rate", costs.fee_rate)
    _validate_fraction("costs", "slippage", costs.slippage)

    notifications = _from_table("notifications", _table(raw, "notifications"), NotificationSettings)

    if env is None:
        env = {**load_dotenv(_find_dotenv(config_path)), **os.environ}

    settings = Settings(
        general=general,
        strategy=strategy,
        risk=risk,
        costs=costs,
        notifications=notifications,
        initial_cash=initial_cash,
        api_key=_secret(env, ENV_API_KEY),
        api_secret=_secret(env, ENV_API_SECRET),
        telegram_token=_secret(env, ENV_TELEGRAM_TOKEN),
        telegram_chat_id=_secret(env, ENV_TELEGRAM_CHAT_ID),
    )
    _warn_consistency(settings)
    logger.debug("Configuration effective : %r", settings)
    return settings


def parse_scalar(text: str) -> Any:
    """Convertit un texte de la ligne de commande en ``bool``, ``None``, ``int``, ``float``
    ou ``str`` (guillemets retirés) : ``"true"`` → ``True``, ``"10"`` → ``10``,
    ``"2.5"`` → ``2.5``, ``"none"`` → ``None``, ``"abc"`` → ``"abc"``."""
    value = text.strip()
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    if _INT_RE.match(value):
        return int(value)
    if _FLOAT_RE.match(value):
        return float(value)
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def parse_param_overrides(items: Iterable[str] | None) -> dict[str, Any]:
    """Transforme ``["fast=10", "slow=2.5", "x=true", "name=abc"]`` en
    ``{"fast": 10, "slow": 2.5, "x": True, "name": "abc"}`` (option ``--param`` de la CLI).

    Lève ``ValueError`` si un élément n'est pas de la forme ``nom=valeur``.
    """
    overrides: dict[str, Any] = {}
    for item in items or ():
        if "=" not in item:
            raise ValueError(
                f"Paramètre invalide {item!r} : format attendu nom=valeur (ex. fast=10)"
            )
        key, _, raw_value = item.partition("=")
        key = key.strip()
        if not key.isidentifier():
            raise ValueError(
                f"Nom de paramètre invalide {key!r} dans {item!r} : lettres, chiffres et "
                "tirets bas uniquement (ex. fast=10)"
            )
        overrides[key] = parse_scalar(raw_value)
    return overrides
