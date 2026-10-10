"""Boucle de trading temps réel (paper ou live).

Principe : une itération (:meth:`LiveTrader.step`) = vérifier la position courante contre
le prix du broker, puis récupérer les bougies **clôturées**, calculer le signal de la
*dernière* bougie clôturée, vérifier les garde-fous du
:class:`~tradingbot.risk.RiskManager`, et mettre le portefeuille en accord avec le signal
(« rebalance ») par des ordres au marché. Le signal décidé sur la bougie ``t`` est donc
exécuté **maintenant**, c'est-à-dire à l'ouverture de la bougie ``t+1`` : exactement la
règle du moteur de backtest, sans look-ahead.

Ordre d'une itération (le même que le moteur, bougie par bougie) :

1. prix courant et position du broker ; **stop / objectif** de la position mémorisée
   (``state_dir/position.json``) vérifiés à chaque appel, **avant** toute récupération
   de bougies : une panne de la source de données ne laisse jamais une position sans
   protection (actions ``"stop_loss"`` / ``"take_profit"``) ;
2. **kill switch** et **limite de perte journalière** évalués à chaque appel avec
   l'équité courante (pas seulement à la clôture d'une bougie) ; une position ouverte est
   fermée immédiatement (actions ``"kill_switch"`` / ``"daily_limit"``) ;
3. bougies clôturées ; idempotence : la dernière bougie traitée est mémorisée dans
   ``state_dir/last_bar.json`` et une page de bougies **plus ancienne** (source en retard)
   est ignorée comme une bougie déjà traitée (action ``"skip"``) ;
4. signal de la dernière bougie clôturée, puis rebalance.

Règles reprises du moteur de backtest :

- une position ne s'ouvre **jamais sans stop** : ``risk.protective_stop(signal, prix,
  stop de la stratégie, ATR(14) des bougies clôturées)`` choisit le stop (stratégie si du
  bon côté du prix, sinon ``default_stop_atr_mult × ATR``, sinon ``default_stop_pct``) et
  la taille vient de ``risk.position_size(équité, prix, stop)``, plafonnée par le cash
  disponible pour un achat ;
- ``reentry_after_stop = "wait"`` (défaut) : après une sortie décidée par le robot
  (stop, objectif, limite journalière) le côté sorti est **bloqué** (``blocked_side``,
  persisté dans ``position.json``) tant que le signal brut d'une bougie clôturée ultérieure
  ne l'a pas quitté (0 ou côté opposé) ; ``"immediate"`` = ré-ouverture dès que le signal
  le demande encore ;
- la limite de perte journalière n'a de sens qu'en intrajournalier : elle est désactivée
  (``RiskManager.set_daily_limit_active(False)``) quand la bougie fait un jour ou plus ;
- un solde de l'actif de base déjà présent sur le compte alors que le robot est plat
  n'est **pas** adopté (``adopt_existing_position=False``) : le robot le signale une fois,
  reste plat et refuse d'ouvrir (action ``"refused"``) tant que l'utilisateur ne lui
  confie pas explicitement cette position.

Actions possibles renvoyées par :meth:`LiveTrader.step` (clé ``"action"``) :

``open`` (entrée), ``close`` (sortie sur signal), ``reverse`` (sortie + entrée opposée),
``hold`` (rien à changer), ``skip`` (bougie déjà traitée, chauffe, taille trop petite…),
``stop_loss``, ``take_profit``, ``kill_switch``, ``daily_limit``, ``refused`` (refus du
gestionnaire de risque ou position existante non gérée), ``error`` (``BrokerError``
attrapée : la boucle survit ; un ordre rejeté ou partiellement exécuté est une erreur et
la position mémorisée reste protégée).

Un seul robot par répertoire d'état : :meth:`LiveTrader.step` et :meth:`LiveTrader.run`
prennent un verrou exclusif sur ``state_dir/lock`` (``fcntl.flock``, ou ``msvcrt`` sous
Windows), libéré à la fin de ``run`` / par :meth:`LiveTrader.close` ; un second robot
sur le même répertoire lève :class:`StateLockedError` (message en français).

Chaque action significative est journalisée et envoyée au ``notifier`` (une ligne, en
français). L'état du gestionnaire de risque est lui aussi persisté
(``state_dir/risk.json``) afin qu'un redémarrage ne remette pas le kill switch à zéro.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
import weakref
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from tradingbot import indicators as ind
from tradingbot.execution.broker import Broker, BrokerError, Order, Position
from tradingbot.live.notifier import Notifier
from tradingbot.risk import DEFAULT_STOP_ATR_WINDOW, RiskManager
from tradingbot.strategies.base import Strategy

try:  # normalisation canonique des bougies si le module data est disponible
    from tradingbot.data.loader import ensure_ohlcv as _ensure_ohlcv
except ImportError:  # pragma: no cover - dépend de l'ordre d'intégration des modules
    _ensure_ohlcv = None
try:
    from tradingbot.data.loader import timeframe_to_timedelta as _timeframe_to_timedelta
except ImportError:  # pragma: no cover
    _timeframe_to_timedelta = None
try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Windows
    _fcntl = None
try:
    import msvcrt as _msvcrt  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - POSIX
    _msvcrt = None

logger = logging.getLogger(__name__)

__all__ = [
    "LiveTrader",
    "PositionRecord",
    "StateDirLock",
    "StateLockedError",
    "default_poll_seconds",
    "normalise_ohlcv",
    "parse_timeframe",
    "stop_hit",
    "LAST_BAR_FILE",
    "LOCK_FILE",
    "POSITION_FILE",
    "RISK_FILE",
    "NOTIFIED_ACTIONS",
    "REENTRY_POLICIES",
]

LAST_BAR_FILE = "last_bar.json"
POSITION_FILE = "position.json"
RISK_FILE = "risk.json"
LOCK_FILE = "lock"
#: intervalle minimal entre deux interrogations du marché (secondes)
MIN_POLL_SECONDS = 10.0
#: nombre d'interrogations par bougie quand ``poll_seconds`` n'est pas fourni
POLLS_PER_CANDLE = 12
#: politiques de ré-entrée après une sortie décidée par le robot (cf. moteur de backtest)
REENTRY_POLICIES: tuple[str, ...] = ("wait", "immediate")
#: marge gardée sur le cash disponible lors d'un achat (frais + glissement d'un ordre au
#: marché), pour ne jamais envoyer un achat que le broker refuserait pour fonds insuffisants
CASH_SAFETY_MARGIN = 0.005
#: en deçà de cette fraction de la taille demandée, une exécution est signalée partielle
PARTIAL_FILL_TOLERANCE = 1e-6
#: actions qui déclenchent une notification (les autres sont seulement journalisées)
NOTIFIED_ACTIONS = frozenset(
    {
        "open",
        "close",
        "reverse",
        "stop_loss",
        "take_profit",
        "kill_switch",
        "daily_limit",
        "refused",
        "error",
    }
)
_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
# même grammaire que le loader : « M » (mois) est refusé, H/D/W tolérés en majuscules
_TIMEFRAME_RE = re.compile(r"^\s*(\d+)\s*([mhdwHDW])\s*$")
_TIMEFRAME_UNITS = {"m": "minutes", "h": "hours", "d": "days", "w": "weeks"}
_ONE_DAY = pd.Timedelta(days=1)


# ============================================================================ utilitaires
def _opt_float(value: Any) -> float | None:
    """``None`` si la valeur est absente, NaN ou infinie, sinon ``float``."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _to_utc(value: Any) -> pd.Timestamp:
    """Convertit un instant (datetime, Timestamp, str) en ``pd.Timestamp`` UTC.

    Un horodatage naïf est supposé déjà en UTC.
    """
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def parse_timeframe(timeframe: str) -> pd.Timedelta:
    """Durée d'une bougie (``"1h"`` → 1 heure). Lève ``ValueError`` si invalide."""
    if _timeframe_to_timedelta is not None:
        return _timeframe_to_timedelta(timeframe)
    if not isinstance(timeframe, str):
        raise TypeError(f"Le timeframe doit être une chaîne (ex. '1h'), reçu {timeframe!r}")
    match = _TIMEFRAME_RE.match(timeframe)
    if match is None or int(match.group(1)) <= 0:
        raise ValueError(
            f"Timeframe invalide : {timeframe!r} (attendu <entier><unité>, ex. 1m, 1h, 4h, 1d)."
        )
    return pd.Timedelta(**{_TIMEFRAME_UNITS[match.group(2).lower()]: int(match.group(1))})


def default_poll_seconds(timeframe: str | pd.Timedelta) -> float:
    """Intervalle d'interrogation par défaut : ``max(10 s, durée_bougie / 12)``."""
    delta = parse_timeframe(timeframe) if isinstance(timeframe, str) else timeframe
    return max(MIN_POLL_SECONDS, delta.total_seconds() / POLLS_PER_CANDLE)


def normalise_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise les bougies via ``tradingbot.data.loader.ensure_ohlcv`` si disponible,
    sinon applique des vérifications minimales (colonnes, index UTC trié, float64).

    Lève ``ValueError`` si le format est inexploitable.
    """
    if _ensure_ohlcv is not None:
        return _ensure_ohlcv(df)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Les bougies doivent être un pd.DataFrame, reçu {type(df).__name__}")
    rename = {c: str(c).strip().lower() for c in df.columns}
    out = df.rename(columns=rename)
    missing = [c for c in _OHLCV_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"Colonnes OHLCV manquantes : {missing}")
    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("L'index des bougies doit être un DatetimeIndex.")
    index = out.index.tz_localize("UTC") if out.index.tz is None else out.index.tz_convert("UTC")
    out = pd.DataFrame({c: out[c].to_numpy(dtype="float64") for c in _OHLCV_COLUMNS}, index=index)
    out = out.sort_index(kind="stable")
    out = out.loc[~out.index.duplicated(keep="last")]
    out.index.name = "timestamp"
    if len(out) == 0:
        raise ValueError("Le DataFrame de bougies est vide.")
    return out


def stop_hit(
    side: int, price: float, stop_loss: float | None, take_profit: float | None
) -> str | None:
    """``"stop_loss"`` / ``"take_profit"`` si ``price`` franchit le niveau actif, sinon ``None``.

    Le stop est testé avant l'objectif (hypothèse pessimiste, comme dans le backtest).
    """
    if side > 0:
        if stop_loss is not None and price <= stop_loss:
            return "stop_loss"
        if take_profit is not None and price >= take_profit:
            return "take_profit"
    elif side < 0:
        if stop_loss is not None and price >= stop_loss:
            return "stop_loss"
        if take_profit is not None and price <= take_profit:
            return "take_profit"
    return None


def _write_json(path: Path, data: Any) -> None:
    """Écriture atomique d'un fichier JSON (fichier temporaire puis ``os.replace``)."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        os.replace(tmp, path)
    except OSError as exc:
        raise BrokerError(f"Impossible d'écrire l'état du bot dans {path} : {exc}") from exc


def _read_json(path: Path) -> Any | None:
    """Contenu JSON de ``path`` ; ``None`` si le fichier n'existe pas ou est illisible."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Fichier d'état illisible, ignoré : %s (%s)", path, exc)
        return None


# ============================================================================ verrou
class StateLockedError(RuntimeError):
    """Le répertoire d'état est déjà utilisé par un autre robot (paper ou live)."""


def _lock_fd(fd: int) -> None:
    """Verrou exclusif non bloquant sur ``fd`` ; ``OSError`` s'il est déjà pris."""
    if _fcntl is not None:
        _fcntl.flock(fd, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
    elif _msvcrt is not None:  # pragma: no cover - Windows
        os.lseek(fd, 0, os.SEEK_SET)
        _msvcrt.locking(fd, _msvcrt.LK_NBLCK, 1)
    else:  # pragma: no cover - plateforme exotique
        logger.warning(
            "Aucun mécanisme de verrou disponible : le répertoire d'état n'est pas protégé"
        )


def _release_fd(fd: int) -> None:
    """Libère le verrou et ferme le descripteur (jamais d'exception)."""
    try:
        if _fcntl is not None:
            _fcntl.flock(fd, _fcntl.LOCK_UN)
        elif _msvcrt is not None:  # pragma: no cover - Windows
            os.lseek(fd, 0, os.SEEK_SET)
            _msvcrt.locking(fd, _msvcrt.LK_UNLCK, 1)
    except OSError:  # pragma: no cover - le fichier a pu disparaître
        pass
    try:
        os.close(fd)
    except OSError:  # pragma: no cover
        pass


class StateDirLock:
    """Verrou exclusif inter-processus sur un répertoire d'état (fichier ``lock``).

    ``acquire`` est idempotent pour le même objet ; il lève :class:`StateLockedError` si
    un autre processus (ou un autre :class:`LiveTrader` du même processus) tient déjà le
    verrou. Le PID du détenteur est écrit dans le fichier pour le message d'erreur. Le
    verrou est libéré par :meth:`release`, à la destruction de l'objet ou à la fin du
    processus.
    """

    def __init__(self, directory: str | Path, name: str = LOCK_FILE) -> None:
        self.path = Path(directory) / name
        self._fd: int | None = None
        self._finalizer: weakref.finalize | None = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def acquire(self) -> None:
        if self._fd is not None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        except OSError as exc:
            raise BrokerError(f"Impossible de créer le verrou {self.path} : {exc}") from exc
        try:
            _lock_fd(fd)
        except OSError as exc:
            owner = self._read_owner(fd)
            os.close(fd)
            raise StateLockedError(
                f"Le répertoire d'état {self.path.parent} est déjà utilisé par un autre robot"
                f"{f' (PID {owner})' if owner else ''} : une seule boucle paper/live par "
                "répertoire. Arrêtez l'autre processus ou choisissez un autre --state-dir."
            ) from exc
        try:
            os.ftruncate(fd, 0)
            os.write(fd, f"{os.getpid()}\n".encode())
        except OSError:  # pragma: no cover - informatif seulement
            pass
        self._fd = fd
        self._finalizer = weakref.finalize(self, _release_fd, fd)

    @staticmethod
    def _read_owner(fd: int) -> str:
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            return os.read(fd, 64).decode("utf-8", errors="replace").strip()
        except OSError:  # pragma: no cover
            return ""

    def release(self) -> None:
        if self._fd is None:
            return
        if self._finalizer is not None:
            self._finalizer.detach()
            self._finalizer = None
        fd, self._fd = self._fd, None
        _release_fd(fd)

    def __enter__(self) -> StateDirLock:
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()

    def __repr__(self) -> str:
        return f"StateDirLock(path={str(self.path)!r}, held={self.held})"


# ============================================================================ position
@dataclass
class PositionRecord:
    """Position suivie par le bot, avec son stop et son objectif actifs.

    Persistée dans ``state_dir/position.json``. ``side`` vaut +1 (long), -1 (short) ou 0
    (plat) ; ``opened_at`` est un horodatage ISO 8601 (UTC). ``amount`` est la quantité
    que le robot a lui-même achetée (jamais plus que le solde du compte).
    """

    side: int = 0
    amount: float = 0.0
    entry_price: float = 0.0
    stop_loss: float | None = None
    take_profit: float | None = None
    opened_at: str | None = None

    @property
    def is_open(self) -> bool:
        return self.side != 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Any) -> PositionRecord:
        """Reconstruit un enregistrement depuis un dict JSON (tolérant aux champs absents)."""
        if not isinstance(data, dict):
            return cls()
        side = int(data.get("side") or 0)
        if side not in (-1, 0, 1):
            raise ValueError(f"side invalide dans position.json : {side!r}")
        if side == 0:
            return cls()
        opened = data.get("opened_at")
        return cls(
            side=side,
            amount=abs(_opt_float(data.get("amount")) or 0.0),
            entry_price=_opt_float(data.get("entry_price")) or 0.0,
            stop_loss=_opt_float(data.get("stop_loss")),
            take_profit=_opt_float(data.get("take_profit")),
            opened_at=str(opened) if opened else None,
        )

    @classmethod
    def from_broker(cls, position: Position) -> PositionRecord:
        """Enregistrement construit depuis la position du broker (sans stop ni objectif)."""
        if not position.is_open:
            return cls()
        opened = position.opened_at
        return cls(
            side=position.side,
            amount=abs(position.amount),
            entry_price=float(position.entry_price or 0.0),
            opened_at=opened.isoformat() if isinstance(opened, datetime) else None,
        )


# ============================================================================ LiveTrader
class LiveTrader:
    """Boucle de trading temps réel : stratégie + broker + gestion du risque.

    Paramètres :
    - ``strategy`` : instance de :class:`~tradingbot.strategies.base.Strategy` ;
    - ``broker`` : :class:`~tradingbot.execution.broker.Broker` (paper ou ccxt) ;
    - ``symbol`` / ``timeframe`` : ex. ``"BTC/USDT"`` / ``"1h"`` ;
    - ``risk`` : :class:`~tradingbot.risk.RiskManager` ;
    - ``data_source`` : ``n_bars -> DataFrame`` de bougies **clôturées** (les plus
      récentes) ; toute bougie dont la clôture est dans le futur est de toute façon
      retirée par sécurité ;
    - ``state_dir`` : répertoire des fichiers d'état (créé au besoin) ;
    - ``lookback`` : nombre de bougies demandées à ``data_source`` ;
    - ``notifier`` : canal de notification (:mod:`tradingbot.live.notifier`) ou ``None`` ;
    - ``allow_short`` : si ``False``, un signal -1 est traité comme 0 ;
    - ``adopt_existing_position`` : si ``True``, un solde de l'actif de base déjà présent
      sur le compte alors que le robot est plat est adopté comme position du robot (sans
      stop jusqu'au prochain stop de la stratégie) ; par défaut (``False``) il n'est pas
      touché et le robot refuse d'ouvrir tant qu'il est là ;
    - ``reentry_after_stop`` : ``"wait"`` (défaut) ou ``"immediate"`` (voir le module) ;
    - ``lock_state_dir`` : prendre le verrou exclusif ``state_dir/lock`` (défaut ``True``) ;
    - ``sleep`` / ``clock`` : injectables pour les tests (``time.sleep`` et horloge UTC).

    L'état (dernière bougie traitée, position et stop actifs, côté bloqué, gestionnaire
    de risque) est rechargé depuis ``state_dir`` à la construction : un redémarrage
    reprend sans passer d'ordre en double.
    """

    def __init__(
        self,
        strategy: Strategy,
        broker: Broker,
        symbol: str,
        timeframe: str,
        risk: RiskManager,
        data_source: Callable[[int], pd.DataFrame],
        state_dir: str | Path = "state",
        lookback: int = 500,
        notifier: Notifier | None = None,
        allow_short: bool = False,
        sleep: Callable[[float], Any] = time.sleep,
        clock: Callable[[], datetime] | None = None,
        adopt_existing_position: bool = False,
        reentry_after_stop: str = "wait",
        lock_state_dir: bool = True,
    ) -> None:
        if not symbol or not str(symbol).strip():
            raise ValueError("Le symbole est vide (attendu par ex. 'BTC/USDT').")
        if not callable(data_source):
            raise TypeError("data_source doit être une fonction n_bars -> DataFrame.")
        if int(lookback) < 1:
            raise ValueError(f"lookback doit être >= 1 (reçu {lookback!r}).")
        if reentry_after_stop not in REENTRY_POLICIES:
            raise ValueError(
                f"reentry_after_stop={reentry_after_stop!r} invalide : valeurs possibles "
                + ", ".join(repr(p) for p in REENTRY_POLICIES)
            )
        self.strategy = strategy
        self.broker = broker
        self.symbol = str(symbol).strip()
        self.timeframe = timeframe
        self.timeframe_delta: pd.Timedelta = parse_timeframe(timeframe)
        self.risk = risk
        self.data_source = data_source
        self.state_dir = Path(state_dir)
        self.lookback = int(lookback)
        self.notifier = notifier
        self.allow_short = bool(allow_short)
        self.adopt_existing_position = bool(adopt_existing_position)
        self.reentry_after_stop = str(reentry_after_stop)
        self._sleep = sleep
        self._clock: Callable[[], datetime] = clock if clock is not None else Broker.now
        self._lock: StateDirLock | None = StateDirLock(self.state_dir) if lock_state_dir else None

        self.last_bar: pd.Timestamp | None = None
        self.last_signal: int | None = None
        self.position: PositionRecord = PositionRecord()
        #: côté bloqué après une sortie décidée par le robot (politique ``"wait"``)
        self.blocked_side: int = 0
        #: solde de l'actif de base présent sur le compte mais non géré par le robot
        self.unmanaged_amount: float = 0.0
        self._unmanaged_warned = False
        self.last_state: dict[str, Any] | None = None
        self._load_state()
        self._configure_daily_limit()

    # ------------------------------------------------------------------ propriétés
    @property
    def quote(self) -> str:
        """Devise de cotation déduite du symbole (``"BTC/USDT"`` → ``"USDT"``)."""
        if "/" in self.symbol:
            return self.symbol.split("/", 1)[1].split(":")[0]
        return ""

    @property
    def last_bar_path(self) -> Path:
        return self.state_dir / LAST_BAR_FILE

    @property
    def position_path(self) -> Path:
        return self.state_dir / POSITION_FILE

    @property
    def risk_path(self) -> Path:
        return self.state_dir / RISK_FILE

    @property
    def lock_path(self) -> Path:
        return self.state_dir / LOCK_FILE

    @property
    def daily_limit_active(self) -> bool:
        """False quand la limite journalière est désactivée (bougies >= 1 jour)."""
        return bool(getattr(self.risk, "daily_limit_active", True))

    @property
    def wait_after_exit(self) -> bool:
        return self.reentry_after_stop == "wait"

    # ------------------------------------------------------------------ persistance
    def _load_state(self) -> None:
        last = _read_json(self.last_bar_path)
        if isinstance(last, dict) and last.get("last_bar"):
            try:
                self.last_bar = _to_utc(last["last_bar"])
            except (ValueError, TypeError) as exc:
                logger.warning("Horodatage illisible dans %s : %s", self.last_bar_path, exc)
                self.last_bar = None
            sig = last.get("signal")
            self.last_signal = int(sig) if sig is not None else None
        pos = _read_json(self.position_path)
        if pos is not None:
            try:
                self.position = PositionRecord.from_dict(pos)
            except (ValueError, TypeError) as exc:
                logger.warning(
                    "Position mémorisée illisible dans %s (%s) : le stop actif est perdu, "
                    "la position sera reprise depuis le broker.",
                    self.position_path,
                    exc,
                )
                self.position = PositionRecord()
            if isinstance(pos, dict):
                blocked = pos.get("blocked_side")
                try:
                    blocked = int(blocked or 0)
                except (TypeError, ValueError):
                    blocked = 0
                self.blocked_side = blocked if blocked in (-1, 1) and self.wait_after_exit else 0
        risk_state = _read_json(self.risk_path)
        if isinstance(risk_state, dict):
            try:
                self.risk.load_state(risk_state)
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning("État du risque illisible dans %s : %s", self.risk_path, exc)
        if self.last_bar is not None or self.position.is_open:
            logger.info(
                "État rechargé depuis %s : dernière bougie %s, position %s%s",
                self.state_dir,
                self.last_bar.isoformat() if self.last_bar is not None else "aucune",
                self.position.side,
                f", côté bloqué {self.blocked_side:+d}" if self.blocked_side else "",
            )

    def _configure_daily_limit(self) -> None:
        """La limite de perte journalière n'a de sens qu'en intrajournalier."""
        active = self.timeframe_delta < _ONE_DAY
        setter = getattr(self.risk, "set_daily_limit_active", None)
        if callable(setter):
            setter(active, "" if active else "bougies >= 1 jour")
        cfg = getattr(self.risk, "config", None)
        if not active and getattr(cfg, "daily_loss_limit", None):
            logger.info(
                "limite de perte journalière désactivée : bougies >= 1 jour (timeframe %s)",
                self.timeframe,
            )

    def _persist(self) -> None:
        """Écrit last_bar.json, position.json et risk.json (atomiquement)."""
        _write_json(
            self.last_bar_path,
            {
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "last_bar": self.last_bar.isoformat() if self.last_bar is not None else None,
                "signal": self.last_signal,
                "updated_at": _to_utc(self._clock()).isoformat(),
            },
        )
        _write_json(
            self.position_path,
            {"symbol": self.symbol, **self.position.to_dict(), "blocked_side": self.blocked_side},
        )
        self._persist_risk()

    def _persist_risk(self) -> None:
        _write_json(self.risk_path, self.risk.state())

    # ------------------------------------------------------------------ verrou
    def _acquire_lock(self) -> None:
        if self._lock is not None:
            self._lock.acquire()

    def close(self) -> None:
        """Libère le verrou du répertoire d'état (à appeler quand le robot a fini)."""
        if self._lock is not None:
            self._lock.release()

    def __enter__(self) -> LiveTrader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ étape
    def step(self, now: datetime | pd.Timestamp | str | None = None) -> dict[str, Any]:
        """Une itération complète ; ne lève jamais ``BrokerError`` (action ``"error"``).

        ``now`` : instant courant (UTC par défaut), injectable pour les tests. Lève
        :class:`StateLockedError` si un autre robot utilise le répertoire d'état.
        """
        now_ts = _to_utc(now) if now is not None else _to_utc(self._clock())
        try:
            self._acquire_lock()  # StateLockedError (pas une BrokerError) remonte à l'appelant
            state = self._step(now_ts)
        except BrokerError as exc:
            state = self._error_state(now_ts, exc)
        self.last_state = state
        return state

    def _fetch_closed(self, now: pd.Timestamp) -> pd.DataFrame:
        """Bougies clôturées à ``now`` (normalisées). ``BrokerError`` si la source échoue."""
        try:
            raw = self.data_source(self.lookback)
        except BrokerError:
            raise
        except Exception as exc:
            raise BrokerError(
                f"Récupération des bougies impossible pour {self.symbol} : "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        try:
            df = normalise_ohlcv(raw)
        except (TypeError, ValueError) as exc:
            raise BrokerError(f"Bougies invalides reçues pour {self.symbol} : {exc}") from exc
        dropped = 0
        while len(df) > 0 and df.index[-1] + self.timeframe_delta > now:
            df = df.iloc[:-1]
            dropped += 1
        if dropped:
            logger.debug(
                "%d bougie(s) non clôturée(s) ignorée(s) (dernière clôture > %s)",
                dropped,
                now.isoformat(),
            )
        return df

    def _sync_position(self, broker_position: Position) -> None:
        """Met l'enregistrement local en accord avec la position réelle du broker.

        Le robot ne possède jamais plus que le solde du compte, mais il peut posséder
        moins (solde acheté à la main en plus) : sa propre quantité est conservée. Un
        solde présent alors que le robot est plat n'est adopté que si
        ``adopt_existing_position`` est vrai ; sinon il est mémorisé comme « non géré ».
        """
        rec = self.position
        self.unmanaged_amount = 0.0
        if broker_position.is_open:
            held = abs(broker_position.amount)
            if rec.side == broker_position.side:
                if rec.amount <= 0 or rec.amount > held:
                    rec.amount = held
                if broker_position.entry_price:
                    rec.entry_price = float(broker_position.entry_price)
            elif rec.side == 0 and not self.adopt_existing_position:
                self.unmanaged_amount = held
                if not self._unmanaged_warned:
                    self._unmanaged_warned = True
                    logger.warning(
                        "Le compte détient déjà %.8g %s alors que le robot est plat : position "
                        "NON gérée par le robot (ni vendue, ni protégée). Aucune ouverture tant "
                        "qu'elle est là ; relancez avec --adopt-position pour la lui confier.",
                        held,
                        self.symbol.split("/")[0],
                    )
            else:
                logger.warning(
                    "Position du broker (%s %.8g) différente de l'état mémorisé (side %d) : "
                    "position du broker adoptée, sans stop ni objectif.",
                    self.symbol,
                    held,
                    rec.side,
                )
                self.position = PositionRecord.from_broker(broker_position)
        else:
            self._unmanaged_warned = False
            if rec.is_open:
                logger.warning(
                    "Position mémorisée (side %d) mais le broker est plat sur %s : état remis à zéro.",
                    rec.side,
                    self.symbol,
                )
                self.position = PositionRecord()

    def _close_managed(self) -> Order | None:
        """Ferme la position du robot (sa propre quantité, jamais plus que le compte).

        Un ordre rejeté lève ``BrokerError`` (le broker) ; une exécution partielle met à
        jour la quantité restante, la persiste et lève ``BrokerError`` : l'enregistrement
        (et son stop) survit, la clôture sera retentée à l'interrogation suivante.
        """
        rec = self.position
        broker_position = self.broker.get_position(self.symbol)
        held = abs(broker_position.amount) if broker_position.side == rec.side else 0.0
        amount = min(rec.amount, held) if rec.amount > 0 else held
        if amount <= 0:
            self.position = PositionRecord()
            return None
        side = "sell" if rec.side > 0 else "buy"
        order = self.broker.market_order(self.symbol, side, amount)
        executed = float(order.amount)
        if executed < amount * (1.0 - PARTIAL_FILL_TOLERANCE):
            rec.amount = max(amount - executed, 0.0)
            self._persist()
            raise BrokerError(
                f"Clôture incomplète de la position {self.symbol} : {executed:.8g} exécuté(s) "
                f"sur {amount:.8g} (ordre {order.id}, statut {order.status}) ; il reste "
                f"{rec.amount:.8g}, nouvel essai à la prochaine interrogation."
            )
        self.position = PositionRecord()
        return order

    def _block_side(self, side: int) -> None:
        """Mémorise le côté sorti sur décision du robot (politique ``"wait"``)."""
        if self.wait_after_exit and side in (-1, 1):
            self.blocked_side = side

    def _guard_block(self, now: pd.Timestamp, equity: float) -> tuple[str, str] | None:
        if self.risk.kill_switch_active():
            return "kill_switch", "kill switch actif : drawdown maximal dépassé"
        if self.risk.daily_limit_hit(now, equity):
            return "daily_limit", "limite de perte journalière atteinte"
        return None

    def _protective_stop(
        self, side: int, price: float, strategy_stop: float | None, df: pd.DataFrame
    ) -> float:
        """Stop d'une entrée : stratégie si du bon côté, sinon ATR(14) des bougies
        clôturées, sinon pourcentage de dernier recours (``RiskManager.protective_stop``)."""
        atr_value: float | None = None
        if len(df) > DEFAULT_STOP_ATR_WINDOW:
            try:
                atr_value = _opt_float(ind.atr(df, DEFAULT_STOP_ATR_WINDOW).iloc[-1])
            except (ValueError, TypeError, KeyError) as exc:
                logger.debug("ATR indisponible pour le stop de secours : %s", exc)
        return float(self.risk.protective_stop(side, price, strategy_stop, atr_value))

    def _step(self, now: pd.Timestamp) -> dict[str, Any]:
        price = self.broker.get_price(self.symbol)
        self._sync_position(self.broker.get_position(self.symbol))

        # 1) stop / objectif de la position courante, à chaque interrogation, AVANT les
        #    bougies : une panne de la source de données ne désarme jamais le stop.
        if self.position.is_open:
            hit = stop_hit(
                self.position.side, price, self.position.stop_loss, self.position.take_profit
            )
            if hit is not None:
                side = self.position.side
                level = self.position.stop_loss if hit == "stop_loss" else self.position.take_profit
                order = self._close_managed()
                self._block_side(side)
                self._persist()
                label = "Stop touché" if hit == "stop_loss" else "Objectif atteint"
                reason = f"{label} : prix {price:.6g} contre niveau {level:.6g}"
                return self._finish(now, self.last_bar, self.last_signal, hit, reason, price, order)

        # 2) garde-fous à chaque interrogation (pas seulement à la clôture d'une bougie)
        balance = self.broker.get_balance(self.symbol)
        equity = balance.equity
        self.risk.update(now, equity)
        guard = self._guard_block(now, equity)
        if guard is not None:
            action, why = guard
            if self.position.is_open:
                side = self.position.side
                order = self._close_managed()
                if action == "daily_limit":
                    self._block_side(side)
                self._persist()
                reason = f"{why} : position clôturée, aucune ouverture autorisée"
                return self._finish(
                    now, self.last_bar, self.last_signal, action, reason, price, order
                )
            self._persist_risk()

        # 3) bougies clôturées (dans leur propre bloc : une panne ici est une "error",
        #    mais le stop et les garde-fous ont déjà été évalués)
        df = self._fetch_closed(now)
        if len(df) == 0:
            return self._finish(
                now, None, None, "skip", "aucune bougie clôturée disponible", price=None
            )
        last_bar = df.index[-1]

        # 4) idempotence : une bougie n'est traitée qu'une fois ; une page plus ancienne
        #    (source de données en retard) ne fait jamais reculer last_bar
        if self.last_bar is not None and last_bar <= self.last_bar:
            if last_bar < self.last_bar:
                logger.warning(
                    "Source de données en retard : dernière bougie reçue %s antérieure à la "
                    "dernière traitée %s ; page ignorée.",
                    last_bar.isoformat(),
                    self.last_bar.isoformat(),
                )
                reason = "source de données en retard : bougie antérieure à la dernière traitée"
                shown = self.last_bar
            else:
                reason = "bougie déjà traitée"
                shown = last_bar
            return self._finish(now, shown, self.last_signal, "skip", reason, price, equity=equity)

        # 5) signal de la dernière bougie clôturée
        warmup = int(self.strategy.warmup_bars())
        if len(df) <= warmup:
            self.last_bar = last_bar
            self.last_signal = None
            self._persist()
            return self._finish(
                now,
                last_bar,
                None,
                "skip",
                f"période de chauffe : {len(df)} bougie(s) disponible(s), {warmup + 1} requise(s)",
                price,
                equity=equity,
            )
        signals = Strategy.validate_signals(df, self.strategy.generate_signals(df))
        raw_signal = int(signals["signal"].iloc[-1])
        stop_loss = _opt_float(signals["stop_loss"].iloc[-1])
        take_profit = _opt_float(signals["take_profit"].iloc[-1])
        signal = raw_signal
        notes: list[str] = []
        if signal < 0 and not self.allow_short:
            signal = 0
            notes.append("signal short ignoré (allow_short = false)")
        # le signal brut a quitté le côté bloqué : la ré-entrée redevient possible
        if self.blocked_side != 0 and raw_signal != self.blocked_side:
            self.blocked_side = 0

        # 6) garde-fou actif et position plate : aucune ouverture
        if guard is not None:
            action, why = guard
            if signal != 0:
                reason = f"{why} : ouverture refusée"
            else:
                action = "hold"
                reason = f"aucune position ; {why}"
            self.last_bar = last_bar
            self.last_signal = signal
            self._persist()
            return self._finish(now, last_bar, signal, action, reason, price, equity=equity)

        # 7) politique de ré-entrée "wait" : le côté sorti sur stop / objectif / limite
        #    journalière reste bloqué tant que le signal ne l'a pas quitté une fois
        if self.blocked_side != 0 and signal == self.blocked_side:
            signal = 0
            notes.append(
                f"ré-entrée {'longue' if self.blocked_side > 0 else 'courte'} bloquée après une "
                "sortie forcée : en attente d'un nouveau signal (reentry_after_stop = wait)"
            )

        # 8) rebalance
        current = self.position.side
        closed_order: Order | None = None
        opened_order: Order | None = None
        action = "hold"
        if signal == current:
            if current != 0:
                reason = "position conservée"
                if stop_loss is not None and stop_loss != self.position.stop_loss:
                    if stop_hit(current, price, stop_loss, None):
                        logger.debug(
                            "Stop de la stratégie %.6g du mauvais côté du prix %.6g : ignoré",
                            stop_loss,
                            price,
                        )
                        notes.append(
                            f"stop de la stratégie {stop_loss:.6g} ignoré (mauvais côté du prix)"
                        )
                    else:
                        self.position.stop_loss = stop_loss
                        notes.append(f"stop mis à jour à {stop_loss:.6g}")
                if take_profit is not None and take_profit != self.position.take_profit:
                    self.position.take_profit = take_profit
                    notes.append(f"objectif mis à jour à {take_profit:.6g}")
            else:
                reason = "aucun signal, aucune position"
        else:
            reason = ""
            if current != 0:
                closed_order = self._close_managed()
                self._persist()
                action = "close"
                reason = "inversion du signal" if signal != 0 else "signal de sortie"
            if signal != 0:
                if self.unmanaged_amount > 0:
                    if closed_order is None:
                        action = "refused"
                    reason = _join(
                        reason,
                        "ouverture refusée : position existante non gérée par le robot "
                        f"({self.unmanaged_amount:.8g} {self.symbol.split('/')[0]} sur le compte) ; "
                        "relancez avec --adopt-position pour la lui confier",
                    )
                else:
                    balance = self.broker.get_balance(self.symbol)
                    equity = balance.equity
                    ok, why = self.risk.can_open(signal, now, equity)
                    if not ok:
                        if closed_order is None:
                            action = "refused"
                        reason = _join(reason, f"ouverture refusée : {why}")
                    else:
                        stop_price = self._protective_stop(signal, price, stop_loss, df)
                        if stop_loss is None:
                            notes.append(f"stop de secours à {stop_price:.6g}")
                        elif stop_price != stop_loss:
                            notes.append(
                                f"stop de la stratégie {stop_loss:.6g} du mauvais côté du prix : "
                                f"stop de secours à {stop_price:.6g}"
                            )
                        size = self.risk.position_size(equity, price, stop_price=stop_price)
                        if signal > 0 and size > 0:
                            affordable = balance.cash * (1.0 - CASH_SAFETY_MARGIN) / price
                            if size > affordable:
                                size = max(affordable, 0.0)
                                notes.append("taille plafonnée par le cash disponible")
                        if size <= 0:
                            if closed_order is None:
                                action = "skip"
                            reason = _join(reason, "taille trop petite")
                        else:
                            side = "buy" if signal > 0 else "sell"
                            opened_order = self.broker.market_order(self.symbol, side, size)
                            executed = float(opened_order.amount)
                            if executed < size * (1.0 - PARTIAL_FILL_TOLERANCE):
                                logger.warning(
                                    "Entrée %s exécutée partiellement : %.8g sur %.8g demandés "
                                    "(ordre %s, statut %s)",
                                    self.symbol,
                                    executed,
                                    size,
                                    opened_order.id,
                                    opened_order.status,
                                )
                                notes.append(f"exécution partielle : {executed:.8g} sur {size:.8g}")
                            self.position = PositionRecord(
                                side=signal,
                                amount=executed,
                                entry_price=float(opened_order.price),
                                stop_loss=stop_price,
                                take_profit=take_profit,
                                opened_at=now.isoformat(),
                            )
                            action = "reverse" if closed_order is not None else "open"
                            reason = _join(reason, "entrée sur signal")

        self.last_bar = last_bar
        self.last_signal = signal
        self._persist()
        if notes:
            reason = _join(reason, " ; ".join(notes))
        order = opened_order if opened_order is not None else closed_order
        return self._finish(now, last_bar, signal, action, reason, price, order)

    # ------------------------------------------------------------------ compte rendu
    def _equity_or_none(self) -> float | None:
        try:
            return float(self.broker.get_balance(self.symbol).equity)
        except BrokerError as exc:
            logger.warning("Équité indisponible : %s", exc)
            return None

    def _finish(
        self,
        now: pd.Timestamp,
        last_bar: pd.Timestamp | None,
        signal: int | None,
        action: str,
        reason: str,
        price: float | None,
        order: Order | None = None,
        equity: float | None = None,
    ) -> dict[str, Any]:
        state: dict[str, Any] = {
            "timestamp": last_bar.isoformat() if last_bar is not None else None,
            "now": now.isoformat(),
            "symbol": self.symbol,
            "signal": signal,
            "action": action,
            "position": self.position.side,
            "amount": self.position.amount,
            "entry_price": self.position.entry_price if self.position.is_open else None,
            "stop_loss": self.position.stop_loss,
            "take_profit": self.position.take_profit,
            "price": price,
            "equity": float(equity) if equity is not None else self._equity_or_none(),
            "reason": reason,
            "order": order.id if order is not None else None,
        }
        self._report(state, order)
        return state

    def _error_state(self, now: pd.Timestamp, exc: BrokerError) -> dict[str, Any]:
        logger.error("Erreur broker pendant l'itération : %s", exc)
        state: dict[str, Any] = {
            "timestamp": self.last_bar.isoformat() if self.last_bar is not None else None,
            "now": now.isoformat(),
            "symbol": self.symbol,
            "signal": self.last_signal,
            "action": "error",
            "position": self.position.side,
            "amount": self.position.amount,
            "entry_price": self.position.entry_price if self.position.is_open else None,
            "stop_loss": self.position.stop_loss,
            "take_profit": self.position.take_profit,
            "price": None,
            "equity": None,
            "reason": str(exc),
            "order": None,
        }
        self._notify(f"[{self.symbol}] ERREUR broker : {exc}")
        return state

    def _message(self, state: dict[str, Any], order: Order | None) -> str:
        sym = self.symbol
        action = state["action"]
        quote = self.quote
        equity = state.get("equity")
        equity_txt = f" — équité {equity:.2f} {quote}".rstrip() if equity is not None else ""
        if order is not None:
            verb = "ACHAT" if order.side == "buy" else "VENTE"
            fill = f"{verb} {order.amount:.8g} @ {order.price:.6g} (frais {order.fee:.4f})"
        else:
            fill = ""
        if action == "open":
            stop = state.get("stop_loss")
            tp = state.get("take_profit")
            levels = [f"stop {stop:.6g}"] if stop is not None else []
            if tp is not None:
                levels.append(f"objectif {tp:.6g}")
            lv = f" [{', '.join(levels)}]" if levels else ""
            return f"[{sym}] Entrée {'LONG' if state['position'] > 0 else 'SHORT'} : {fill}{lv}{equity_txt}"
        if action == "reverse":
            return f"[{sym}] Inversion de position : {fill}{equity_txt}"
        if action == "close":
            return f"[{sym}] Sortie sur signal : {fill}{equity_txt}"
        if action in ("stop_loss", "take_profit"):
            return f"[{sym}] {state['reason']} → position clôturée : {fill}{equity_txt}"
        if action == "kill_switch":
            return f"[{sym}] KILL SWITCH — {state['reason']}{(' : ' + fill) if fill else ''}{equity_txt}"
        if action == "daily_limit":
            return f"[{sym}] LIMITE JOURNALIÈRE — {state['reason']}{(' : ' + fill) if fill else ''}{equity_txt}"
        if action == "refused":
            return f"[{sym}] Refus du gestionnaire de risque : {state['reason']}{equity_txt}"
        if action == "error":
            return f"[{sym}] ERREUR broker : {state['reason']}"
        return f"[{sym}] {action} : {state['reason']}{equity_txt}"

    def _report(self, state: dict[str, Any], order: Order | None) -> None:
        text = self._message(state, order)
        action = state["action"]
        if action in NOTIFIED_ACTIONS:
            logger.info("%s", text)
            self._notify(text)
        elif action == "skip" and state["reason"] == "bougie déjà traitée":
            logger.debug("%s", text)
        else:
            logger.info("%s", text)

    def _notify(self, text: str) -> None:
        if self.notifier is None:
            return
        try:
            ok = self.notifier.send(text)
        except Exception as exc:  # le contrat dit « jamais », on se protège quand même
            logger.warning("Le notificateur a levé une exception : %s", exc)
            return
        if not ok:
            logger.warning("Notification non envoyée : %s", text)

    # ------------------------------------------------------------------ boucle
    def run(self, poll_seconds: float | None = None, max_iterations: int | None = None) -> None:
        """Boucle : :meth:`step` puis attente ``poll_seconds`` (défaut : durée de bougie / 12,
        au moins 10 s). S'arrête après ``max_iterations`` itérations ou dès que le kill
        switch est actif. Sur Ctrl-C, s'arrête proprement (journal, position mémorisée,
        verrou libéré) puis **relance** ``KeyboardInterrupt`` pour que l'appelant sache que
        l'arrêt a été demandé (la CLI renvoie alors le code 130).

        Lève :class:`StateLockedError` si un autre robot utilise déjà ``state_dir``.
        """
        if poll_seconds is None:
            poll_seconds = default_poll_seconds(self.timeframe_delta)
        if not math.isfinite(poll_seconds) or poll_seconds <= 0:
            raise ValueError(f"poll_seconds doit être > 0 (reçu {poll_seconds!r}).")
        if max_iterations is not None and max_iterations < 0:
            raise ValueError(f"max_iterations doit être >= 0 (reçu {max_iterations!r}).")
        self._acquire_lock()
        logger.info(
            "Démarrage de la boucle %s %s : interrogation toutes les %.0f s%s",
            self.symbol,
            self.timeframe,
            poll_seconds,
            f", {max_iterations} itération(s) max" if max_iterations is not None else "",
        )
        iterations = 0
        interrupted = False
        try:
            while max_iterations is None or iterations < max_iterations:
                try:
                    self.step()
                except Exception as exc:
                    logger.exception("Erreur inattendue pendant l'itération %d", iterations + 1)
                    self._notify(f"[{self.symbol}] ERREUR inattendue : {type(exc).__name__}: {exc}")
                iterations += 1
                if self.risk.kill_switch_active():
                    msg = (
                        f"[{self.symbol}] Kill switch actif : arrêt de la boucle. "
                        "Relancez après analyse (voir state/risk.json)."
                    )
                    logger.warning("%s", msg)
                    self._notify(msg)
                    break
                if max_iterations is not None and iterations >= max_iterations:
                    break
                self._sleep(poll_seconds)
        except KeyboardInterrupt:
            interrupted = True
            logger.info("Arrêt demandé (Ctrl-C) après %d itération(s).", iterations)
            if self.position.is_open:
                logger.warning(
                    "Position %s toujours ouverte (side %d, stop %s) : elle est mémorisée "
                    "dans %s et sera reprise au prochain démarrage.",
                    self.symbol,
                    self.position.side,
                    self.position.stop_loss,
                    self.state_dir,
                )
        finally:
            self.close()
        logger.info("Boucle terminée après %d itération(s).", iterations)
        if interrupted:
            raise KeyboardInterrupt

    def __repr__(self) -> str:
        return (
            f"LiveTrader(symbol={self.symbol!r}, timeframe={self.timeframe!r}, "
            f"strategy={self.strategy!r}, position={self.position.side}, "
            f"last_bar={self.last_bar.isoformat() if self.last_bar is not None else None})"
        )


def _join(first: str, second: str) -> str:
    return f"{first} ; {second}" if first else second
