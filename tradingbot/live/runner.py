"""Boucle de trading temps réel (paper ou live).

Principe : une itération (:meth:`LiveTrader.step`) = récupérer les bougies **clôturées**,
calculer le signal de la *dernière* bougie clôturée, vérifier les garde-fous du
:class:`~tradingbot.risk.RiskManager`, puis mettre le portefeuille en accord avec le
signal (« rebalance ») par des ordres au marché. Le signal décidé sur la bougie ``t`` est
donc exécuté **maintenant**, c'est-à-dire à l'ouverture de la bougie ``t+1`` : exactement
la règle du moteur de backtest, sans look-ahead.

Idempotence : la dernière bougie traitée est mémorisée dans ``state_dir/last_bar.json`` ;
tant qu'aucune nouvelle bougie n'est clôturée, :meth:`LiveTrader.step` ne passe aucun
ordre de signal (action ``"skip"``). En revanche, le **stop** et l'**objectif** de la
position courante (mémorisés dans ``state_dir/position.json``) sont vérifiés à chaque
appel contre le prix courant du broker : s'ils sont franchis, la position est clôturée
immédiatement (actions ``"stop_loss"`` / ``"take_profit"``).

Actions possibles renvoyées par :meth:`LiveTrader.step` (clé ``"action"``) :

``open`` (entrée), ``close`` (sortie sur signal), ``reverse`` (sortie + entrée opposée),
``hold`` (rien à changer), ``skip`` (bougie déjà traitée, chauffe, taille trop petite…),
``stop_loss``, ``take_profit``, ``kill_switch``, ``daily_limit``, ``refused`` (refus du
gestionnaire de risque), ``error`` (``BrokerError`` attrapée : la boucle survit).

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
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from tradingbot.execution.broker import Broker, BrokerError, Order, Position
from tradingbot.live.notifier import Notifier
from tradingbot.risk import RiskManager
from tradingbot.strategies.base import Strategy

try:  # normalisation canonique des bougies si le module data est disponible
    from tradingbot.data.loader import ensure_ohlcv as _ensure_ohlcv
except ImportError:  # pragma: no cover - dépend de l'ordre d'intégration des modules
    _ensure_ohlcv = None
try:
    from tradingbot.data.loader import timeframe_to_timedelta as _timeframe_to_timedelta
except ImportError:  # pragma: no cover
    _timeframe_to_timedelta = None

logger = logging.getLogger(__name__)

__all__ = [
    "LiveTrader",
    "PositionRecord",
    "default_poll_seconds",
    "normalise_ohlcv",
    "parse_timeframe",
    "stop_hit",
    "LAST_BAR_FILE",
    "POSITION_FILE",
    "RISK_FILE",
    "NOTIFIED_ACTIONS",
]

LAST_BAR_FILE = "last_bar.json"
POSITION_FILE = "position.json"
RISK_FILE = "risk.json"
#: intervalle minimal entre deux interrogations du marché (secondes)
MIN_POLL_SECONDS = 10.0
#: nombre d'interrogations par bougie quand ``poll_seconds`` n'est pas fourni
POLLS_PER_CANDLE = 12
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


# ============================================================================ position
@dataclass
class PositionRecord:
    """Position suivie par le bot, avec son stop et son objectif actifs.

    Persistée dans ``state_dir/position.json``. ``side`` vaut +1 (long), -1 (short) ou 0
    (plat) ; ``opened_at`` est un horodatage ISO 8601 (UTC).
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
    - ``sleep`` / ``clock`` : injectables pour les tests (``time.sleep`` et horloge UTC).

    L'état (dernière bougie traitée, position et stop actifs, gestionnaire de risque)
    est rechargé depuis ``state_dir`` à la construction : un redémarrage reprend sans
    passer d'ordre en double.
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
    ) -> None:
        if not symbol or not str(symbol).strip():
            raise ValueError("Le symbole est vide (attendu par ex. 'BTC/USDT').")
        if not callable(data_source):
            raise TypeError("data_source doit être une fonction n_bars -> DataFrame.")
        if int(lookback) < 1:
            raise ValueError(f"lookback doit être >= 1 (reçu {lookback!r}).")
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
        self._sleep = sleep
        self._clock: Callable[[], datetime] = clock if clock is not None else Broker.now

        self.last_bar: pd.Timestamp | None = None
        self.last_signal: int | None = None
        self.position: PositionRecord = PositionRecord()
        self.last_state: dict[str, Any] | None = None
        self._load_state()

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
        risk_state = _read_json(self.risk_path)
        if isinstance(risk_state, dict):
            try:
                self.risk.load_state(risk_state)
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning("État du risque illisible dans %s : %s", self.risk_path, exc)
        if self.last_bar is not None or self.position.is_open:
            logger.info(
                "État rechargé depuis %s : dernière bougie %s, position %s",
                self.state_dir,
                self.last_bar.isoformat() if self.last_bar is not None else "aucune",
                self.position.side,
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
        _write_json(self.position_path, {"symbol": self.symbol, **self.position.to_dict()})
        _write_json(self.risk_path, self.risk.state())

    # ------------------------------------------------------------------ étape
    def step(self, now: datetime | pd.Timestamp | str | None = None) -> dict[str, Any]:
        """Une itération complète ; ne lève jamais ``BrokerError`` (action ``"error"``).

        ``now`` : instant courant (UTC par défaut), injectable pour les tests.
        """
        now_ts = _to_utc(now) if now is not None else _to_utc(self._clock())
        try:
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
        """Met l'enregistrement local en accord avec la position réelle du broker."""
        rec = self.position
        if broker_position.is_open:
            if rec.side != broker_position.side:
                logger.warning(
                    "Position du broker (%s %.8g) différente de l'état mémorisé (side %d) : "
                    "position du broker adoptée, sans stop ni objectif.",
                    self.symbol,
                    broker_position.amount,
                    rec.side,
                )
                self.position = PositionRecord.from_broker(broker_position)
            else:
                rec.amount = abs(broker_position.amount)
                if broker_position.entry_price:
                    rec.entry_price = float(broker_position.entry_price)
        elif rec.is_open:
            logger.warning(
                "Position mémorisée (side %d) mais le broker est plat sur %s : état remis à zéro.",
                rec.side,
                self.symbol,
            )
            self.position = PositionRecord()

    def _step(self, now: pd.Timestamp) -> dict[str, Any]:
        df = self._fetch_closed(now)
        if len(df) == 0:
            return self._finish(
                now, None, None, "skip", "aucune bougie clôturée disponible", price=None
            )
        last_bar = df.index[-1]
        price = self.broker.get_price(self.symbol)
        self._sync_position(self.broker.get_position(self.symbol))

        # 1) stop / objectif de la position courante, vérifiés à chaque interrogation
        if self.position.is_open:
            hit = stop_hit(
                self.position.side, price, self.position.stop_loss, self.position.take_profit
            )
            if hit is not None:
                level = self.position.stop_loss if hit == "stop_loss" else self.position.take_profit
                order = self.broker.close_position(self.symbol)
                self.position = PositionRecord()
                # pas de ré-entrée sur la même bougie (règle du backtest)
                self.last_bar = last_bar
                self._persist()
                label = "Stop touché" if hit == "stop_loss" else "Objectif atteint"
                reason = f"{label} : prix {price:.6g} contre niveau {level:.6g}"
                return self._finish(now, last_bar, self.last_signal, hit, reason, price, order)

        # 2) idempotence : une bougie n'est traitée qu'une fois
        if self.last_bar is not None and last_bar == self.last_bar:
            return self._finish(
                now, last_bar, self.last_signal, "skip", "bougie déjà traitée", price
            )

        # 3) signal de la dernière bougie clôturée
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

        # 4) garde-fous
        equity = self.broker.get_balance(self.symbol).equity
        self.risk.update(now, equity)
        blocked: tuple[str, str] | None = None
        if self.risk.kill_switch_active():
            blocked = ("kill_switch", "kill switch actif : drawdown maximal dépassé")
        elif self.risk.daily_limit_hit(now, equity):
            blocked = ("daily_limit", "limite de perte journalière atteinte")
        if blocked is not None:
            action, why = blocked
            order: Order | None = None
            if self.position.is_open:
                order = self.broker.close_position(self.symbol)
                self.position = PositionRecord()
                reason = f"{why} : position clôturée, aucune ouverture autorisée"
            elif signal != 0:
                reason = f"{why} : ouverture refusée"
            else:
                action = "hold"
                reason = f"aucune position ; {why}"
            self.last_bar = last_bar
            self.last_signal = signal
            self._persist()
            return self._finish(now, last_bar, signal, action, reason, price, order)

        # 5) rebalance
        current = self.position.side
        closed_order: Order | None = None
        opened_order: Order | None = None
        action = "hold"
        if signal == current:
            if current != 0:
                reason = "position conservée"
                if stop_loss is not None and stop_loss != self.position.stop_loss:
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
                closed_order = self.broker.close_position(self.symbol)
                self.position = PositionRecord()
                self._persist()
                action = "close"
                reason = "inversion du signal" if signal != 0 else "signal de sortie"
            if signal != 0:
                equity = self.broker.get_balance(self.symbol).equity
                ok, why = self.risk.can_open(signal, now, equity)
                if not ok:
                    if closed_order is None:
                        action = "refused"
                    reason = _join(reason, f"ouverture refusée : {why}")
                else:
                    if stop_loss is not None and stop_hit(signal, price, stop_loss, None):
                        logger.warning(
                            "Stop %.6g du mauvais côté du prix %.6g pour un signal %+d : "
                            "il sera déclenché dès la prochaine vérification.",
                            stop_loss,
                            price,
                            signal,
                        )
                    size = self.risk.position_size(equity, price, stop_price=stop_loss)
                    if size <= 0:
                        if closed_order is None:
                            action = "skip"
                        reason = _join(reason, "taille trop petite")
                    else:
                        side = "buy" if signal > 0 else "sell"
                        opened_order = self.broker.market_order(self.symbol, side, size)
                        self.position = PositionRecord(
                            side=signal,
                            amount=float(opened_order.amount),
                            entry_price=float(opened_order.price),
                            stop_loss=stop_loss,
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
            "equity": self._equity_or_none(),
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
            levels = []
            if stop is not None:
                levels.append(f"stop {stop:.6g}")
            if tp is not None:
                levels.append(f"objectif {tp:.6g}")
            lv = f" [{', '.join(levels)}]" if levels else " [sans stop]"
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
        au moins 10 s). S'arrête proprement sur Ctrl-C, après ``max_iterations`` itérations,
        ou dès que le kill switch est actif.
        """
        if poll_seconds is None:
            poll_seconds = default_poll_seconds(self.timeframe_delta)
        if not math.isfinite(poll_seconds) or poll_seconds <= 0:
            raise ValueError(f"poll_seconds doit être > 0 (reçu {poll_seconds!r}).")
        if max_iterations is not None and max_iterations < 0:
            raise ValueError(f"max_iterations doit être >= 0 (reçu {max_iterations!r}).")
        logger.info(
            "Démarrage de la boucle %s %s : interrogation toutes les %.0f s%s",
            self.symbol,
            self.timeframe,
            poll_seconds,
            f", {max_iterations} itération(s) max" if max_iterations is not None else "",
        )
        iterations = 0
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
        logger.info("Boucle terminée après %d itération(s).", iterations)

    def __repr__(self) -> str:
        return (
            f"LiveTrader(symbol={self.symbol!r}, timeframe={self.timeframe!r}, "
            f"strategy={self.strategy!r}, position={self.position.side}, "
            f"last_bar={self.last_bar.isoformat() if self.last_bar is not None else None})"
        )


def _join(first: str, second: str) -> str:
    return f"{first} ; {second}" if first else second
