"""Gestion du risque : taille des positions et garde-fous.

Le :class:`RiskManager` répond à deux questions :

- **Combien acheter ?** (:meth:`RiskManager.position_size`) : on risque au plus
  ``risk_per_trade`` du capital si le stop est touché, sans jamais dépasser
  ``max_position_fraction`` du capital en notionnel.
- **Où placer le stop ?** (:meth:`RiskManager.protective_stop`) : une position ne s'ouvre
  **jamais** sans stop. Priorité : stop de la stratégie (s'il est du bon côté du prix),
  puis ``stop_loss_pct`` de la configuration de backtest, puis un stop de secours à
  ``default_stop_atr_mult × ATR(14)``, enfin un stop de dernier recours à
  ``default_stop_pct`` du prix (avec un avertissement).
- **A-t-on encore le droit de trader ?** (:meth:`RiskManager.can_open`) : le *kill switch*
  bloque définitivement après un drawdown trop important (jusqu'à :meth:`RiskManager.reset`),
  et la limite de perte journalière bloque jusqu'au jour UTC suivant. La limite
  journalière n'a de sens qu'en intrajournalier : l'appelant la désactive via
  :meth:`RiskManager.set_daily_limit_active` quand les bougies font un jour ou plus.

Toutes les fractions sont exprimées en décimal (``0.01`` = 1 %), tous les montants
dans la devise de cotation.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["DEFAULT_STOP_ATR_WINDOW", "RiskConfig", "RiskManager"]

_ZERO_OFFSET = timedelta(0)
_ONE_DAY = timedelta(days=1)
#: fenêtre de l'ATR utilisé pour le stop de secours (``default_stop_atr_mult × ATR(14)``)
DEFAULT_STOP_ATR_WINDOW = 14


@dataclass
class RiskConfig:
    """Paramètres de gestion du risque (miroir de la section ``[risk]`` de config.toml)."""

    #: fraction du capital perdue si le stop est touché (0.01 = 1 %)
    risk_per_trade: float = 0.01
    #: notionnel maximal d'une position, en fraction de l'équité
    max_position_fraction: float = 0.95
    #: drawdown (depuis le plus haut) qui déclenche le kill switch ; 0 ou None = désactivé
    max_drawdown: float | None = 0.20
    #: perte journalière (depuis l'équité de début de jour UTC) qui bloque jusqu'au
    #: lendemain ; 0 ou None = désactivé
    daily_loss_limit: float | None = 0.05
    #: autoriser la vente à découvert
    allow_short: bool = False
    #: en dessous de ce notionnel, on ne trade pas (ex. 10 USDT)
    min_trade_notional: float = 10.0
    #: volatilité annualisée cible ; si définie, la taille est multipliée par
    #: ``min(1, vol_target / vol_réalisée)`` (jamais de levier)
    vol_target: float | None = None
    #: stop de secours quand la stratégie n'en donne pas : ``prix ∓ mult × ATR(14)``
    default_stop_atr_mult: float = 3.0
    #: stop de dernier recours (fraction du prix) quand l'ATR est indisponible (NaN ou 0)
    default_stop_pct: float = 0.05


def _is_bad(x: float | None) -> bool:
    """True si ``x`` est None, NaN ou infini."""
    return x is None or not math.isfinite(x)


def _to_utc(timestamp: Any) -> datetime:
    """Convertit n'importe quel horodatage (datetime, Timestamp, str, epoch) en datetime UTC.

    Un horodatage naïf est supposé déjà en UTC.
    """
    if isinstance(timestamp, datetime):
        # Chemin rapide (pd.Timestamp est une sous-classe de datetime).
        if timestamp.tzinfo is None:
            return timestamp.replace(tzinfo=UTC)
        if timestamp.utcoffset() == _ZERO_OFFSET:
            return timestamp
        return timestamp.astimezone(UTC)
    ts = pd.Timestamp(timestamp)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.to_pydatetime()


class RiskManager:
    """Calcule la taille des positions et applique les garde-fous.

    Utilisation typique (moteur de backtest ou bot live) : à chaque bougie clôturée,
    appeler :meth:`update` avec l'équité courante, puis interroger
    :meth:`kill_switch_active` / :meth:`daily_limit_hit` / :meth:`can_open`.
    """

    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config: RiskConfig = config if config is not None else RiskConfig()
        self._validate_config()
        self._peak_equity: float | None = None
        self._kill_switch: bool = False
        self._day: date | None = None
        self._day_start_equity: float | None = None
        self._daily_locked: bool = False
        self._last_equity: float | None = None
        self._last_timestamp: datetime | None = None
        # Décision du contexte d'appel (timeframe), pas un état : ni sérialisée, ni remise
        # à zéro par reset().
        self._daily_limit_active: bool = True
        self._daily_limit_inactive_reason: str = ""

    # ------------------------------------------------------------------ config
    def _validate_config(self) -> None:
        cfg = self.config
        if _is_bad(cfg.risk_per_trade) or cfg.risk_per_trade <= 0:
            raise ValueError("risk_per_trade doit être un nombre strictement positif")
        if (
            _is_bad(cfg.max_position_fraction)
            or cfg.max_position_fraction <= 0
            or cfg.max_position_fraction > 1
        ):
            raise ValueError(
                "max_position_fraction doit être compris entre 0 (exclu) et 1 (inclus) : "
                "1.0 = tout le capital, jamais de levier"
            )
        if cfg.max_drawdown is not None and cfg.max_drawdown < 0:
            raise ValueError("max_drawdown doit être positif, 0 ou None")
        if cfg.daily_loss_limit is not None and cfg.daily_loss_limit < 0:
            raise ValueError("daily_loss_limit doit être positif, 0 ou None")
        if _is_bad(cfg.min_trade_notional) or cfg.min_trade_notional < 0:
            raise ValueError("min_trade_notional doit être positif ou nul")
        if cfg.vol_target is not None and (_is_bad(cfg.vol_target) or cfg.vol_target <= 0):
            raise ValueError("vol_target doit être strictement positif ou None")
        if _is_bad(cfg.default_stop_atr_mult) or cfg.default_stop_atr_mult <= 0:
            raise ValueError("default_stop_atr_mult doit être un nombre strictement positif")
        if _is_bad(cfg.default_stop_pct) or not (0 < cfg.default_stop_pct < 1):
            raise ValueError("default_stop_pct doit être une fraction entre 0 et 1 (exclus)")

    @property
    def peak_equity(self) -> float | None:
        """Plus haut d'équité observé depuis le dernier :meth:`reset`."""
        return self._peak_equity

    @property
    def day_start_equity(self) -> float | None:
        """Équité de référence du jour UTC courant."""
        return self._day_start_equity

    @property
    def daily_limit_active(self) -> bool:
        """False si l'appelant a désactivé la limite journalière (bougies >= 1 jour)."""
        return self._daily_limit_active

    @property
    def daily_limit_inactive_reason(self) -> str:
        """Raison de la désactivation (chaîne vide si la limite est active)."""
        return self._daily_limit_inactive_reason

    def set_daily_limit_active(self, active: bool, reason: str = "") -> None:
        """Active / désactive la limite de perte journalière pour ce contexte d'exécution.

        La limite journalière compare l'équité à celle du début du jour UTC : sur des
        bougies d'un jour ou plus, chaque bougie est un nouveau jour et une seule bougie
        à -5 % la déclencherait à tort. Le moteur de backtest et le ``LiveTrader``
        appellent cette méthode d'après le timeframe. Désactivée, la limite ne se
        verrouille jamais (:meth:`update`), :meth:`daily_limit_hit` renvoie toujours
        ``False`` et :meth:`can_open` ne la vérifie plus. Ce choix n'est ni sérialisé par
        :meth:`state` ni effacé par :meth:`reset` : il dépend du timeframe, pas de
        l'historique.
        """
        self._daily_limit_active = bool(active)
        self._daily_limit_inactive_reason = "" if active else str(reason)
        if not active:
            self._daily_locked = False

    # ------------------------------------------------------------------ stop
    def protective_stop(
        self,
        side: int,
        price: float,
        strategy_stop: float | None = None,
        atr: float | None = None,
        stop_loss_pct: float | None = None,
    ) -> float:
        """Prix du stop protecteur d'une entrée : une position ne s'ouvre jamais sans stop.

        Ordre de priorité :

        1. ``strategy_stop`` s'il est fini **et du bon côté** du prix d'entrée (long :
           ``stop < price`` ; short : ``stop > price``) ;
        2. ``stop_loss_pct`` (ex. ``BacktestConfig.stop_loss_pct``) s'il est fourni et > 0 :
           ``price × (1 ∓ stop_loss_pct)`` ;
        3. ``price ∓ default_stop_atr_mult × atr`` si ``atr`` est fini et > 0 (ATR(14) des
           bougies déjà clôturées, voir ``DEFAULT_STOP_ATR_WINDOW``) ;
        4. sinon ``price × (1 ∓ default_stop_pct)``, avec un avertissement journalisé.

        :param side: +1 long, -1 short.
        :param price: prix d'exécution prévu (slippage inclus), strictement positif.
        :raises ValueError: ``side`` hors de {-1, 1} ou ``price`` invalide.
        """
        if side not in (1, -1):
            raise ValueError(f"side doit valoir 1 (long) ou -1 (short), reçu {side!r}")
        if _is_bad(price) or price <= 0:
            raise ValueError(f"price doit être un nombre strictement positif, reçu {price!r}")
        cfg = self.config
        if not _is_bad(strategy_stop):
            if (side == 1 and strategy_stop < price) or (side == -1 and strategy_stop > price):  # type: ignore[operator]
                return float(strategy_stop)  # type: ignore[arg-type]
            logger.debug(
                "Stop de la stratégie %.6g du mauvais côté du prix %.6g (side %d) : ignoré",
                strategy_stop,
                price,
                side,
            )
        if stop_loss_pct is not None and not _is_bad(stop_loss_pct) and stop_loss_pct > 0:
            return float(price * (1.0 - side * stop_loss_pct))
        if not _is_bad(atr) and atr > 0:  # type: ignore[operator]
            return float(price - side * cfg.default_stop_atr_mult * atr)  # type: ignore[operator]
        logger.warning(
            "Aucun stop de stratégie et ATR indisponible : stop de dernier recours à %.1f %% "
            "du prix %.6g (side %d)",
            cfg.default_stop_pct * 100,
            price,
            side,
        )
        return float(price * (1.0 - side * cfg.default_stop_pct))

    # ------------------------------------------------------------------ sizing
    def position_size(
        self,
        equity: float,
        price: float,
        stop_price: float | None = None,
        realized_vol: float | None = None,
    ) -> float:
        """Taille de position en unités de l'actif (jamais négative, jamais NaN).

        - Avec ``stop_price`` : ``size = equity * risk_per_trade / |price - stop_price|``.
        - Sans stop (ou stop égal au prix) : ``size = equity * max_position_fraction / price``.
        - Plafond : ``size * price <= equity * max_position_fraction``.
        - Si ``vol_target`` est configuré et ``realized_vol`` fourni :
          ``size *= min(1, vol_target / realized_vol)``.
        - Si le notionnel ``size * price`` est inférieur à ``min_trade_notional`` : ``0.0``.
        - Entrées invalides (prix ou équité nuls, négatifs, NaN) : ``0.0``.
        """
        if _is_bad(equity) or _is_bad(price) or equity <= 0 or price <= 0:
            return 0.0
        cfg = self.config
        max_size = equity * cfg.max_position_fraction / price
        if not _is_bad(stop_price) and abs(price - stop_price) > 0:  # type: ignore[arg-type]
            size = equity * cfg.risk_per_trade / abs(price - stop_price)  # type: ignore[operator]
        else:
            size = max_size
        size = min(size, max_size)
        if cfg.vol_target is not None and not _is_bad(realized_vol) and realized_vol > 0:  # type: ignore[operator]
            size *= min(1.0, cfg.vol_target / realized_vol)  # type: ignore[operator]
        if _is_bad(size) or size <= 0 or size * price < cfg.min_trade_notional:
            return 0.0
        return float(size)

    # ------------------------------------------------------------------ suivi
    def update(self, timestamp: Any, equity: float) -> None:
        """Enregistre l'équité à ``timestamp`` : met à jour le plus haut, le jour courant,
        le kill switch et la limite journalière."""
        if _is_bad(equity):
            logger.warning("Équité invalide ignorée par le gestionnaire de risque : %r", equity)
            return
        ts = _to_utc(timestamp)
        day = ts.date()
        if self._day is None or day != self._day:
            # Référence du jour = dernière équité connue (clôture de la veille)... sauf si
            # elle date de plus d'un jour (redémarrage après une coupure, bougies espacées) :
            # elle ne décrit plus le début de ce jour, on prend l'équité courante.
            stale = (
                self._last_equity is None
                or self._last_timestamp is None
                or ts - self._last_timestamp > _ONE_DAY
            )
            self.reset_daily(ts, equity if stale else self._last_equity)
        self._last_timestamp = ts
        self._last_equity = float(equity)

        if self._peak_equity is None or equity > self._peak_equity:
            self._peak_equity = float(equity)
        dd = self.drawdown()
        max_dd = self.config.max_drawdown
        # INFO et non WARNING : en backtest / walk-forward ces événements sont simulés des
        # centaines de fois ; en live, LiveTrader avertit et notifie lui-même l'utilisateur.
        if max_dd and dd > max_dd and not self._kill_switch:
            self._kill_switch = True
            logger.info(
                "Kill switch déclenché à %s : drawdown %.2f %% > %.2f %%",
                ts.isoformat(),
                dd * 100,
                max_dd * 100,
            )
        if self._daily_limit_active and self._daily_loss(equity) > (
            self.config.daily_loss_limit or math.inf
        ):
            if not self._daily_locked:
                logger.info(
                    "Limite de perte journalière atteinte à %s (%.2f %%)",
                    ts.isoformat(),
                    self._daily_loss(equity) * 100,
                )
            self._daily_locked = True

    def drawdown(self) -> float:
        """Drawdown courant (fraction positive, 0 au plus haut)."""
        if self._peak_equity is None or self._last_equity is None or self._peak_equity <= 0:
            return 0.0
        return max(0.0, 1.0 - self._last_equity / self._peak_equity)

    def _daily_loss(self, equity: float) -> float:
        if self._day_start_equity is None or self._day_start_equity <= 0:
            return 0.0
        return max(0.0, 1.0 - equity / self._day_start_equity)

    def kill_switch_active(self) -> bool:
        """True une fois le drawdown maximal dépassé ; reste vrai jusqu'à :meth:`reset`."""
        return self._kill_switch

    def daily_limit_hit(self, timestamp: Any, equity: float) -> bool:
        """True si la limite de perte journalière bloque le trading à ``timestamp``.

        Ne modifie pas l'état : un horodatage appartenant à un autre jour UTC que le jour
        suivi renvoie False (la limite est remise à zéro par :meth:`update`). Toujours
        False si la limite est désactivée (config ou :meth:`set_daily_limit_active`).
        """
        if not self.config.daily_loss_limit or not self._daily_limit_active:
            return False
        day = _to_utc(timestamp).date()
        if self._day is None or day != self._day:
            return False
        if self._daily_locked:
            return True
        return (not _is_bad(equity)) and self._daily_loss(equity) > self.config.daily_loss_limit

    def can_open(self, side: int, timestamp: Any, equity: float) -> tuple[bool, str]:
        """(autorisé, raison) pour l'ouverture d'une position ``side`` (+1 long, -1 short)."""
        if side == 0:
            return False, "aucune position demandée (signal plat)"
        if side < 0 and not self.config.allow_short:
            return False, "vente à découvert désactivée (allow_short = false)"
        if _is_bad(equity) or equity <= 0:
            return False, "équité nulle ou négative"
        if self._kill_switch:
            return False, "kill switch actif : drawdown maximal dépassé"
        if self.daily_limit_hit(timestamp, equity):
            return False, "limite de perte journalière atteinte, reprise au prochain jour UTC"
        return True, "ok"

    def reset_daily(self, timestamp: Any, equity: float | None = None) -> None:
        """Démarre un nouveau jour UTC : nouvelle équité de référence, verrou levé.

        ``equity`` vaut par défaut la dernière équité connue.
        """
        ts = _to_utc(timestamp)
        self._day = ts.date()
        self._daily_locked = False
        if equity is None:
            equity = self._last_equity
        self._day_start_equity = None if _is_bad(equity) else float(equity)  # type: ignore[arg-type]

    def reset(self) -> None:
        """Remise à zéro complète (plus haut, kill switch, suivi journalier)."""
        self._peak_equity = None
        self._kill_switch = False
        self._day = None
        self._day_start_equity = None
        self._daily_locked = False
        self._last_equity = None
        self._last_timestamp = None

    # ------------------------------------------------------------------ état
    def state(self) -> dict[str, Any]:
        """État interne sérialisable en JSON (horodatages en ISO 8601)."""
        return {
            "config": asdict(self.config),
            "peak_equity": self._peak_equity,
            "kill_switch": self._kill_switch,
            "day": self._day.isoformat() if self._day is not None else None,
            "day_start_equity": self._day_start_equity,
            "daily_locked": self._daily_locked,
            "last_equity": self._last_equity,
            "last_timestamp": (
                self._last_timestamp.isoformat() if self._last_timestamp is not None else None
            ),
        }

    def load_state(self, d: dict[str, Any]) -> None:
        """Restaure un état produit par :meth:`state` (la config n'est pas modifiée)."""
        self._peak_equity = _opt_float(d.get("peak_equity"))
        self._kill_switch = bool(d.get("kill_switch", False))
        day = d.get("day")
        self._day = date.fromisoformat(day) if day else None
        self._day_start_equity = _opt_float(d.get("day_start_equity"))
        self._daily_locked = bool(d.get("daily_locked", False))
        self._last_equity = _opt_float(d.get("last_equity"))
        last_ts = d.get("last_timestamp")
        self._last_timestamp = _to_utc(last_ts) if last_ts else None

    def __repr__(self) -> str:
        return (
            f"RiskManager(peak={self._peak_equity}, kill_switch={self._kill_switch}, "
            f"day={self._day}, daily_locked={self._daily_locked}, "
            f"daily_limit_active={self._daily_limit_active})"
        )


def _opt_float(x: Any) -> float | None:
    return None if x is None else float(x)
