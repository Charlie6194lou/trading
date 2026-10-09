"""Gestion du risque : taille des positions et garde-fous.

Le :class:`RiskManager` répond à deux questions :

- **Combien acheter ?** (:meth:`RiskManager.position_size`) : on risque au plus
  ``risk_per_trade`` du capital si le stop est touché, sans jamais dépasser
  ``max_position_fraction`` du capital en notionnel.
- **A-t-on encore le droit de trader ?** (:meth:`RiskManager.can_open`) : le *kill switch*
  bloque définitivement après un drawdown trop important (jusqu'à :meth:`RiskManager.reset`),
  et la limite de perte journalière bloque jusqu'au jour UTC suivant.

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

__all__ = ["RiskConfig", "RiskManager"]

_ZERO_OFFSET = timedelta(0)


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

    # ------------------------------------------------------------------ config
    def _validate_config(self) -> None:
        cfg = self.config
        if _is_bad(cfg.risk_per_trade) or cfg.risk_per_trade <= 0:
            raise ValueError("risk_per_trade doit être un nombre strictement positif")
        if _is_bad(cfg.max_position_fraction) or cfg.max_position_fraction <= 0:
            raise ValueError("max_position_fraction doit être un nombre strictement positif")
        if cfg.max_drawdown is not None and cfg.max_drawdown < 0:
            raise ValueError("max_drawdown doit être positif, 0 ou None")
        if cfg.daily_loss_limit is not None and cfg.daily_loss_limit < 0:
            raise ValueError("daily_loss_limit doit être positif, 0 ou None")
        if _is_bad(cfg.min_trade_notional) or cfg.min_trade_notional < 0:
            raise ValueError("min_trade_notional doit être positif ou nul")
        if cfg.vol_target is not None and (_is_bad(cfg.vol_target) or cfg.vol_target <= 0):
            raise ValueError("vol_target doit être strictement positif ou None")

    @property
    def peak_equity(self) -> float | None:
        """Plus haut d'équité observé depuis le dernier :meth:`reset`."""
        return self._peak_equity

    @property
    def day_start_equity(self) -> float | None:
        """Équité de référence du jour UTC courant."""
        return self._day_start_equity

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
            self.reset_daily(ts, equity if self._last_equity is None else self._last_equity)
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
        if self._daily_loss(equity) > (self.config.daily_loss_limit or math.inf):
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
        suivi renvoie False (la limite est remise à zéro par :meth:`update`).
        """
        if not self.config.daily_loss_limit:
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
            f"day={self._day}, daily_locked={self._daily_locked})"
        )


def _opt_float(x: Any) -> float | None:
    return None if x is None else float(x)
