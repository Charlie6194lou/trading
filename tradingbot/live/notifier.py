"""Notifications du bot : journal, Telegram, ou plusieurs canaux à la fois.

Règle d'or : **une notification qui échoue ne doit jamais arrêter le bot**. Toutes les
implémentations de :class:`Notifier` attrapent leurs exceptions, écrivent un
avertissement dans le journal et renvoient ``False``.

- :class:`LogNotifier` écrit simplement le message dans le journal Python ;
- :class:`TelegramNotifier` envoie le message via l'API Bot de Telegram
  (``POST https://api.telegram.org/bot<token>/sendMessage``). L'objet ``session`` est
  injectable (tout objet avec une méthode ``post(url, json=..., timeout=...)``, par
  exemple ``requests.Session``) ; sans session, une petite implémentation reposant sur
  ``urllib`` (bibliothèque standard) est utilisée, donc aucune dépendance externe ;
- :class:`MultiNotifier` diffuse à plusieurs notificateurs.

Le jeton Telegram n'apparaît jamais dans le journal ni dans ``repr``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from json import dumps as _json_dumps
from json import loads as _json_loads
from typing import Any
from urllib import error as _urlerror
from urllib import request as _urlrequest

logger = logging.getLogger(__name__)

__all__ = [
    "LogNotifier",
    "MultiNotifier",
    "Notifier",
    "TelegramNotifier",
    "UrllibSession",
    "TELEGRAM_API_URL",
    "TELEGRAM_MAX_LENGTH",
]

#: gabarit de l'URL de l'API Bot Telegram
TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"
#: longueur maximale d'un message Telegram (au-delà, l'API refuse le message)
TELEGRAM_MAX_LENGTH = 4096
#: marque qui remplace le jeton dans les journaux
_MASK = "***"


class Notifier(ABC):
    """Interface d'un canal de notification : :meth:`send` renvoie ``True`` si le message
    est parti, ``False`` sinon, et ne lève **jamais** d'exception."""

    @abstractmethod
    def send(self, text: str) -> bool:
        """Envoie ``text`` ; renvoie ``True`` en cas de succès, ``False`` sinon."""


class LogNotifier(Notifier):
    """Notificateur qui écrit le message dans le journal Python (niveau ``level``)."""

    def __init__(self, level: int = logging.INFO, logger_name: str | None = None) -> None:
        self.level = int(level)
        self._logger = logging.getLogger(logger_name) if logger_name else logger

    def send(self, text: str) -> bool:
        try:
            self._logger.log(self.level, "%s", text)
        except Exception as exc:  # pragma: no cover - logging ne lève normalement jamais
            logger.warning("Notification journal impossible : %s", exc)
            return False
        return True

    def __repr__(self) -> str:
        return f"LogNotifier(level={logging.getLevelName(self.level)})"


class _UrllibResponse:
    """Réponse HTTP minimale, compatible avec l'interface de ``requests.Response``."""

    def __init__(self, status_code: int, body: bytes) -> None:
        self.status_code = int(status_code)
        self.content = body

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    def json(self) -> Any:
        return _json_loads(self.text)

    def __repr__(self) -> str:
        return f"_UrllibResponse(status_code={self.status_code})"


class UrllibSession:
    """Session HTTP minimale (bibliothèque standard) exposant ``post(url, json, timeout)``.

    Sert de solution par défaut à :class:`TelegramNotifier` quand aucune session n'est
    injectée : aucune dépendance à ``requests``.
    """

    def post(self, url: str, json: Any = None, timeout: float | None = 10.0) -> _UrllibResponse:
        """``POST`` JSON ; renvoie la réponse même en cas de statut HTTP d'erreur (4xx/5xx)."""
        data = _json_dumps(json if json is not None else {}).encode("utf-8")
        req = _urlrequest.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with _urlrequest.urlopen(req, timeout=timeout) as resp:
                return _UrllibResponse(resp.status, resp.read())
        except _urlerror.HTTPError as exc:
            body = exc.read() if hasattr(exc, "read") else b""
            return _UrllibResponse(exc.code, body)


class TelegramNotifier(Notifier):
    """Envoie les messages sur un chat Telegram via l'API Bot.

    Paramètres :
    - ``token`` : jeton du bot (créé via @BotFather) ;
    - ``chat_id`` : identifiant du chat destinataire ;
    - ``session`` : objet avec une méthode ``post(url, json=..., timeout=...)``
      (``requests.Session`` par exemple) ; ``None`` → :class:`UrllibSession` ;
    - ``timeout`` : délai maximal en secondes pour la requête HTTP.

    Toute exception (réseau, délai dépassé, réponse illisible) est attrapée :
    :meth:`send` renvoie ``False`` et un avertissement est journalisé, **sans le jeton ni
    l'identifiant de chat** (tous deux secrets : avec le jeton, le chat_id permet d'écrire
    dans la conversation ; ``repr`` les masque aussi).
    """

    def __init__(
        self,
        token: str,
        chat_id: str | int,
        session: Any | None = None,
        timeout: float = 10.0,
    ) -> None:
        if not token or not str(token).strip():
            raise ValueError("Le jeton Telegram (token) est vide.")
        if chat_id is None or not str(chat_id).strip():
            raise ValueError("L'identifiant de chat Telegram (chat_id) est vide.")
        if timeout is not None and timeout <= 0:
            raise ValueError(f"timeout doit être strictement positif (reçu {timeout!r}).")
        self._token = str(token).strip()
        self.chat_id = str(chat_id).strip()
        self.url = TELEGRAM_API_URL.format(token=self._token)
        self.session = session if session is not None else UrllibSession()
        self.timeout = timeout

    # ------------------------------------------------------------------ utilitaires
    def _mask(self, text: str) -> str:
        """Retire le jeton d'un texte destiné au journal."""
        return text.replace(self._token, _MASK) if self._token else text

    def _check_response(self, resp: Any) -> bool:
        """Interprète la réponse : statut HTTP puis champ ``ok`` du JSON Telegram."""
        status = getattr(resp, "status_code", None)
        if status is None:
            status = getattr(resp, "status", None)
        if status is not None and not 200 <= int(status) < 300:
            logger.warning(
                "Telegram a refusé la notification (HTTP %s) : %s",
                status,
                self._mask(_response_excerpt(resp)),
            )
            return False
        payload: Any = None
        json_method = getattr(resp, "json", None)
        if callable(json_method):
            try:
                payload = json_method()
            except Exception:
                payload = None
        if isinstance(payload, dict) and payload.get("ok") is False:
            logger.warning(
                "Telegram a refusé la notification : %s",
                self._mask(str(payload.get("description", payload))),
            )
            return False
        return True

    # ------------------------------------------------------------------ envoi
    def send(self, text: str) -> bool:
        message = str(text)
        if len(message) > TELEGRAM_MAX_LENGTH:
            message = message[: TELEGRAM_MAX_LENGTH - 1] + "…"
        try:
            resp = self.session.post(
                self.url,
                json={"chat_id": self.chat_id, "text": message},
                timeout=self.timeout,
            )
        except Exception as exc:
            logger.warning(
                "Notification Telegram impossible : %s: %s",
                type(exc).__name__,
                self._mask(str(exc)),
            )
            return False
        try:
            return self._check_response(resp)
        except Exception as exc:  # réponse exotique : on ne propage jamais
            logger.warning("Réponse Telegram illisible : %s", self._mask(str(exc)))
            return False

    def __repr__(self) -> str:
        tail = self._token[-4:] if len(self._token) > 4 else ""
        return f"TelegramNotifier(chat_id='{_MASK}', token='{_MASK}{tail}')"


class MultiNotifier(Notifier):
    """Diffuse chaque message à plusieurs notificateurs.

    :meth:`send` renvoie ``True`` seulement si **tous** les canaux ont réussi ; un canal
    qui échoue (ou lève) n'empêche pas les autres d'être appelés.
    """

    def __init__(self, notifiers: Iterable[Notifier]) -> None:
        self.notifiers: list[Notifier] = list(notifiers)

    def send(self, text: str) -> bool:
        all_ok = True
        for notifier in self.notifiers:
            try:
                ok = bool(notifier.send(text))
            except Exception as exc:
                logger.warning("Le notificateur %r a levé une exception : %s", notifier, exc)
                ok = False
            all_ok = all_ok and ok
        return all_ok

    def __repr__(self) -> str:
        return f"MultiNotifier({self.notifiers!r})"


def _response_excerpt(resp: Any, limit: int = 200) -> str:
    """Extrait court du corps d'une réponse, pour le journal."""
    text = getattr(resp, "text", None)
    if text is None:
        content = getattr(resp, "content", b"")
        text = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else ""
    text = str(text).strip()
    return text[:limit] + ("…" if len(text) > limit else "")
