"""Tests des notificateurs : journal, Telegram avec session factice (capture URL / JSON,
exceptions, statuts HTTP), session ``urllib`` avec ``urlopen`` remplacé, multi-canal.
Aucun accès réseau."""

from __future__ import annotations

import io
import json
import logging
import urllib.error
from typing import Any

import pytest

from tradingbot.live import notifier as notifier_module
from tradingbot.live.notifier import (
    TELEGRAM_MAX_LENGTH,
    LogNotifier,
    MultiNotifier,
    Notifier,
    TelegramNotifier,
    UrllibSession,
)

TOKEN = "123456789:ABCdefGHIjklMNOpqrSTUvwxYZ"
CHAT = "987654321"
EXPECTED_URL = f"https://api.telegram.org/bot{TOKEN}/sendMessage"


class FakeResponse:
    """Réponse HTTP factice : statut + JSON (ou exception à la lecture du JSON)."""

    def __init__(
        self, status_code: int = 200, payload: Any = None, text: str | None = None
    ) -> None:
        self.status_code = status_code
        self._payload = payload if payload is not None else {"ok": True, "result": {}}
        self.text = text if text is not None else json.dumps(self._payload, default=str)

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeSession:
    """Capture les appels ``post`` ; peut lever une exception ou renvoyer une réponse."""

    def __init__(self, response: Any = None, raise_exc: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.response = response if response is not None else FakeResponse()
        self.raise_exc = raise_exc

    def post(self, url: str, json: Any = None, timeout: float | None = None) -> Any:
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.response


class ListNotifier(Notifier):
    def __init__(self, result: bool = True) -> None:
        self.messages: list[str] = []
        self.result = result

    def send(self, text: str) -> bool:
        self.messages.append(text)
        return self.result


class ExplodingNotifier(Notifier):
    def send(self, text: str) -> bool:
        raise RuntimeError("boum")


# ============================================================================ interface
def test_notifier_is_abstract() -> None:
    with pytest.raises(TypeError):
        Notifier()  # type: ignore[abstract]


# ============================================================================ LogNotifier
def test_log_notifier_logs_at_requested_level(caplog: pytest.LogCaptureFixture) -> None:
    n = LogNotifier(level=logging.WARNING)
    with caplog.at_level(logging.DEBUG, logger="tradingbot.live.notifier"):
        assert n.send("Entrée LONG BTC/USDT") is True
    records = [r for r in caplog.records if r.name == "tradingbot.live.notifier"]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    assert records[0].getMessage() == "Entrée LONG BTC/USDT"


def test_log_notifier_default_level_is_info_and_custom_logger(
    caplog: pytest.LogCaptureFixture,
) -> None:
    n = LogNotifier(logger_name="mon.bot")
    assert n.level == logging.INFO
    with caplog.at_level(logging.INFO, logger="mon.bot"):
        assert n.send("bonjour") is True
    assert any(r.name == "mon.bot" and r.getMessage() == "bonjour" for r in caplog.records)
    assert "INFO" in repr(n)


# ============================================================================ Telegram
def test_telegram_posts_json_to_bot_api() -> None:
    session = FakeSession()
    n = TelegramNotifier(TOKEN, CHAT, session=session, timeout=7)
    assert n.send("Achat 0.1 BTC @ 50000") is True
    assert len(session.calls) == 1
    call = session.calls[0]
    assert call["url"] == EXPECTED_URL
    assert call["json"] == {"chat_id": CHAT, "text": "Achat 0.1 BTC @ 50000"}
    assert call["timeout"] == 7


def test_telegram_default_timeout_is_10_seconds() -> None:
    session = FakeSession()
    TelegramNotifier(TOKEN, CHAT, session=session).send("x")
    assert session.calls[0]["timeout"] == 10.0


def test_telegram_int_chat_id_is_sent_as_string() -> None:
    session = FakeSession()
    TelegramNotifier(TOKEN, 42, session=session).send("x")
    assert session.calls[0]["json"]["chat_id"] == "42"


def test_telegram_session_exception_returns_false_and_hides_token(
    caplog: pytest.LogCaptureFixture,
) -> None:
    session = FakeSession(raise_exc=ConnectionError(f"échec vers {EXPECTED_URL}"))
    n = TelegramNotifier(TOKEN, CHAT, session=session)
    with caplog.at_level(logging.WARNING, logger="tradingbot.live.notifier"):
        assert n.send("bonjour") is False
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "Telegram" in warnings[0].getMessage()
    assert TOKEN not in caplog.text
    assert "***" in caplog.text


def test_telegram_timeout_exception_is_contained() -> None:
    session = FakeSession(raise_exc=TimeoutError("délai dépassé"))
    assert TelegramNotifier(TOKEN, CHAT, session=session).send("x") is False


def test_telegram_http_error_status_returns_false(caplog: pytest.LogCaptureFixture) -> None:
    resp = FakeResponse(401, {"ok": False, "description": "Unauthorized"})
    n = TelegramNotifier(TOKEN, CHAT, session=FakeSession(response=resp))
    with caplog.at_level(logging.WARNING, logger="tradingbot.live.notifier"):
        assert n.send("x") is False
    assert "401" in caplog.text


def test_telegram_ok_false_payload_returns_false(caplog: pytest.LogCaptureFixture) -> None:
    resp = FakeResponse(200, {"ok": False, "description": "chat not found"})
    n = TelegramNotifier(TOKEN, CHAT, session=FakeSession(response=resp))
    with caplog.at_level(logging.WARNING, logger="tradingbot.live.notifier"):
        assert n.send("x") is False
    assert "chat not found" in caplog.text


def test_telegram_unreadable_json_with_2xx_status_is_success() -> None:
    resp = FakeResponse(200, payload=ValueError("pas du JSON"), text="<html>")
    assert TelegramNotifier(TOKEN, CHAT, session=FakeSession(response=resp)).send("x") is True


def test_telegram_bare_response_object_is_accepted() -> None:
    """Une réponse sans statut ni json() (session exotique) est considérée envoyée."""
    assert TelegramNotifier(TOKEN, CHAT, session=FakeSession(response=object())).send("x")


def test_telegram_truncates_messages_over_limit() -> None:
    session = FakeSession()
    TelegramNotifier(TOKEN, CHAT, session=session).send("a" * (TELEGRAM_MAX_LENGTH + 500))
    sent = session.calls[0]["json"]["text"]
    assert len(sent) == TELEGRAM_MAX_LENGTH
    assert sent.endswith("…")
    assert sent[:-1] == "a" * (TELEGRAM_MAX_LENGTH - 1)


def test_telegram_short_message_not_modified() -> None:
    session = FakeSession()
    TelegramNotifier(TOKEN, CHAT, session=session).send("é" * TELEGRAM_MAX_LENGTH)
    assert session.calls[0]["json"]["text"] == "é" * TELEGRAM_MAX_LENGTH


@pytest.mark.parametrize("token", ["", "   ", None])
def test_telegram_requires_token(token: Any) -> None:
    with pytest.raises(ValueError, match="jeton"):
        TelegramNotifier(token, CHAT, session=FakeSession())


@pytest.mark.parametrize("chat", ["", "  ", None])
def test_telegram_requires_chat_id(chat: Any) -> None:
    with pytest.raises(ValueError, match="chat_id"):
        TelegramNotifier(TOKEN, chat, session=FakeSession())


def test_telegram_requires_positive_timeout() -> None:
    with pytest.raises(ValueError, match="timeout"):
        TelegramNotifier(TOKEN, CHAT, session=FakeSession(), timeout=0)


def test_telegram_repr_hides_token() -> None:
    n = TelegramNotifier(TOKEN, CHAT, session=FakeSession())
    text = repr(n)
    assert TOKEN not in text
    assert "***" in text
    assert CHAT in text


def test_telegram_default_session_is_urllib_without_network_call() -> None:
    n = TelegramNotifier(TOKEN, CHAT)
    assert isinstance(n.session, UrllibSession)


# ============================================================================ UrllibSession
class _FakeHTTPResponse:
    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeHTTPResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def test_urllib_session_posts_json_body(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_urlopen(req: Any, timeout: float | None = None) -> _FakeHTTPResponse:
        seen["req"] = req
        seen["timeout"] = timeout
        return _FakeHTTPResponse(200, b'{"ok": true}')

    monkeypatch.setattr(notifier_module._urlrequest, "urlopen", fake_urlopen)
    resp = UrllibSession().post(EXPECTED_URL, json={"chat_id": CHAT, "text": "é"}, timeout=3)
    req = seen["req"]
    assert req.full_url == EXPECTED_URL
    assert req.get_method() == "POST"
    assert json.loads(req.data.decode("utf-8")) == {"chat_id": CHAT, "text": "é"}
    assert req.get_header("Content-type") == "application/json"
    assert seen["timeout"] == 3
    assert resp.status_code == 200
    assert resp.ok is True
    assert resp.json() == {"ok": True}
    assert "200" in repr(resp)


def test_urllib_session_returns_response_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(req: Any, timeout: float | None = None) -> _FakeHTTPResponse:
        raise urllib.error.HTTPError(
            EXPECTED_URL, 401, "Unauthorized", None, io.BytesIO(b'{"ok": false}')
        )

    monkeypatch.setattr(notifier_module._urlrequest, "urlopen", fake_urlopen)
    resp = UrllibSession().post(EXPECTED_URL, json={})
    assert resp.status_code == 401
    assert resp.ok is False
    assert resp.json() == {"ok": False}


def test_telegram_end_to_end_with_patched_urlopen(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[Any] = []

    def fake_urlopen(req: Any, timeout: float | None = None) -> _FakeHTTPResponse:
        captured.append(req)
        return _FakeHTTPResponse(200, b'{"ok": true}')

    monkeypatch.setattr(notifier_module._urlrequest, "urlopen", fake_urlopen)
    assert TelegramNotifier(TOKEN, CHAT).send("salut") is True
    assert captured[0].full_url == EXPECTED_URL
    assert json.loads(captured[0].data) == {"chat_id": CHAT, "text": "salut"}


def test_telegram_url_error_through_urllib_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(req: Any, timeout: float | None = None) -> _FakeHTTPResponse:
        raise urllib.error.URLError("réseau indisponible")

    monkeypatch.setattr(notifier_module._urlrequest, "urlopen", fake_urlopen)
    assert TelegramNotifier(TOKEN, CHAT).send("salut") is False


# ============================================================================ MultiNotifier
def test_multi_notifier_sends_to_all_and_returns_true() -> None:
    a, b = ListNotifier(), ListNotifier()
    multi = MultiNotifier([a, b])
    assert multi.send("msg") is True
    assert a.messages == ["msg"]
    assert b.messages == ["msg"]
    assert "MultiNotifier" in repr(multi)


def test_multi_notifier_one_failure_still_calls_others() -> None:
    a, b, c = ListNotifier(), ListNotifier(result=False), ListNotifier()
    assert MultiNotifier([a, b, c]).send("msg") is False
    assert a.messages == b.messages == c.messages == ["msg"]


def test_multi_notifier_contains_exceptions(caplog: pytest.LogCaptureFixture) -> None:
    ok = ListNotifier()
    with caplog.at_level(logging.WARNING, logger="tradingbot.live.notifier"):
        assert MultiNotifier([ExplodingNotifier(), ok]).send("msg") is False
    assert ok.messages == ["msg"]
    assert "boum" in caplog.text


def test_multi_notifier_empty_is_success() -> None:
    assert MultiNotifier([]).send("msg") is True


def test_multi_notifier_accepts_any_iterable() -> None:
    a = ListNotifier()
    assert MultiNotifier(iter([a])).send("x") is True
    assert a.messages == ["x"]
