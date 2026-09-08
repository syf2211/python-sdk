"""Regression tests for client disconnect while reading POST bodies."""

from __future__ import annotations

import logging
import uuid

import pytest
from starlette.types import Message, Scope

from mcp.server.sse import SseServerTransport
from mcp.server.streamable_http import StreamableHTTPServerTransport
from mcp.shared._context_streams import create_context_streams


class _DisconnectBeforeBody:
    """ASGI POST whose client disconnects before sending a request body."""

    def __init__(
        self,
        *,
        path: str = "/",
        query_string: bytes = b"",
        headers: list[tuple[bytes, bytes]],
    ) -> None:
        self.scope: Scope = {
            "type": "http",
            "method": "POST",
            "path": path,
            "query_string": query_string,
            "headers": headers,
        }
        self.sent: list[Message] = []

    async def receive(self) -> Message:
        return {"type": "http.disconnect"}

    async def send(self, message: Message) -> None:
        self.sent.append(message)


@pytest.mark.anyio
async def test_streamable_http_client_disconnect_before_body_is_not_an_error(caplog: pytest.LogCaptureFixture) -> None:
    transport = StreamableHTTPServerTransport(mcp_session_id=None)
    post = _DisconnectBeforeBody(
        headers=[
            (b"host", b"127.0.0.1"),
            (b"content-type", b"application/json"),
            (b"accept", b"application/json, text/event-stream"),
            (b"content-length", b"58"),
        ],
    )

    caplog.set_level(logging.DEBUG, logger="mcp.server.streamable_http")
    async with transport.connect():
        await transport.handle_request(post.scope, post.receive, post.send)

    assert post.sent == []
    assert not any(
        record.levelno >= logging.ERROR and record.getMessage() == "Error handling POST request"
        for record in caplog.records
    )
    assert any(record.message == "Client disconnected before POST body was received" for record in caplog.records)


@pytest.mark.anyio
async def test_sse_client_disconnect_before_body_is_not_an_error(caplog: pytest.LogCaptureFixture) -> None:
    transport = SseServerTransport("/messages/")
    session_id = uuid.uuid4()
    send_stream, receive_stream = create_context_streams()
    transport._read_stream_writers[session_id] = send_stream

    post = _DisconnectBeforeBody(
        path="/messages/",
        query_string=f"session_id={session_id.hex}".encode(),
        headers=[
            (b"host", b"127.0.0.1"),
            (b"content-type", b"application/json"),
        ],
    )

    caplog.set_level(logging.DEBUG, logger="mcp.server.sse")
    await transport.handle_post_message(post.scope, post.receive, post.send)

    assert post.sent == []
    assert not any(
        record.levelno >= logging.ERROR and record.getMessage() == "Error handling POST request"
        for record in caplog.records
    )
    assert any(record.message == "Client disconnected before POST body was received" for record in caplog.records)

    await send_stream.aclose()
    await receive_stream.aclose()
