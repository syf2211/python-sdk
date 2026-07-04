"""Regression test for issue #3060.

Concurrent POSTs on the same stateful streamable-HTTP session must not reuse
an in-flight JSON-RPC request id. The second request should be rejected with
INVALID_REQUEST instead of cross-wiring responses.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import anyio
import httpx
import pytest
from mcp_types import (
    INVALID_REQUEST,
    CallToolRequestParams,
    CallToolResult,
    ListToolsResult,
    PaginatedRequestParams,
    TextContent,
    Tool,
)
from starlette.applications import Starlette
from starlette.routing import Mount

from mcp.server import Server, ServerRequestContext
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from tests.interaction.transports import StreamingASGITransport

BASE_URL = "http://127.0.0.1:8000"
CONTENT_TYPE_JSON = "application/json"
HEADERS = {
    "accept": "application/json, text/event-stream",
    "content-type": "application/json",
}


def _parse_sse_response(text: str) -> dict[str, Any] | None:
    for line in text.splitlines():
        if line.startswith("data: "):
            payload = json.loads(line.removeprefix("data: "))
            if "result" in payload or "error" in payload:
                return payload
    return None


def _parse_response(response: httpx.Response) -> dict[str, Any] | None:
    content_type = response.headers.get("content-type", "")
    if CONTENT_TYPE_JSON in content_type:
        return response.json()
    return _parse_sse_response(response.text)


async def _handle_list_tools(
    ctx: ServerRequestContext, params: PaginatedRequestParams | None
) -> ListToolsResult:
    return ListToolsResult(
        tools=[
            Tool(
                name="echo",
                description="Echo a sentinel after a short delay",
                input_schema={
                    "type": "object",
                    "properties": {"sentinel": {"type": "string"}},
                    "required": ["sentinel"],
                },
            )
        ]
    )


async def _handle_call_tool(ctx: ServerRequestContext, params: CallToolRequestParams) -> CallToolResult:
    sentinel = str(params.arguments.get("sentinel", ""))
    await anyio.sleep(0.3)
    return CallToolResult(content=[TextContent(type="text", text=f"ECHO:{sentinel}")])


@asynccontextmanager
async def slow_echo_app() -> AsyncIterator[Starlette]:
    server = Server(
        "slow-echo-server",
        on_list_tools=_handle_list_tools,
        on_call_tool=_handle_call_tool,
    )
    session_manager = StreamableHTTPSessionManager(
        app=server,
        stateless=False,
        security_settings=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )
    app = Starlette(routes=[Mount("/mcp", app=session_manager.handle_request)])
    async with session_manager.run():
        yield app


async def _initialize(client: httpx.AsyncClient) -> str:
    response = await client.post(
        f"{BASE_URL}/mcp",
        headers=HEADERS,
        json={
            "jsonrpc": "2.0",
            "id": "init-1",
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "repro", "version": "0"},
            },
        },
    )
    response.raise_for_status()
    session_id = response.headers["mcp-session-id"]
    await client.post(
        f"{BASE_URL}/mcp",
        headers={**HEADERS, "mcp-session-id": session_id},
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
    )
    return session_id


async def _call_tool(
    client: httpx.AsyncClient,
    session_id: str,
    rpc_id: int,
    sentinel: str,
) -> tuple[str, str, dict[str, Any] | None]:
    body = {
        "jsonrpc": "2.0",
        "id": rpc_id,
        "method": "tools/call",
        "params": {"name": "echo", "arguments": {"sentinel": sentinel}},
    }
    try:
        response = await asyncio.wait_for(
            client.post(
                f"{BASE_URL}/mcp",
                headers={**HEADERS, "mcp-session-id": session_id},
                json=body,
            ),
            timeout=5.0,
        )
    except TimeoutError:
        return sentinel, "TIMEOUT", None

    payload = _parse_response(response)
    if payload is None:
        return sentinel, "OTHER", None
    if "error" in payload:
        return sentinel, "ERROR", payload
    result_text = json.dumps(payload["result"])
    if f"ECHO:{sentinel}" in result_text:
        return sentinel, "OK", payload
    return sentinel, "SWAPPED", payload


@pytest.mark.anyio
async def test_duplicate_in_flight_request_id_is_rejected() -> None:
    async with slow_echo_app() as app:
        transport = StreamingASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url=BASE_URL, follow_redirects=True
        ) as client:
            session_id = await _initialize(client)

            first, second = await asyncio.gather(
                _call_tool(client, session_id, 1, "SENT_A"),
                _call_tool(client, session_id, 1, "SENT_B"),
            )

            statuses = {first[1], second[1]}
            assert statuses == {"OK", "ERROR"}

            error_payload = first[2] if first[1] == "ERROR" else second[2]
            ok_payload = first[2] if first[1] == "OK" else second[2]
            assert error_payload is not None
            assert ok_payload is not None
            assert error_payload["id"] == 1
            assert error_payload["error"]["code"] == INVALID_REQUEST
            assert "ECHO:SENT_" in json.dumps(ok_payload["result"])

            # The id should be reusable after the first request completes.
            _, reuse_status, reuse_payload = await _call_tool(client, session_id, 1, "SENT_REUSE")
            assert reuse_status == "OK"
            assert reuse_payload is not None
            assert "ECHO:SENT_REUSE" in json.dumps(reuse_payload["result"])
