"""Regression test for https://github.com/modelcontextprotocol/python-sdk/issues/915

When a streamable HTTP MCP server is unreachable, transport errors must not
escape into the outer task group (which surfaces as an uncatchable
``RuntimeError: Attempted to exit cancel scope in a different task``).
"""

import anyio
import httpx
import pytest

from mcp.client.session_group import ClientSessionGroup, StreamableHttpParameters


def _has_cancel_scope_runtime_error(exc: BaseException) -> bool:
    if isinstance(exc, RuntimeError) and "cancel scope" in str(exc):
        return True
    if exc.__cause__ is not None and _has_cancel_scope_runtime_error(exc.__cause__):
        return True
    if exc.__context__ is not None and exc.__context__ is not exc.__cause__:
        if _has_cancel_scope_runtime_error(exc.__context__):
            return True
    if hasattr(exc, "exceptions"):
        return any(_has_cancel_scope_runtime_error(sub) for sub in exc.exceptions)
    return False


@pytest.mark.anyio
async def test_unreachable_streamable_http_does_not_raise_cancel_scope_runtime_error() -> None:
    async with ClientSessionGroup() as client_session_group:
        server_params = StreamableHttpParameters(url="http://127.0.0.1:1/mcp/")
        with anyio.fail_after(10):
            with pytest.raises(httpx.ConnectError) as exc_info:
                await client_session_group.connect_to_server(server_params)

        assert not _has_cancel_scope_runtime_error(exc_info.value)
