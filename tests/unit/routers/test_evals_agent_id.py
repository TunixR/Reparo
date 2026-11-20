"""``WS /evals/recovery/ws`` closes with 1008 on an ``X-AGENT-ID`` that isn't a UUID."""

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect


def test_malformed_agent_id_is_rejected_with_1008(
    client: TestClient,
    make_robot_key,  # pyright: ignore[reportMissingParameterType]
) -> None:
    _ = make_robot_key(name="k", enabled=True, key_raw="robot-key-0001")
    with (
        pytest.raises(WebSocketDisconnect) as exc,
        client.websocket_connect(
            "/evals/recovery/ws",
            headers={"X-ROBOT-KEY": "robot-key-0001", "X-AGENT-ID": "abc"},
        ),
    ):
        pass
    assert exc.value.code == 1008
