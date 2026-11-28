"""The S3 client: one client per app, reused by every call, and concurrent bulk downloads."""

import asyncio
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from fastapi.testclient import TestClient

from main import app
from s3 import utils as s3_utils
from s3.utils import S3Client
from settings import S3_MAX_CONCURRENT_DOWNLOADS


class _FakeBody:
    def __init__(self, data: bytes) -> None:
        self._data = data

    async def read(self) -> bytes:
        return self._data


class _FakeS3:
    """The S3 API the client calls, in memory, with a delay to make downloads overlap."""

    def __init__(self, delay: float = 0.0) -> None:
        self.objects: dict[str, bytes] = {}
        self.delay = delay
        self.in_flight = 0
        self.max_in_flight = 0

    async def get_object(self, Bucket: str, Key: str) -> dict[str, Any]:
        del Bucket
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
            return {"Body": _FakeBody(self.objects[Key])}
        finally:
            self.in_flight -= 1

    async def put_object(self, Bucket: str, Key: str, Body: bytes, ContentType: str):
        del Bucket, ContentType
        self.objects[Key] = Body


class _FakeSession:
    """Stands in for `aioboto3.Session`, counting sessions and the clients they open."""

    sessions = 0
    clients_opened = 0
    clients_closed = 0
    s3 = _FakeS3()

    def __init__(self) -> None:
        type(self).sessions += 1

    @asynccontextmanager
    async def client(self, *_args: object, **_kwargs: object) -> AsyncIterator[_FakeS3]:
        type(self).clients_opened += 1
        try:
            yield type(self).s3
        finally:
            type(self).clients_closed += 1


@pytest.fixture
def fake_s3(monkeypatch: pytest.MonkeyPatch) -> Iterator[_FakeS3]:
    _FakeSession.sessions = _FakeSession.clients_opened = (
        _FakeSession.clients_closed
    ) = 0
    _FakeSession.s3 = _FakeS3()
    monkeypatch.setattr(s3_utils.aioboto3, "Session", _FakeSession)
    monkeypatch.setattr(s3_utils, "_clients", s3_utils._Clients())  # pyright: ignore[reportPrivateUsage]
    yield _FakeSession.s3


@pytest.mark.asyncio
async def test_an_open_client_is_created_once_and_reused_by_every_call(
    fake_s3: _FakeS3,
):
    await S3Client.open()
    key = await S3Client.upload_bytes(b"one", "image/jpeg")
    _ = await S3Client.download_bytes(key)
    _ = await S3Client.download_bytes(key)
    _ = await S3Client.bulk_download_bytes([key, key])
    assert (_FakeSession.sessions, _FakeSession.clients_opened) == (1, 1)
    assert _FakeSession.clients_closed == 0

    await S3Client.close()

    assert _FakeSession.clients_closed == 1
    assert fake_s3.objects == {key: b"one"}


@pytest.mark.asyncio
async def test_without_an_open_client_each_call_closes_its_own_from_one_session(
    fake_s3: _FakeS3,
):
    fake_s3.objects["k"] = b"bytes"

    assert await S3Client.download_bytes("k") == b"bytes"
    assert await S3Client.download_bytes("k") == b"bytes"

    assert _FakeSession.sessions == 1
    assert _FakeSession.clients_opened == _FakeSession.clients_closed == 2


@pytest.mark.usefixtures("fake_s3")
def test_the_app_opens_the_client_on_startup_and_closes_it_on_shutdown():
    with TestClient(app):
        assert (_FakeSession.clients_opened, _FakeSession.clients_closed) == (1, 0)

    assert _FakeSession.clients_closed == 1


@pytest.mark.usefixtures("fake_s3")
@pytest.mark.asyncio
async def test_a_bulk_download_of_no_keys_opens_no_client():
    assert await S3Client.bulk_download_bytes([]) == {}
    assert _FakeSession.clients_opened == 0


def test_a_client_opened_on_another_event_loop_is_not_reused(fake_s3: _FakeS3):
    fake_s3.objects["k"] = b"bytes"
    asyncio.run(S3Client.open())

    # `asyncio.run` gives this call a new loop, where the open client can't run.
    assert asyncio.run(S3Client.download_bytes("k")) == b"bytes"

    assert _FakeSession.clients_opened == 2


@pytest.mark.asyncio
async def test_bulk_download_overlaps_up_to_the_limit_and_keeps_the_key_order(
    fake_s3: _FakeS3,
):
    fake_s3.delay = 0.01
    keys = [f"key-{i}" for i in range(3 * S3_MAX_CONCURRENT_DOWNLOADS)]
    for key in keys:
        fake_s3.objects[key] = key.encode()

    downloaded = await S3Client.bulk_download_bytes([*keys, keys[0]])

    assert fake_s3.max_in_flight == S3_MAX_CONCURRENT_DOWNLOADS
    assert list(downloaded) == keys
    assert all(downloaded[key] == key.encode() for key in keys)


@pytest.mark.asyncio
async def test_bulk_download_raises_the_first_failure_once_every_download_ended(
    fake_s3: _FakeS3,
):
    fake_s3.delay = 0.01
    fake_s3.objects["present"] = b"bytes"

    with pytest.raises(KeyError, match="missing"):
        _ = await S3Client.bulk_download_bytes(["present", "missing", "absent"])

    assert fake_s3.in_flight == 0
    assert _FakeSession.clients_opened == _FakeSession.clients_closed == 1
