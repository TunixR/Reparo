import asyncio
import uuid
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Any

import aioboto3
from botocore.config import Config
from botocore.exceptions import ClientError

from settings import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET, S3_URL

# Downloads one bulk download runs at once. Below the pool size, so that a few
# exports and the screenshot uploads of running recoveries can share the client.
MAX_CONCURRENT_DOWNLOADS = 8
_MAX_POOL_CONNECTIONS = 32


@dataclass
class _SharedClient:
    """The client the app opened, and the event loop its connections belong to."""

    loop: asyncio.AbstractEventLoop
    client: Any  # aioboto3 ships no type stubs for its clients.
    exit_stack: AsyncExitStack


@dataclass
class _Clients:
    session: aioboto3.Session | None = None
    shared: _SharedClient | None = None


_clients = _Clients()


def _new_client() -> Any:  # aioboto3 ships no type stubs for its clients.
    """A client context of the process's session, which loads the S3 model once."""
    if _clients.session is None:
        _clients.session = aioboto3.Session()
    return _clients.session.client(
        "s3",
        endpoint_url=S3_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name="us-east-1",
        config=Config(max_pool_connections=_MAX_POOL_CONNECTIONS),
    )


class S3Client:
    @staticmethod
    async def open() -> None:
        """Open the client every call on this event loop reuses, until `close`.

        The app opens it in its lifespan. Its connections belong to the loop that
        opened it, so calls on another loop (`populate.py`, `asyncio.run`, tests)
        open a client of their own for the call.
        """
        if _clients.shared is not None:
            return
        exit_stack = AsyncExitStack()
        client = await exit_stack.enter_async_context(_new_client())
        _clients.shared = _SharedClient(
            loop=asyncio.get_running_loop(), client=client, exit_stack=exit_stack
        )

    @staticmethod
    async def close() -> None:
        """Close the client `open` opened, if any."""
        shared, _clients.shared = _clients.shared, None
        if shared is not None:
            await shared.exit_stack.aclose()

    @staticmethod
    @asynccontextmanager
    async def _client() -> AsyncIterator[Any]:
        shared = _clients.shared
        if shared is not None and shared.loop is asyncio.get_running_loop():
            yield shared.client
            return
        async with _new_client() as client:
            yield client

    @staticmethod
    async def ensure_bucket(bucket: str = S3_BUCKET) -> bool:
        """
        Create the bucket if it doesn't exist.

        Returns:
            bool: True if the bucket was created, False if it already existed.
        """
        async with S3Client._client() as s3_client:
            try:
                await s3_client.head_bucket(Bucket=bucket)
            except ClientError as exc:
                if exc.response.get("Error", {}).get("Code") not in (
                    "404",
                    "NoSuchBucket",
                ):
                    raise
            else:
                return False
            await s3_client.create_bucket(Bucket=bucket)
        return True

    @staticmethod
    async def upload_bytes(
        file_bytes: bytes, content_type: str, bucket: str = S3_BUCKET
    ) -> str:
        """
        Uploads bytes to the specified S3 bucket and returns the S3 URI.

        Args:
            file_bytes (bytes): The bytes to upload.
            content_type (str): The content type of the file.
            bucket (str): The S3 bucket to upload to. Defaults to S3_BUCKET.
        Returns:
            str: The S3 key of the uploaded file.
        """
        key = str(uuid.uuid4())
        async with S3Client._client() as s3_client:
            await s3_client.put_object(
                Bucket=bucket, Key=key, Body=file_bytes, ContentType=content_type
            )
        return key

    @staticmethod
    async def download_bytes(key: str, bucket: str = S3_BUCKET) -> bytes:
        """
        Downloads bytes from the specified S3 bucket using the given key.

        Args:
            key (str): The S3 key of the file to download.
            bucket (str): The S3 bucket to download from. Defaults to S3_BUCKET.
        Returns:
            bytes: The downloaded file bytes.
        """
        async with S3Client._client() as s3_client:
            return await _get_object_bytes(s3_client, bucket, key)

    @staticmethod
    async def bulk_download_bytes(
        keys: list[str], bucket: str = S3_BUCKET
    ) -> dict[str, bytes]:
        """
        Downloads multiple files from the specified S3 bucket using the given keys,
        `MAX_CONCURRENT_DOWNLOADS` at a time.

        Args:
            keys (list[str]): The S3 keys of the files to download.
            bucket (str): The S3 bucket to download from. Defaults to S3_BUCKET.
        Returns:
            dict[str, bytes]: A dictionary mapping S3 keys to their downloaded file bytes,
                in the order of `keys`.
        Raises:
            The error of the first key, in the order of `keys`, whose download failed.
        """
        unique_keys = list(dict.fromkeys(keys))
        if not unique_keys:
            return {}
        slots = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
        async with S3Client._client() as s3_client:

            async def download(key: str) -> bytes:
                async with slots:
                    return await _get_object_bytes(s3_client, bucket, key)

            # Every download finishes before the client can close, failed or not.
            results = await asyncio.gather(
                *(download(key) for key in unique_keys), return_exceptions=True
            )
        downloaded: dict[str, bytes] = {}
        for key, result in zip(unique_keys, results, strict=True):
            if isinstance(result, BaseException):
                raise result
            downloaded[key] = result
        return downloaded

    @staticmethod
    async def delete_object(key: str, bucket: str = S3_BUCKET) -> None:
        """
        Deletes an object from the specified S3 bucket using the given key.

        Args:
            key (str): The S3 key of the file to delete.
            bucket (str): The S3 bucket to delete from. Defaults to S3_BUCKET.
        """
        async with S3Client._client() as s3_client:
            await s3_client.delete_object(Bucket=bucket, Key=key)

    @staticmethod
    async def bulk_delete_objects(keys: list[str], bucket: str = S3_BUCKET) -> None:
        """
        Deletes multiple objects from the specified S3 bucket using the given keys.

        Args:
            keys (list[str]): The S3 keys of the files to delete.
            bucket (str): The S3 bucket to delete from. Defaults to S3_BUCKET.
        """
        objects = [{"Key": key} for key in keys]
        async with S3Client._client() as s3_client:
            await s3_client.delete_objects(Bucket=bucket, Delete={"Objects": objects})


async def _get_object_bytes(s3_client: Any, bucket: str, key: str) -> bytes:
    obj = await s3_client.get_object(Bucket=bucket, Key=key)
    return await obj["Body"].read()
