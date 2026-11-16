"""Prepare a fresh environment: create the tables and the S3 bucket, then seed the database.

Safe to rerun: existing tables, the bucket and seeded rows are kept.
"""

import asyncio

from database.general import create_db_and_tables, populate_db
from s3.utils import S3Client
from settings import S3_BUCKET


async def main() -> None:
    await create_db_and_tables()
    print("[populate] Tables ready.")
    created = await S3Client.ensure_bucket()
    print(f"[populate] S3 bucket '{S3_BUCKET}' {'created' if created else 'ready'}.")
    await populate_db()


if __name__ == "__main__":
    asyncio.run(main())
