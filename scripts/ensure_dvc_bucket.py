#!/usr/bin/env python3
"""Create the configured bucket in the local S3-compatible DVC store."""

import argparse
import asyncio
import os

from dotenv import dotenv_values


async def create_bucket(bucket: str, endpoint: str, access_key: str, secret_key: str) -> None:
    from aiobotocore.session import get_session
    from botocore.exceptions import ClientError

    session = get_session()
    async with session.create_client(
        "s3",
        endpoint_url=endpoint,
        region_name="us-east-1",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    ) as client:
        try:
            await client.create_bucket(Bucket=bucket)
            print(f"Created bucket '{bucket}' at {endpoint}.")
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code in {"BucketAlreadyExists", "BucketAlreadyOwnedByYou"}:
                print(f"Bucket '{bucket}' already exists at {endpoint}.")
                return
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args()
    values = dotenv_values(args.env_file)
    bucket = os.environ.get("DVC_BUCKET") or values.get("DVC_BUCKET", "vehicle-classifier-data")
    endpoint = os.environ.get("DVC_ENDPOINT_URL") or values.get("DVC_ENDPOINT_URL", "http://127.0.0.1:9000")
    access_key = values.get("DVC_ACCESS_KEY")
    secret_key = values.get("DVC_SECRET_KEY")
    if not access_key or not secret_key:
        raise SystemExit(f"Missing DVC_ACCESS_KEY or DVC_SECRET_KEY in {args.env_file}")
    asyncio.run(create_bucket(bucket, endpoint, access_key, secret_key))


if __name__ == "__main__":
    main()
