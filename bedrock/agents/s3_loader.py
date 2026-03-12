# agents/s3_loader.py

import os
import boto3
from typing import List

from config import AWS_REGION, S3_BUCKET, S3_PREFIX


# ---------------------------------------------------------
# S3 Client
# ---------------------------------------------------------
s3 = boto3.client("s3", region_name=AWS_REGION)


# ---------------------------------------------------------
# Download PDFs from S3
# ---------------------------------------------------------
def download_pdfs(local_dir: str = "data") -> List[str]:
    """
    Download all PDF files from the configured S3 bucket/prefix
    into a local directory.

    Args:
        local_dir (str): Local directory where PDFs will be stored.

    Returns:
        List[str]: List of local file paths to downloaded PDFs.
    """

    os.makedirs(local_dir, exist_ok=True)

    response = s3.list_objects_v2(
        Bucket=S3_BUCKET,
        Prefix=S3_PREFIX
    )

    downloaded_files = []

    for obj in response.get("Contents", []):
        key = obj["Key"]

        # Only process PDFs
        if not key.lower().endswith(".pdf"):
            continue

        filename = os.path.basename(key)
        local_path = os.path.join(local_dir, filename)

        # Download file
        s3.download_file(
            Bucket=S3_BUCKET,
            Key=key,
            Filename=local_path
        )

        downloaded_files.append(local_path)

    return downloaded_files
