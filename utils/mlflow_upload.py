"""
Upload a file or folder on an MLFlow run
"""

import argparse
from pathlib import Path

import urllib3
from mlflow.tracking import MlflowClient

if __name__ == "__main__":
    parser = argparse.ArgumentParser("MLFlow Upload file")
    parser.add_argument("uuid", type=str, help="uuid of the run to download")
    parser.add_argument("path", type=Path, help="local path to upload")
    parser.add_argument("--dir-to", type=Path, default=None, help="remote directory")
    parser.add_argument("--tracking-uri", type=str, help="MLFlow tracking uri")
    parser.add_argument("--registry-uri", type=str, help="MLFlow registry uri")
    args = parser.parse_args()

    urllib3.disable_warnings()

    client = MlflowClient(
        tracking_uri=args.tracking_uri, registry_uri=args.registry_uri
    )
    client.log_artifact(args.uuid, args.path, args.dir_to)
