"""
Downloads from mlflow an experiment
"""

import argparse
from pathlib import Path

import urllib3
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME

if __name__ == "__main__":

    urllib3.disable_warnings()

    # arguments parser
    parser = argparse.ArgumentParser(
        "MLFlow Tracking downloader, downloads an MLFlow run provided its uuid"
    )
    parser.add_argument("uuid", type=str, help="uuid of the run to download")
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        default=Path.cwd() / "old_experiments",
        help="folder in which save the run data",
    )
    parser.add_argument(
        "--by-name", action="store_true", help="use the name of the run instead of UUID"
    )
    parser.add_argument("--tracking-uri", type=str, help="MLFlow tracking uri")
    parser.add_argument("--registry-uri", type=str, help="MLFlow registry uri")
    args = parser.parse_args()

    # download
    client = MlflowClient(
        tracking_uri=args.tracking_uri, registry_uri=args.registry_uri
    )

    run_name = args.uuid
    if args.by_name:
        run = client.get_run(args.uuid)
        run_name = run.data.tags.get(MLFLOW_RUN_NAME, run_name)

    out_path: Path = args.out / run_name if args.out.exists() else args.out
    out_path.mkdir(exist_ok=True, parents=True)
    client.download_artifacts(args.uuid, ".", out_path)
