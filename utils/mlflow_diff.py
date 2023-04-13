"""
Generates the html diff among two files on mlflow
"""

import difflib
from argparse import ArgumentParser
from pathlib import Path

import mlflow
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME


def save_html_diff(file1: Path, name1: str, file2: Path, name2: str, output: Path):
    lines1 = open(file1).readlines()
    lines2 = open(file2).readlines()
    delta = difflib.HtmlDiff().make_file(lines1, lines2, name1, name2)
    with open(output, "w") as f:
        f.write(delta)


if __name__ == "__main__":
    parser = ArgumentParser("HTML Diff among MLFlow files")
    parser.add_argument("uuid_path_1", type=str, help="<uuid>/<path>")
    parser.add_argument("uuid_path_2", type=str, help="<uuid>/<path>")
    parser.add_argument("--out", "-o", type=Path, default=Path.cwd() / "diff.html")
    parser.add_argument("--tracking-uri", type=str, help="MLFlow tracking uri")
    parser.add_argument("--registry-uri", type=str, help="MLFlow registry uri")
    args = parser.parse_args()

    client = mlflow.tracking.MlflowClient(
        tracking_uri=args.tracking_uri, registry_uri=args.registry_uri
    )

    uuid_1 = args.uuid_path_1.split("/")[0]
    run_1 = client.get_run(uuid_1)
    name_1 = run_1.data.tags.get(MLFLOW_RUN_NAME, uuid_1)
    path_1 = "/".join(args.uuid_path_1.split("/")[1:])

    uuid_2 = args.uuid_path_2.split("/")[0]
    run_2 = client.get_run(uuid_2)
    name_2 = run_2.data.tags.get(MLFLOW_RUN_NAME, uuid_2)
    path_2 = "/".join(args.uuid_path_2.split("/")[1:])

    file_1 = Path(client.download_artifacts(uuid_1, path_1))
    file_2 = Path(client.download_artifacts(uuid_2, path_2))
    save_html_diff(
        file_1, name_1 + "/" + file_1.name, file_2, name_2 + "/" + file_2.name, args.out
    )
