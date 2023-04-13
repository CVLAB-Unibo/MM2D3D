import argparse
import mlflow
import os
from contextlib import contextmanager
import logging
import urllib3

from typing import Dict, List, Optional, Tuple, Union
from mlflow.entities import Run
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID

# functions

_logger = logging.getLogger("mlflow_copy_run")


@contextmanager
def credentials(auth: str = None):
    old_user = os.environ.get("MLFLOW_TRACKING_USERNAME", None)
    old_pwd = os.environ.get("MLFLOW_TRACKING_PASSWORD", None)
    try:
        if auth is not None:
            user, password = auth.split("@")
            os.environ["MLFLOW_TRACKING_USERNAME"] = user
            os.environ["MLFLOW_TRACKING_PASSWORD"] = password
        yield None
    finally:
        os.environ.pop("MLFLOW_TRACKING_USERNAME", None)
        os.environ.pop("MLFLOW_TRACKING_PASSWORD", None)
        if old_user:
            os.environ["MLFLOW_TRACKING_USERNAME"] = old_user
        if old_pwd:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = old_pwd


def _copy_run(
    run_id: str,
    exp_name: str,
    url_from: Optional[str] = None,
    url_from_auth: Optional[str] = None,
    url_to: Optional[str] = None,
    url_to_auth: Optional[str] = None,
    new_tags: Optional[Dict] = None,
) -> Tuple[Run, Run]:

    # copy run locally
    _logger.info(f"Copying RUN {run_id} from {url_from}")
    with credentials(url_from_auth):
        mlflow.set_tracking_uri(url_from)
        client_from = mlflow.tracking.MlflowClient(tracking_uri=url_from)
        run_from = client_from.get_run(run_id)
        _logger.info(f"Copied RUN {run_id} INFO")
        metrics = {
            name: client_from.get_metric_history(run_id, name)
            for name in run_from.data.metrics
        }
        _logger.info(f"Copied RUN {run_id} METRICS")
        artifacts_path = client_from.download_artifacts(run_id, ".")
        _logger.info(f"Copied RUN {run_id} ARTIFACTS")

    # copy it back
    with credentials(url_to_auth):
        mlflow.set_tracking_uri(url_to)
        client_to = mlflow.tracking.MlflowClient(tracking_uri=url_to)

        # create experiment if it does not exist
        exp_to = client_to.get_experiment_by_name(exp_name)
        if not exp_to:
            exp_id = client_to.create_experiment(exp_name)
        else:
            exp_id = exp_to.experiment_id

        # copy run
        tags = dict(**run_from.data.tags)
        if new_tags:
            tags.update(new_tags)
        run_to = client_to.create_run(
            exp_id, start_time=run_from.info.start_time, tags=tags
        )
        for name, value in run_from.data.params.items():
            client_to.log_param(run_to.info.run_id, name, value)
        for name, values in metrics.items():
            for metric in values:
                client_to.log_metric(
                    run_to.info.run_id,
                    name,
                    metric.value,
                    metric.timestamp,
                    metric.step,
                )
        client_to.log_artifacts(run_to.info.run_id, artifacts_path)
        client_to.set_terminated(
            run_to.info.run_id, run_from.info.status, run_from.info.end_time
        )
        _logger.info(f"Copied RUN into {run_to.info.run_id} in {url_to}")

    return run_from, run_to


def find_childrens(
    run_id: str,
    exp_id: str,
    tracking_url: Optional[str] = None,
    tracking_url_auth: Optional[str] = None,
) -> List[Run]:
    with credentials(tracking_url_auth):
        client = mlflow.tracking.MlflowClient(tracking_url)
        run_ids = [run.run_id for run in client.list_run_infos(exp_id)]
        runs = [client.get_run(run_id) for run_id in run_ids]
        return [
            run for run in runs
            if run.data.tags.get(MLFLOW_PARENT_RUN_ID, None) == run_id
        ]


def copy_run(
    run_id: str,
    exp_name: str,
    url_from: Optional[str] = None,
    url_from_auth: Optional[str] = None,
    url_to: Optional[str] = None,
    url_to_auth: Optional[str] = None,
    recurse: bool = False,
) -> List[Tuple[Run, Run]]:
    if not recurse:
        run_from, run_to = _copy_run(
            run_id, exp_name, url_from, url_from_auth, url_to, url_to_auth
        )
        return [(run_from, run_to)]
    else:
        run_from, run_to = _copy_run(
            run_id, exp_name, url_from, url_from_auth, url_to, url_to_auth
        )
        all_runs = [(run_from, run_to)]

        childrens = find_childrens(
            run_from.info.run_id, run_from.info.experiment_id, url_from, url_from_auth
        )

        for child in childrens:
            to_log = _copy_run(
                child.info.run_id,
                exp_name,
                url_from,
                url_from_auth,
                url_to,
                url_to_auth,
                {MLFLOW_PARENT_RUN_ID: run_to.info.run_id},
            )
            all_runs.append(to_log)
        return all_runs


if __name__ == "__main__":

    # parsing
    parser = argparse.ArgumentParser(
        "Copy of run from and experiment to another even on a different tracking server"
    )
    parser.add_argument("run", type=str, help="original RUN ID")
    parser.add_argument("exp", type=str, help="experiment in which copy the run")
    parser.add_argument("--url-from", type=str, default=None, help="tracking url from")
    parser.add_argument(
        "--url-from-auth", type=str, default=None, help="credentials url from"
    )
    parser.add_argument("--url-to", type=str, default=None, help="tracking url to")
    parser.add_argument(
        "--url-to-auth", type=str, default=None, help="credentials url to"
    )
    parser.add_argument(
        "--recurse", action="store_true", help="copy also children runs"
    )
    parser.add_argument("--verbose", action="store_true", help="if print logs")

    args = parser.parse_args()

    # activate logs
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    # remove warnings when INSECURE TLS
    if os.environ.get("MLFLOW_TRACKING_INSECURE_TLS", False):
        urllib3.disable_warnings()

    copied = copy_run(
        args.run,
        args.exp,
        args.url_from,
        args.url_from_auth,
        args.url_to,
        args.url_to_auth,
        args.recurse,
    )

    for run_from, run_to in copied:
        print(f"COPIED: {run_from.info.run_id} TO {run_to.info.run_id}")
