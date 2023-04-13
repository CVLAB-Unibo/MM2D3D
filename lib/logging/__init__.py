import shutil
import tempfile
from argparse import Namespace
from codecs import ignore_errors
from pathlib import Path
from typing import Annotated, Any

import yaml
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID, MLFLOW_RUN_NOTE
from omegaconf import OmegaConf
from pytorch_lightning.loggers import LightningLoggerBase, MLFlowLogger, WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only

# exported functions

__all__ = [
    "create_logger",
    "save_logger_ids",
    "log_artifacts",
    "log_config",
    "log_hyperparams",
]

# logging handling

LoggerDef = Annotated[
    dict[str, LightningLoggerBase], "single logger, multiple loggers or none"
]


def create_logger(cfg: OmegaConf, id_file_path: Path | str) -> LoggerDef:
    """
    Factory to generate the used loggers
    """
    loggers = {}
    resume = cfg.run.get("resume", False)

    if "mlflow" in cfg.logging:
        tags = OmegaConf.to_container(cfg.logging.mlflow.tags, resolve=True)
        if "note" in cfg.logging:
            tags[MLFLOW_RUN_NOTE] = cfg.logging.mlflow.note
        tags["type"] = cfg.run.type
        if cfg.run.type == "test":
            with open(id_file_path, "rt") as f:
                tags[MLFLOW_PARENT_RUN_ID] = yaml.safe_load(f)["mlflow_id"]
        loggers["mlflow"] = MLFlowLogger(
            cfg.logging.mlflow.experiment_name,
            cfg.logging.mlflow.run_name,
            save_dir="mlruns",
            tags=tags,
        )
        if resume and (run_id := yaml.safe_load(open(id_file_path, "rt"))["mlflow_id"]):
            loggers["mlflow"]._run_id = run_id

    if "wandb" in cfg.logging:
        tags = cfg.logging.wandb.get("tags", [])
        tags.append(cfg.run.type)

        wandb_id = None
        if "mlflow" in cfg.logging:
            wandb_id = loggers["mlflow"].run_id
        if resume:
            with open(id_file_path, "rt") as f:
                wandb_id = yaml.safe_load(f)["wandb_id"]

        loggers["wandb"] = WandbLogger(
            name=cfg.logging.wandb.run_name,
            project=cfg.logging.wandb.project,
            tags=tags,
            resume="allow" if resume else False,
            id=wandb_id,
        )
    return loggers


@rank_zero_only
def save_logger_ids(logger: LoggerDef, id_file_path: Path | str):
    with open(id_file_path, "wt") as f:
        yaml.safe_dump(
            {
                "mlflow_id": logger["mlflow"].run_id if "mlflow" in logger else None,
                "wandb_id": logger["wandb"].version if "wandb" in logger else None,
            },
            f,
        )


@rank_zero_only
def log_artifacts(logger: LoggerDef, to_log: list[Path | str], **kwargs):
    if "mlflow" in logger:
        for path_to_log in to_log:
            logger["mlflow"].experiment.log_artifact(
                logger["mlflow"].run_id, path_to_log
            )

    if "wandb" in logger:
        for path_to_log in to_log:

            if (pl := Path(path_to_log)).is_dir():
                base_path = str(pl.parent)
                for plf in pl.glob("**/*"):
                    logger["wandb"].experiment.save(
                        str(plf),
                        base_path=base_path,
                        policy=kwargs.get("policy", "now"),
                    )
            else:
                logger["wandb"].experiment.save(
                    str(path_to_log),
                    policy=kwargs.get("policy", "now"),
                )


@rank_zero_only
def log_config(logger: LoggerDef, file_name: str, cfg: OmegaConf):
    if "mlflow" in logger:
        logger["mlflow"].experiment.log_text(
            logger["mlflow"].run_id, OmegaConf.to_yaml(cfg), file_name
        )

    if "wandb" in logger:
        tmpfile = Path(tempfile.mkdtemp()) / file_name
        with open(tmpfile, "wt") as f:
            f.write(OmegaConf.to_yaml(cfg))
        logger["wandb"].experiment.save(str(tmpfile), policy="now")


@rank_zero_only
def log_hyperparams(
    loggers: LoggerDef,
    /,
    hparams: dict[str, Any] | Namespace = {},
    *,
    disable: bool = False,
    prefix: str | None = None,
    **kwargs,
):
    if not disable:
        for logger in loggers.values():
            if kwargs:
                hparams = hparams | kwargs
            if prefix:
                hparams = {f"{prefix}/{k}": v for k, v in hparams.items()}
            logger.log_hyperparams(hparams)
