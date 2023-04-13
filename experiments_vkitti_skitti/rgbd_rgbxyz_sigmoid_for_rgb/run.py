"""
Code to run a specific experiment
"""

import shutil
import sys
from functools import partial
from pathlib import Path

import torch

lib_path = Path(__file__).parent.parent.parent
sys.path.append(str(lib_path))

import logging
import warnings

import hydra
import pytorch_lightning as pl
import urllib3
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from train import TrainModel

from lib import logging as runlog
from lib.dataset import load_datamodule
from lib.losses import Loss
from lib.optimizers import Optimizer

# TRAIN

code_pos = Path(__file__).parent


@hydra.main(version_base=None, config_path=code_pos / "config", config_name="config")
def main(cfg: DictConfig):
    seed_everything(42, workers=True)
    # settings
    urllib3.disable_warnings()
    warnings.simplefilter("ignore", UserWarning)
    logging.getLogger().setLevel(logging.WARNING)

    # logging
    resume = cfg.run.get("resume", False)
    logger = runlog.create_logger(cfg, code_pos / "run_id.yaml")
    log_hyperparams = partial(runlog.log_hyperparams, disable=resume)
    runlog.save_logger_ids(logger, code_pos / "run_id.yaml")

    # data
    if cfg.run.params.batch_size % cfg.run.runtime.gpus != 0:
        raise RuntimeError("Batch size must be a multiple of the number of used gpus")
    batch_size = cfg.run.params.batch_size // cfg.run.runtime.gpus

    dm = load_datamodule(
        cfg.datasets.name,
        cfg.datasets.DATASET_SOURCE,
        cfg.datasets.DATASET_TARGET,
        batch_size=batch_size,
        max_iterations=cfg.run.params.max_iterations if cfg.run.type == "train" else 1,
        ds_args=cfg.datasets.specific,
        augmentations=cfg.datasets.augmentations,
        short_run=cfg.train.short_run if cfg.run.type == "train" else False,
        reduce_factor=cfg.train.reduce_factor,
    )
    dm.prepare_data()
    dm.setup()

    if cfg.run.type == "train":
        log_hyperparams(
            logger,
            dataset=cfg.datasets.name,
            batch_size=cfg.run.params.batch_size,
            train_splits=cfg.datasets.DATASET_SOURCE,
            val_splits=cfg.datasets.DATASET_TARGET,
            augmenting=cfg.datasets.augmentations,
        )
    else:
        log_hyperparams(
            logger,
            dataset=cfg.datasets.name,
            batch_size=batch_size,
            test_splits=cfg.datasets.DATASET_TARGET,
        )

    # model(s)
    module_names, model_args = [], {}
    optimizers = {}
    for model_cfg in cfg.models:
        mod_name = model_cfg.name
        module_names.append(mod_name)
        model_args[mod_name] = model_cfg.get("args", {})
        optimizers[mod_name] = Optimizer(**model_cfg.optimizer)
        if "lr_scheduler" in model_cfg:
            optimizers[mod_name].set_scheduler(**model_cfg.lr_scheduler)

        if len(cfg.models) > 1:
            log_hyperparams(logger, model_cfg, prefix=f"model/{mod_name}")
            if cfg.run.type == "train":
                log_hyperparams(
                    logger, model_cfg.optimizer, prefix=f"optimizer/{mod_name}"
                )
                if "lr_scheduler" in model_cfg:
                    log_hyperparams(
                        logger,
                        model_cfg.lr_scheduler,
                        prefix=f"lr_scheduler/{mod_name}",
                    )

    if len(cfg.models) == 1:
        module_names = mod_name
        model_args = model_args[mod_name]
        optimizers = optimizers[mod_name]

        log_hyperparams(logger, model_args, prefix="model")
        if cfg.run.type == "train":
            log_hyperparams(logger, model_cfg.optimizer, prefix="optimizer")
            if "lr_scheduler" in model_cfg:
                log_hyperparams(logger, model_cfg.lr_scheduler, prefix="lr_scheduler")

    # # loss
    loss = Loss(OmegaConf.to_container(cfg.losses, resolve=True))
    log_hyperparams(logger, loss.split_by_target(), prefix="loss")

    # # train
    train_args = OmegaConf.to_container(cfg.train, resolve=True)
    for name in train_args.get("to_log", []):
        value = train_args["params"][name]
        log_hyperparams(logger, {name: value})

    if resume:
        model = TrainModel.load_from_checkpoint(
            code_pos / "ckpts/last.ckpt",
            model_modules=module_names,
            loss=loss,
            optimizer=optimizers,
            train_kwargs=train_args["params"],
            model_kwargs=model_args,
        )
    else:
        model = TrainModel(
            model_modules=module_names,
            loss=loss,
            optimizer=optimizers,
            train_kwargs=train_args["params"],
            model_kwargs=model_args,
        )

    # if cfg.run.type == "test":
    #     for model_cfg in cfg.models:
    #         if len(cfg.models) > 1:
    #             model.model[model_cfg.name].model.load_state_dict(
    #                 mlflow.pytorch.load_model(
    #                     str(code_pos / f"mlflow-model/{model_cfg.name}")
    #                 ).state_dict()
    #             )
    #         else:
    #             # yeah.. TrainModel -> ModelWrapper -> real model :)
    #             model.model.model.load_state_dict(
    #                 mlflow.pytorch.load_model(
    #                     str(code_pos / "mlflow-model")
    #                 ).state_dict()
    #             )

    ckpts_2d = ModelCheckpoint(
        code_pos / "ckpts",
        filename="{epoch}-{step}_2d",
        monitor=cfg.metrics.ckpt_monitor_2d,
        mode=cfg.metrics.get("mode_monitor", "min"),
        save_top_k=1,
        save_last=True,
    )

    ckpts_3d = ModelCheckpoint(
        code_pos / "ckpts",
        filename="{epoch}-{step}_3d",
        monitor=cfg.metrics.ckpt_monitor_3d,
        mode=cfg.metrics.get("mode_monitor", "min"),
        save_top_k=1,
        save_last=True,
    )

    # log artifacts
    if cfg.run.type == "train":
        runlog.log_artifacts(
            logger,
            [
                code_pos / "train.py",
                code_pos / "run.py",
                code_pos / "config",
                code_pos / "run_id.yaml",
            ]
            + [p.parent for p in code_pos.glob("**/__init__.py")],
        )
        runlog.log_config(logger, "repeat_run_config.yaml", cfg)

    class TrainCallback(pl.Callback):
        def on_train_start(
            self, trainer: pl.Trainer, pl_module: pl.LightningDataModule
        ):
            if pl_module.global_rank == 0:
                # check torchscript
                for module_cfg in cfg.models:
                    try:
                        script = module_cfg.artifacts.script_model
                        if len(cfg.models) > 1:
                            model.model[model_cfg.name].get_model(script)
                        else:
                            model.model.get_model(script)
                    except Exception:
                        print(
                            f"Error scripting model {model_cfg.name}, either disable scripting or fix the model"
                        )
                        sys.exit(1)

        # def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        #     if pl_module.global_rank == 0:

        #         # for branch, ckpts in zip(["2d", "3d"], [ckpts_2d, ckpts_3d]):
        #         best_model = TrainModel(
        #             module_names,
        #             model_kwargs=model_args,
        #             train_kwargs=train_args["params"],
        #         )

        #         shutil.rmtree(str(code_pos / "mlflow-model"), ignore_errors=True)
        #         for model_cfg, ckpts in zip(cfg.models, [ckpts_2d, ckpts_3d]):
        #             name = model_cfg.name

        #             # save the final model(s)

        #             best_model.load_state_dict(
        #                 torch.load(ckpts.best_model_path)["state_dict"]
        #             )

                    # if len(cfg.models) > 1:
                    #     model = best_model.model[name]
                    #     save_path = str(code_pos / "mlflow-model" / (str(name)))
                    # else:
                    #     model = best_model.model
                    #     save_path = str(code_pos / "mlflow-model")

                    # sign = infer_signature(*model.signature)
                    # env = mlflow.pytorch.get_default_pip_requirements()
                    # env.extend(model.dependencies)

                    # model_to_save = model.get_model(model_cfg.artifacts.script_model)
                    # mlflow.pytorch.save_model(
                    #     model_to_save,
                    #     save_path,
                    #     pip_requirements=env,
                    #     signature=sign,
                    #     code_paths=[code_pos / name],
                    # )
                    # logger["mlflow"].experiment.log_artifact(
                    #     logger["mlflow"].run_id, str(code_pos / "mlflow-model")
                    # )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    # train
    trainer = pl.Trainer(
        gpus=cfg.run.runtime.gpus,
        strategy=DDPStrategy(
            find_unused_parameters=cfg.run.runtime.get("find_unused_parameters", True)
        )
        if cfg.run.runtime.gpus > 1
        else None,
        precision=cfg.run.runtime.precision,
        benchmark=True,
        logger=list(logger.values()) if logger else False,
        callbacks=[TrainCallback(), ckpts_2d, ckpts_3d, lr_monitor],
        max_epochs=cfg.run.params.max_iterations if cfg.run.type == "train" else 1,
        check_val_every_n_epoch=cfg.run.params.checkpoint_period
        if cfg.run.type == "train"
        else 1,
        # max_steps=cfg.run.params.max_iterations,
        # val_check_interval=cfg.run.params.checkpoint_period,
        log_every_n_steps=100,
        multiple_trainloader_mode=cfg.datasets.get(
            "multiple_trainloader_mode", "max_size_cycle"
        ),
        # debug
        limit_train_batches=cfg.run.params.get("limit_train_batches", None),
        limit_val_batches=cfg.run.params.get("limit_val_batches", None),
        limit_test_batches=cfg.run.params.get("limit_test_batches", None),
        fast_dev_run=cfg.run.params.get("fast_dev_run", False),
    )

    # if cfg.run.type == "train":
    dl_train = dm.train_dataloader()
    dl_val = dm.val_dataloader()
    trainer.fit(
        model,
        train_dataloaders=dl_train,
        val_dataloaders=dl_val,
        ckpt_path=str(code_pos / "ckpts/last.ckpt") if resume else None,
        )
    # else:
    dl_test = dm.test_dataloader()
    trainer.test(model, dataloaders=dl_test)


if __name__ == "__main__":
    main()
