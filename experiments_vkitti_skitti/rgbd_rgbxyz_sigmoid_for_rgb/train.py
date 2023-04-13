import imp
import importlib
import inspect
from statistics import median
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import ChainedScheduler
from torchmetrics import JaccardIndex

from lib.losses import Loss
from lib.optimizers import Optimizer
from lib.utils.visualize import draw_points_image_labels_with_confidence


class TrainModel(pl.LightningModule):
    def __init__(
        self,
        # models
        # a single module name or multiple names
        # (a single loss and optimizer for a single model or a dict name, value
        # for multiple models)
        model_modules: str | list[str] = None,
        optimizer: Optimizer | dict[str, Optimizer] | None = None,
        loss: Loss | None = None,
        # training params
        train_kwargs={},
        # model params
        model_kwargs={},
    ):
        super().__init__()
        # params
        self.train_log_step = train_kwargs.get("train_log_images_step", 2000)
        self.val_log_step = train_kwargs.get("val_log_images_step", 500)
        self.test_log_step = train_kwargs.get("test_log_images_step", 500)
        self.lambda_xm_src = train_kwargs.get("lambda_xm_src", 0.8)
        self.lambda_xm_trg = train_kwargs.get("lambda_xm_trg", 0.1)
        # model info
        self.num_classes = (
            model_kwargs["num_classes"]
            if model_modules is isinstance(model_modules, str)
            else model_kwargs[model_modules[0]]["num_classes"]
        )
        self.class_names = train_kwargs["class_names"]
        self.rgb_palette = np.array(train_kwargs["class_palette"])

        # load models
        self.loss = loss
        self.modules_name = model_modules
        self.model = _load_models(model_modules, optimizer, loss, **model_kwargs)
        model_modules[0]

        self.best_source_iou = 0
        self.best_target_iou = 0
        self.best_source_iou_3d = 0
        self.best_target_iou_3d = 0
        self.best_source_iou_avg = 0
        self.best_target_iou_avg = 0

        # metrics
        self.segm_iou_train = JaccardIndex(
            num_classes=self.num_classes, average="none"
        ).to(self.device)
        self.segm_iou_train.reset()
        self.segm_iou_val_source = JaccardIndex(
            num_classes=self.num_classes, average="none"
        ).to(self.device)
        self.segm_iou_val_source.reset()
        self.segm_iou_val_target = JaccardIndex(
            num_classes=self.num_classes, average="none"
        ).to(self.device)
        self.segm_iou_val_target.reset()
        self.segm_iou_test_target = JaccardIndex(
            num_classes=self.num_classes, average="none"
        ).to(self.device)
        self.segm_iou_test_target.reset()

        self.segm_ious_splits_2d = {
            "train": self.segm_iou_train,
            "val/source": self.segm_iou_val_source,
            "val/target": self.segm_iou_val_target,
            "test/target": self.segm_iou_test_target,
        }

        self.segm_iou_train_3d = JaccardIndex(
            num_classes=self.num_classes, average="none"
        ).to(self.device)
        self.segm_iou_train_3d.reset()
        self.segm_iou_val_source_3d = JaccardIndex(
            num_classes=self.num_classes, average="none"
        ).to(self.device)
        self.segm_iou_val_source_3d.reset()
        self.segm_iou_val_target_3d = JaccardIndex(
            num_classes=self.num_classes, average="none"
        ).to(self.device)
        self.segm_iou_val_target_3d.reset()
        self.segm_iou_test_target_3d = JaccardIndex(
            num_classes=self.num_classes, average="none"
        ).to(self.device)
        self.segm_iou_test_target_3d.reset()

        self.segm_ious_splits_3d = {
            "train": self.segm_iou_train_3d,
            "val/source": self.segm_iou_val_source_3d,
            "val/target": self.segm_iou_val_target_3d,
            "test/target": self.segm_iou_test_target_3d,
        }

        self.segm_iou_train_avg = JaccardIndex(
            num_classes=self.num_classes, average="none"
        ).to(self.device)
        self.segm_iou_train_avg.reset()
        self.segm_iou_val_source_avg = JaccardIndex(
            num_classes=self.num_classes, average="none"
        ).to(self.device)
        self.segm_iou_val_source_avg.reset()
        self.segm_iou_val_target_avg = JaccardIndex(
            num_classes=self.num_classes, average="none"
        ).to(self.device)
        self.segm_iou_val_target_avg.reset()
        self.segm_iou_test_target_avg = JaccardIndex(
            num_classes=self.num_classes, average="none"
        ).to(self.device)
        self.segm_iou_test_target_avg.reset()

        self.segm_ious_splits_avg = {
            "train": self.segm_iou_train_avg,
            "val/source": self.segm_iou_val_source_avg,
            "val/target": self.segm_iou_val_target_avg,
            "test/target": self.segm_iou_test_target_avg,
        }
        self.l1_loss_fn = torch.nn.L1Loss(reduce=True, size_average=True)

    def configure_optimizers(self):
        if isinstance(self.model, nn.ModuleDict):
            optimizers, schedulers = zip(
                *[m.build_optimizer() for m in self.model.values()]
            )
            # return list(optimizers), [sc for sc in schedulers if sc is not None]
            hoptim = HybridOptim(optimizers)
            hscheduler1 = HybridLRS(hoptim, 0, schedulers[0])
            hscheduler2 = HybridLRS(hoptim, 1, schedulers[1])
            scheduler = ChainedScheduler([hscheduler1, hscheduler2])

            return [hoptim], [{"scheduler": scheduler, "interval": "step"}]
        else:
            optimizer, scheduler = self.model.build_optimizer()
            return [optimizer], [scheduler] if scheduler is not None else []

    def forward(self, *args, model_name: str | None = None, **kwargs):
        return _dispatch(model_name, self.model)(*args, **kwargs)

    def cross_modal_loss(
        self,
        gt_for_2d,
        prediction_avg,
        gt_for_3d,
        prediction_3d,
    ):
        loss_avg = (
            F.kl_div(
                F.log_softmax(prediction_avg, dim=1),
                F.softmax(gt_for_2d.detach(), dim=1),
                reduction="none",
            )
            .sum(1)
            .mean()
        )

        loss_3d = (
            F.kl_div(
                F.log_softmax(prediction_3d, dim=1),
                F.softmax(gt_for_3d.detach(), dim=1),
                reduction="none",
            )
            .sum(1)
            .mean()
        )

        return loss_avg, loss_3d

    def _generic_step(self, batch, stage, step=None, log_step=1000):
        data_batch_src = batch["source"]
        data_batch_trg = batch["target"]
        loss_2d = []
        loss_3d = []

        preds_2d_fe, out_2D_feature, img_indices, preds_2d_be = self(
            data_batch_src, model_name=self.modules_name[0]
        )
        preds_3d_fe, _, preds_3d_be = self(
            data_batch_src, model_name=self.modules_name[1]
        )

        seg_loss_src_2d = self.loss(
            "segmentation",
            pred=preds_2d_fe["seg_logit"],
            gt=data_batch_src["seg_label"],
        )

        seg_loss_src_3d = self.loss(
            "segmentation",
            pred=preds_3d_fe["seg_logit"],
            gt=data_batch_src["seg_label"],
        )
        loss_3d.append(seg_loss_src_3d)
        loss_2d.append(seg_loss_src_2d)

        seg_logit_2d_avg = preds_2d_be["seg_logit_avg"]
        seg_logit_3d_point = preds_3d_be["seg_logit_point"]

        xm_loss_src_2d, xm_loss_src_3d = self.cross_modal_loss(
            preds_3d_fe["seg_logit"],
            seg_logit_2d_avg,
            preds_2d_fe["seg_logit"],
            seg_logit_3d_point,
        )

        loss_2d.append(self.lambda_xm_src * xm_loss_src_2d)
        loss_3d.append(self.lambda_xm_src * xm_loss_src_3d)

        if self.global_step % 2000 == 0:
            num_points_first_pc = len(data_batch_src["img_indices"][0])
            draw_points_image_labels_with_confidence(
                data_batch_src["img"][0].permute(1, 2, 0).cpu().numpy(),
                preds_2d_fe["seg_logit_2d"].argmax(1)[0].cpu().numpy(),
                preds_2d_fe["seg_logit"][:num_points_first_pc].argmax(1).cpu().numpy(),
                preds_3d_fe["seg_logit"][:num_points_first_pc].argmax(1).cpu().numpy(),
                preds_3d_fe["confidence"][:num_points_first_pc].detach().cpu().numpy(),
                data_batch_src["seg_label"][:num_points_first_pc].cpu().numpy(),
                data_batch_src["img_indices"][0],
                color_palette=self.rgb_palette,
                stage=stage + "_source",
                current_epoch=self.current_epoch,
                logger=self.loggers[1],
                step=self.global_step,
            )

        ######### target domain optimization #########
        preds_2d_fe, out_2D_feature, img_indices, preds_2d_be = self(
            data_batch_trg, model_name=self.modules_name[0]
        )
        preds_3d_fe, out_3D_feature, preds_3d_be = self(
            data_batch_trg, model_name=self.modules_name[1]
        )
        seg_logit_2d_avg = preds_2d_be["seg_logit_avg"]
        seg_logit_3d_point = preds_3d_be["seg_logit_point"]

        (xm_loss_trg_2d, xm_loss_trg_3d,) = self.cross_modal_loss(
            preds_3d_fe["seg_logit"],
            seg_logit_2d_avg,
            preds_2d_fe["seg_logit"],
            seg_logit_3d_point,
        )

        if self.global_step % 2000 == 0:
            num_points_first_pc = len(data_batch_trg["img_indices"][0])
            draw_points_image_labels_with_confidence(
                data_batch_trg["img"][0].permute(1, 2, 0).cpu().numpy(),
                preds_2d_fe["seg_logit_2d"].argmax(1)[0].cpu().numpy(),
                preds_2d_fe["seg_logit"][:num_points_first_pc].argmax(1).cpu().numpy(),
                preds_3d_fe["seg_logit"][:num_points_first_pc].argmax(1).cpu().numpy(),
                preds_3d_fe["confidence"][:num_points_first_pc].detach().cpu().numpy(),
                data_batch_trg["seg_label"][:num_points_first_pc].cpu().numpy(),
                data_batch_trg["img_indices"][0],
                color_palette=self.rgb_palette,
                stage=stage + "_target",
                current_epoch=self.current_epoch,
                logger=self.loggers[1],
                step=self.global_step,
            )

        loss_2d.append(self.lambda_xm_trg * xm_loss_trg_2d)
        loss_3d.append(self.lambda_xm_trg * xm_loss_trg_3d)

        self.log_dict(
            {
                f"{stage}/loss_segmentation": seg_loss_src_2d,
                f"{stage}/loss_segmentation_3d": seg_loss_src_3d,
                f"{stage}/xm_loss_src_2d": xm_loss_src_2d,
                f"{stage}/xm_loss_tgt_2d": xm_loss_trg_2d,
                f"{stage}/xm_loss_src_3d": xm_loss_src_3d,
                f"{stage}/xm_loss_tgt_3d": xm_loss_trg_3d,
            },
            prog_bar=True,
            add_dataloader_idx=False,
        )
        return sum(loss_2d) + sum(loss_3d)

    def training_step(self, batch, batch_idx):
        return self._generic_step(batch, "train", step=self.global_step)

    def _generic_step_val(self, batch, stage, step=None, log_step=1000):
        self.model.eval()
        preds_seg_2d, _, _, _ = self(batch, model_name=self.modules_name[0])

        preds_seg_3d, _, _ = self(batch, model_name=self.modules_name[1])

        loss_2d = self.loss(
            "segmentation",
            pred=preds_seg_2d["seg_logit"],
            gt=batch["seg_label"],
        )

        loss_3d = self.loss(
            "segmentation",
            pred=preds_seg_3d["seg_logit"],
            gt=batch["seg_label"],
        )

        ensembl_pred = (
            F.softmax(preds_seg_2d["seg_logit"], dim=1)
            + F.softmax(preds_seg_3d["seg_logit"], dim=1)
        ) / 2
        self.segm_ious_splits_2d[stage](
            preds_seg_2d["seg_logit"].argmax(1)[batch["seg_label"] != -100],
            batch["seg_label"][batch["seg_label"] != -100],
        )
        self.segm_ious_splits_3d[stage](
            preds_seg_3d["seg_logit"].argmax(1)[batch["seg_label"] != -100],
            batch["seg_label"][batch["seg_label"] != -100],
        )
        self.segm_ious_splits_avg[stage](
            ensembl_pred.argmax(1)[batch["seg_label"] != -100],
            batch["seg_label"][batch["seg_label"] != -100],
        )

        self.log_dict(
            {
                f"{stage}/loss_segmentation": loss_2d,
                f"{stage}/loss_segmentation_3d": loss_3d,
            },
            prog_bar=True,
            add_dataloader_idx=False,
        )

        if step % log_step == 0:
            num_points_first_pc = len(batch["img_indices"][0])
            draw_points_image_labels_with_confidence(
                batch["img"][0].permute(1, 2, 0).cpu().numpy(),
                preds_seg_2d["seg_logit_2d"].argmax(1)[0].cpu().numpy(),
                preds_seg_2d["seg_logit"][:num_points_first_pc].argmax(1).cpu().numpy(),
                preds_seg_3d["seg_logit"][:num_points_first_pc].argmax(1).cpu().numpy(),
                preds_seg_3d["confidence"][:num_points_first_pc].detach().cpu().numpy(),
                batch["seg_label"][:num_points_first_pc].cpu().numpy(),
                batch["img_indices"][0],
                color_palette=self.rgb_palette,
                stage=stage,
                current_epoch=self.current_epoch,
                logger=self.loggers[1],
                step=self.global_step,
            )

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0:
            label = "val/target"
        else:
            label = "test/target"

        out = self._generic_step_val(
            batch,
            label,
            self._val_step,
            log_step=self.val_log_step,
        )
        self._val_step += 1
        return {"loss": out, "domain": label}

    def test_step(self, batch, _):
        label = "test/target"
        out = self._generic_step_val(
            batch,
            label,
            self._test_step,
            log_step=self.test_log_step,
        )
        self._test_step += 1
        return {"loss": out, "domain": label}

    # def on_train_epoch_start(self) -> None:
    # self._reset_metrics()

    def on_validation_start(self):
        # self._reset_metrics()
        self._val_step = 0

    def on_test_start(self) -> None:
        # self._reset_metrics()
        self._test_step = 0

    # def _reset_metrics(self):
    #     for _, m in self.segm_ious_splits_2d.items():
    #         m.reset()
    #     for _, m in self.segm_ious_splits_3d.items():
    #         m.reset()

    def _evaluation_end(self, stage):

        # 2D
        segm_iou = self.segm_ious_splits_2d[stage]
        # print(segm_iou.confmat)
        iou_2d = segm_iou.compute().mean()

        if "val" in stage or "test" in stage:
            print("2d")
            print(
                [
                    (class_name, iou.item())
                    for class_name, iou in zip(self.class_names, segm_iou.compute())
                ]
            )

        if stage == "val/source":
            if iou_2d > self.best_source_iou:
                self.log("best_source_iou", iou_2d)
                self.best_source_iou = iou_2d
        elif stage == "val/target":
            if iou_2d > self.best_target_iou:
                self.best_target_iou = iou_2d
                self.log("best_target_iou", iou_2d)

        # 3D
        segm_iou = self.segm_ious_splits_3d[stage]
        # print(segm_iou.confmat)
        iou_3d = segm_iou.compute().mean()
        if "val" in stage or "test" in stage:
            print("3d")
            print(
                [
                    (class_name, iou.item())
                    for class_name, iou in zip(self.class_names, segm_iou.compute())
                ]
            )

        if stage == "val/source":
            if iou_3d > self.best_source_iou_3d:
                self.log("best_source_iou_3d", iou_3d)
                self.best_source_iou_3d = iou_3d
        elif stage == "val/target":
            if iou_3d > self.best_target_iou_3d:
                self.best_target_iou_3d = iou_3d
                self.log("best_target_iou_3d", iou_3d)

        # AVG
        segm_iou = self.segm_ious_splits_avg[stage]
        iou_avg = segm_iou.compute().mean()

        if stage == "val/source":
            if iou_avg > self.best_source_iou_avg:
                self.log("best_source_iou_avg", iou_avg)
                self.best_source_iou_avg = iou_avg
        elif stage == "val/target":
            if iou_avg > self.best_target_iou_avg:
                self.best_target_iou_avg = iou_avg
                self.log("best_target_iou_avg", iou_avg)

        if "val" in stage or "test" in stage:
            print("avg")
            print(
                [
                    (class_name, iou.item())
                    for class_name, iou in zip(self.class_names, segm_iou.compute())
                ]
            )

        self.log_dict(
            {
                f"{stage}/iou": iou_2d,
                f"{stage}/iou_3d": iou_3d,
                f"{stage}/iou_avg": iou_avg,
                # **{
                #     f"{stage}/iou-{cl}": segm_iou[idx]
                #     for idx, cl in enumerate(self.class_names)
                # },
            },
            add_dataloader_idx=False,
        )

        self.segm_ious_splits_2d[stage].reset()
        self.segm_ious_splits_3d[stage].reset()
        self.segm_ious_splits_avg[stage].reset()

    def train_epoch_end(self) -> None:
        self._evaluation_end("train")

    def validation_epoch_end(self, out):
        if len(out) > 0:
            stage = out[0][0]["domain"]
            self._evaluation_end(stage)
            stage = out[1][0]["domain"]
            self._evaluation_end(stage)

    def test_epoch_end(self, out) -> None:
        # stage = out[0]["domain"]
        # print(stage)
        self._evaluation_end("test/target")

    def on_save_checkpoint(self, checkpoint) -> None:
        checkpoint["best_source_iou"] = self.best_source_iou
        checkpoint["best_target_iou"] = self.best_target_iou
        checkpoint["best_source_iou_3d"] = self.best_source_iou_3d
        checkpoint["best_target_iou_3d"] = self.best_target_iou_3d
        checkpoint["best_source_iou_avg"] = self.best_source_iou_avg
        checkpoint["best_target_iou_avg"] = self.best_target_iou_avg

    def on_load_checkpoint(self, checkpoint) -> None:
        self.best_source_iou = checkpoint["best_source_iou"]
        self.best_target_iou = checkpoint["best_target_iou"]
        self.best_source_iou_3d = checkpoint["best_source_iou_3d"]
        self.best_target_iou_3d = checkpoint["best_target_iou_3d"]
        self.best_source_iou_avg = checkpoint["best_source_iou_avg"]
        self.best_target_iou_avg = checkpoint["best_target_iou_avg"]


# Utilities
# def set_bn_eval(module):
#     if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
#         module.eval()


def _dispatch(key: str | None, items: dict | Any | None):
    if items is None:
        raise ValueError("any model registered for training")
    elif not isinstance(items, (nn.ModuleDict, dict)):
        return items
    elif key is not None:
        return items[key]
    raise ValueError("Multiple models found, choose one with model_name")


class ModelWrapper(nn.Module):
    def __init__(
        self,
        module_name: str,
        optimizer: Optimizer | None = None,
        **args,
    ):
        super().__init__()

        # loss and optimizer
        self.optimizer = optimizer
        self.name = module_name

        # load specific model
        model_mod = importlib.import_module(module_name)
        self.signature = model_mod.signature
        self.dependencies = model_mod.dependencies
        model_params = {
            n: v.default
            for n, v in inspect.signature(model_mod.Model).parameters.items()
        }
        model_params.update({k: v for k, v in args.items() if k in model_params})
        self.model_parameters = model_params
        self.model = model_mod.Model(**model_params)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def build_optimizer(self):
        opt, scheduler = self.optimizer.build(self.parameters())
        return [opt, scheduler]

    def get_model(self, script: bool = False):
        model = self.model
        if script:
            model = torch.jit.script(model)
        return model


def _load_models(model_modules, optimizer, loss, **kwargs):
    # prepare models
    out_model = None
    if isinstance(model_modules, (list, tuple, set)):
        if len(model_modules) == 0:
            raise ValueError("invalid empty model_modules list")
        out_model = nn.ModuleDict()
        for name in model_modules:
            args = kwargs[name] if name in kwargs else {}
            out_model[name] = ModelWrapper(
                name,
                optimizer[name] if optimizer is not None else None,
                **args,
            )
    elif isinstance(model_modules, str):
        out_model = ModelWrapper(model_modules, optimizer, **kwargs)
    elif model_modules is None:
        out_model = None
    else:
        raise ValueError(f"invalid model_modules type {type(model_modules)}")

    return out_model


from collections import defaultdict


class HybridOptim(torch.optim.Optimizer):
    # Wrapper around multiple optimizers that should be executed at the same time
    def __init__(self, optimizers):
        self.optimizers = optimizers
        self.defaults = optimizers[0].defaults

    @property
    def state(self):
        state = defaultdict(dict)
        for optimizer in self.optimizers:
            state = {**state, **optimizer.state}
        return state

    @property
    def param_groups(self):
        param_groups = []
        for optimizer in self.optimizers:
            param_groups = param_groups + optimizer.param_groups
        return param_groups

    def __getstate__(self):
        return [optimizer.__getstate__() for optimizer in self.optimizers]

    def __setstate__(self, state):
        for opt_state, optimizer in zip(self.optimizers, state):
            optimizer.__setstate__(opt_state)

    def __repr__(self):
        format_string = self.__class__.__name__ + " ("
        for optimizer in self.optimizers:
            format_string += "\n"
            format_string += optimizer.__repr__()
        format_string += ")"
        return format_string

    def _hook_for_profile(self):
        for optimizer in self.optimizers:
            optimizer._hook_for_profile()

    def state_dict(self):
        return [optimizer.state_dict() for optimizer in self.optimizers]

    def load_state_dict(self, state_dict):
        for state, optimizer in zip(state_dict, self.optimizers):
            optimizer.load_state_dict(state)

    def zero_grad(self, set_to_none: bool = False):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def add_param_group(self, param_group):
        raise NotImplementedError()

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for optimizer in self.optimizers:
            optimizer.step()

        return loss


class HybridLRS(torch.optim.lr_scheduler._LRScheduler):
    """Wrapper Class around lr_scheduler to return a 'dummy' optimizer to pass
    pytorch lightning checks
    """

    def __init__(self, hybrid_optimizer, idx, lr_scheduler) -> None:
        self.optimizer = hybrid_optimizer
        self.idx = idx
        self.lr_scheduler = lr_scheduler

    def __getattribute__(self, __name: str):
        if __name in {"optimizer", "idx", "lr_scheduler"}:
            return super().__getattribute__(__name)
        else:
            return self.lr_scheduler.__getattribute__(__name)
