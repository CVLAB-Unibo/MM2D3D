"""
Compose possible losses
"""

from copy import deepcopy
from typing import Any, Literal, TypedDict

import torch
from torch.nn import functional as F
from typing_extensions import NotRequired

__all__ = ["Loss"]


# utilities


class _GenericLoss:
    def __init__(self, **args):
        self.other_args = args

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        repr = self.name
        if self.other_args:
            repr = (
                repr
                + "["
                + ",".join([f"{n}={v}" for n, v in self.other_args.items()])
                + "]"
            )
        return repr


class L1(_GenericLoss):
    name = "l1"
    default_target = "depth"

    def __call__(self, pred: torch.Tensor, gt: torch.Tensor, **kwargs):
        mask = gt > 0
        return torch.mean(torch.abs(pred[mask] - gt[mask]))


class L2(_GenericLoss):
    name = "l2"
    default_target = "depth"

    def __call__(self, pred: torch.Tensor, gt: torch.Tensor, **kwargs):
        mask = gt > 0
        return torch.mean(torch.square(pred[mask] - gt[mask]))


class CrossEntropy(_GenericLoss):
    name = "cross_entropy"
    default_target = "segmentation"

    def __call__(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        weight: list[float] | None = None,
        **kwargs,
    ):
        if isinstance(weight, list):
            weight = torch.tensor(weight, dtype=pred.dtype, device=gt.device)
        return F.cross_entropy(pred, gt, weight=weight)


# interface


class FullLossCfg(TypedDict):
    name: str
    weight: NotRequired[float]
    target: NotRequired[str]
    args: NotRequired[dict[str, Any]]


class Loss:
    def __init__(self, cfg: str | list[str] | FullLossCfg):

        losses = {
            "l1": L1,
            "l2": L2,
            "cross_entropy": CrossEntropy,
        }

        if isinstance(cfg, str):
            loss_fn = losses[cfg]()
            self._losses = [(1.0, loss_fn.default_target, loss_fn)]
        elif isinstance(cfg, list):
            self._losses = []
            for loss in cfg:
                if isinstance(loss, str):
                    loss_fn = losses[loss]()
                    self._losses.append((1.0, loss_fn.default_target, loss_fn))
                else:
                    loss_fn = losses[loss["name"]]
                    self._losses.append(
                        (
                            loss.get("weight", 1.0),
                            loss.get("target", loss_fn.default_target),
                            loss_fn(**loss.get("args", {})),
                        )
                    )
        else:
            raise ValueError(f"not recognized cfg {cfg}")

    def update_loss_params(self, loss_name: str, loss_target: str, **kwargs):
        """
        Update some parameters for a specific loss, if the indicated loss is
        not present this function does not do nothing
        """
        for _, target, loss in self._losses:
            if loss.name == loss_name and target == loss_target:
                loss.other_args.update(**kwargs)

    def __call__(
        self,
        target: str,
        image: torch.Tensor | None = None,
        pred: torch.Tensor | None = None,
        gt: torch.Tensor | None = None,
    ):

        losses_filtered = [(w, l) for w, t, l in self._losses if t == target]
        if not losses_filtered:
            raise RuntimeError(f"no losses for loss target {target}")

        out = 0.0
        for weight, loss in losses_filtered:
            out += weight * loss(image=image, pred=pred, gt=gt, **loss.other_args)
        return out

    def __repr__(self):
        if len(self._losses) == 1:
            w, _, loss = self._losses[0]
            return str(w) + str(loss) if w != 1.0 else str(loss)
        else:
            return "+".join(
                [f"{w if w != 1.0 else ''}{str(loss)}" for w, _, loss in self._losses]
            )

    def split_by_target(self) -> "dict[str, Loss]":
        targets = {t for w, t, l in self._losses}
        losses = {}
        for t in targets:
            _loss_copy = deepcopy(self)
            _loss_copy._losses = [deepcopy(l) for l in self._losses if l[1] == t]
            losses[t] = _loss_copy
        return losses
