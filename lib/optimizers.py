"""
Dispatcher to programmatically create optimizers and schedulers
"""

from torch import optim
from torch.optim import lr_scheduler

__all__ = ["Optimizer"]


class Optimizer:
    def __init__(self, name: str, **kwargs):
        self._optim_name = name
        self._optim_args = kwargs
        self._use_scheduler = False

    def set_scheduler(self, name: str, **kwargs):
        self._scheduler_name = name
        self._scheduler_args = kwargs
        self._use_scheduler = True

    def build(self, params) -> tuple[optim.Optimizer, lr_scheduler._LRScheduler | None]:

        optimizer = {
            "adam": optim.Adam,
            "adamw": optim.AdamW,
            "sgd": optim.SGD,
            "rmsprop": optim.RMSprop,
        }[self._optim_name](params, **self._optim_args)

        scheduler = None
        if self._use_scheduler:
            scheduler = {
                "step": lr_scheduler.StepLR,
                "cosine_annealing": lr_scheduler.CosineAnnealingLR,
                "cyclic": lr_scheduler.CyclicLR,
                "reduce_on_plateau": lr_scheduler.ReduceLROnPlateau,
                "multi_step_lr": lr_scheduler.MultiStepLR,
                "one_cycle": lr_scheduler.OneCycleLR,
            }[self._scheduler_name](optimizer, **self._scheduler_args)

        return optimizer, scheduler
