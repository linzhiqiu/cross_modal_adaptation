''' scheuler.py
'''

from torch.optim.lr_scheduler import _LRScheduler
import torch

AVAI_SCHEDS = ["cosine", "linear"]
AVAI_WARMUP_SCHEDS = ["constant", "linear"]


class _BaseWarmupScheduler(_LRScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        last_epoch=-1,
        verbose=False
    ):
        self.successor = successor
        self.warmup_epoch = warmup_epoch
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_epoch:
            self.successor.step(epoch)
            self._last_lr = self.successor.get_last_lr()
        else:
            super().step(epoch)


class ConstantWarmupScheduler(_BaseWarmupScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        cons_lr,
        last_epoch=-1,
        verbose=False
    ):
        self.cons_lr = cons_lr
        super().__init__(
            optimizer, successor, warmup_epoch, last_epoch, verbose
        )

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        return [self.cons_lr for _ in self.base_lrs]


class LinearWarmupScheduler(_BaseWarmupScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        min_lr,
        last_epoch=-1,
        verbose=False
    ):
        self.min_lr = min_lr
        super().__init__(
            optimizer, successor, warmup_epoch, last_epoch, verbose
        )

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        if self.last_epoch == 0:
            return [self.min_lr for _ in self.base_lrs]
        return [
            lr * self.last_epoch / self.warmup_epoch for lr in self.base_lrs
        ]


def build_lr_scheduler(optimizer,
                       lr_scheduler,
                       warmup_iter,
                       max_iter,
                       warmup_type=None,
                       warmup_lr=None,
                       verbose=False):
    """A function wrapper for building a learning rate scheduler.

    Args:
        optimizer (Optimizer): an Optimizer.
        lr_scheduler (str): learning rate scheduler name, either "cosine" or "linear".
        warmup_iter (int): number of warmup iterations.
        max_iter (int): maximum iteration (not including warmup iter).
        warmup_type (str): warmup type, either constant or linear.
        warmup_lr (float): warmup learning rate.
        verbose (bool): If ``True``, prints a message to stdout
    """
    if verbose:
        print(f"Building scheduler: {lr_scheduler} with warmup: {warmup_type}")

    if lr_scheduler not in AVAI_SCHEDS:
        raise ValueError(
            f"scheduler must be one of {AVAI_SCHEDS}, but got {lr_scheduler}"
        )

    if lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(max_iter)
        )
    elif lr_scheduler == "linear":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda x: 1 - x / float(max_iter),
            last_epoch=-1
        )
        
    if warmup_iter > 0:
        if warmup_type not in AVAI_WARMUP_SCHEDS:
            raise ValueError(
                f"warmup_type must be one of {AVAI_WARMUP_SCHEDS}, "
                f"but got {warmup_type}"
            )

        if warmup_type == "constant":
            scheduler = ConstantWarmupScheduler(
                optimizer, scheduler, warmup_iter,
                warmup_lr
            )

        elif warmup_type == "linear":
            scheduler = LinearWarmupScheduler(
                optimizer, scheduler, warmup_iter,
                warmup_lr
            )

        else:
            raise ValueError

    return scheduler
