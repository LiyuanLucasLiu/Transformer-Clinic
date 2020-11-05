# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.optim.lr_scheduler

from . import FairseqLRScheduler, register_lr_scheduler


@register_lr_scheduler('reduce_lr_on_plateau')
class ReduceLROnPlateau(FairseqLRScheduler):
    """Decay the LR by a factor every time the validation loss plateaus."""

    def __init__(self, args, optimizer):
        """
        Initialize the optimizer.

        Args:
            self: (todo): write your description
            optimizer: (todo): write your description
        """
        super().__init__(args, optimizer)
        if len(args.lr) > 1:
            raise ValueError(
                'Cannot use a fixed learning rate schedule with reduce_lr_on_plateau.'
                ' Consider --lr-scheduler=fixed instead.'
            )
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer.optimizer, patience=0, factor=args.lr_shrink,
            threshold=args.lr_threshold)

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        # fmt: off
        parser.add_argument('--lr-shrink', default=0.1, type=float, metavar='LS',
                            help='shrink factor for annealing, lr_new = (lr * lr_shrink)')
        parser.add_argument('--lr-threshold', default=1e-4, type=float, metavar='LT',
                            help='Threshold for measuring the new optimum, \
                            to only focus on significant changes')
        # fmt: on

    def state_dict(self):
        """Return the LR scheduler state dict."""
        return {
            'best': self.lr_scheduler.best,
            'last_epoch': self.lr_scheduler.last_epoch,
        }

    def load_state_dict(self, state_dict):
        """Load an LR scheduler state dict."""
        self.lr_scheduler.best = state_dict['best']
        if 'last_epoch' in state_dict:
            self.lr_scheduler.last_epoch = state_dict['last_epoch']

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        if val_loss is not None:
            self.lr_scheduler.step(val_loss, epoch)
        else:
            self.lr_scheduler.last_epoch = epoch
        return self.optimizer.get_lr()
