# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        """
        Forward a copy of x

        Args:
            ctx: (todo): write your description
            x: (todo): write your description
            scale: (float): write your description
        """
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        """
        Compute backward backward backward backward.

        Args:
            ctx: (todo): write your description
            grad: (array): write your description
        """
        return grad * ctx.scale, None
