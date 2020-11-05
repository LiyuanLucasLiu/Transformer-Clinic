# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import torch


class ScalarBias(torch.autograd.Function):
    """
    Adds a vector of scalars, used in self-attention mechanism to allow
    the model to optionally attend to this vector instead of the past
    """

    @staticmethod
    def forward(ctx, input, dim, bias_init):
        """
        Forward computation.

        Args:
            ctx: (todo): write your description
            input: (todo): write your description
            dim: (int): write your description
            bias_init: (todo): write your description
        """
        size = list(input.size())
        size[dim] += 1
        output = input.new(*size).fill_(bias_init)
        output.narrow(dim, 1, size[dim] - 1).copy_(input)
        ctx.dim = dim
        return output

    @staticmethod
    def backward(ctx, grad):
        """
        Compute the backward backward backward.

        Args:
            ctx: (todo): write your description
            grad: (todo): write your description
        """
        return grad.narrow(ctx.dim, 1, grad.size(ctx.dim) - 1), None, None


def scalar_bias(input, dim, bias_init=0):
    """
    Apply scalar product.

    Args:
        input: (array): write your description
        dim: (int): write your description
        bias_init: (todo): write your description
    """
    return ScalarBias.apply(input, dim, bias_init)
