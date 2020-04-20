# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    if not export and torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm
            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)

# from ipdb import set_trace

# class LayerNorm(nn.Module):

#   def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
#       super(LayerNorm, self).__init__()

#       # if not export and torch.cuda.is_available():
#       #     try:
#       #         from apex.normalization import FusedLayerNorm
#       #         self.ln = FusedLayerNorm(normalized_shape, eps, elementwise_affine)
#       #     except ImportError:
#       #         pass

#       self.ln = torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)

#   def forward(self, x):
#     previous_type = x.type()
#     x = x.float()
#     x = self.ln(x)
#     x = x.type(previous_type)
#     return x