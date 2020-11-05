# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from . import BaseWrapperDataset


class NumelDataset(BaseWrapperDataset):

    def __init__(self, dataset, reduce=False):
        """
        Initialize the dataset.

        Args:
            self: (todo): write your description
            dataset: (todo): write your description
            reduce: (str): write your description
        """
        super().__init__(dataset)
        self.reduce = reduce

    def __getitem__(self, index):
        """
        Return a tensor of the index.

        Args:
            self: (todo): write your description
            index: (int): write your description
        """
        item = self.dataset[index]
        if torch.is_tensor(item):
            return torch.numel(item)
        else:
            return np.size(item)

    def __len__(self):
        """
        Returns the number of the dataset.

        Args:
            self: (todo): write your description
        """
        return len(self.dataset)

    def collater(self, samples):
        """
        Collater samples.

        Args:
            self: (todo): write your description
            samples: (array): write your description
        """
        if self.reduce:
            return sum(samples)
        else:
            return torch.tensor(samples)
