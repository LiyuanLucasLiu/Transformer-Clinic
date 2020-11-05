# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from . import FairseqDataset


class RawLabelDataset(FairseqDataset):

    def __init__(self, labels):
        """
        Initialize the state variables.

        Args:
            self: (todo): write your description
            labels: (dict): write your description
        """
        super().__init__()
        self.labels = labels

    def __getitem__(self, index):
        """
        Return the item for the given index.

        Args:
            self: (todo): write your description
            index: (int): write your description
        """
        return self.labels[index]

    def __len__(self):
        """
        Return the length of the length.

        Args:
            self: (todo): write your description
        """
        return len(self.labels)

    def collater(self, samples):
        """
        Collater samples.

        Args:
            self: (todo): write your description
            samples: (array): write your description
        """
        return torch.tensor(samples)
