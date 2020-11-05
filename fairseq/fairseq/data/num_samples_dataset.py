# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from . import FairseqDataset


class NumSamplesDataset(FairseqDataset):

    def __getitem__(self, index):
        """
        Returns the item at index.

        Args:
            self: (todo): write your description
            index: (int): write your description
        """
        return 1

    def __len__(self):
        """
        Returns the number of bytes.

        Args:
            self: (todo): write your description
        """
        return 0

    def collater(self, samples):
        """
        Collater samples.

        Args:
            self: (todo): write your description
            samples: (array): write your description
        """
        return sum(samples)
