# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache

from . import BaseWrapperDataset


class LRUCacheDataset(BaseWrapperDataset):

    def __init__(self, dataset, token=None):
        """
        Initialize the dataset.

        Args:
            self: (todo): write your description
            dataset: (todo): write your description
            token: (str): write your description
        """
        super().__init__(dataset)

    @lru_cache(maxsize=8)
    def __getitem__(self, index):
        """
        Return the item at index.

        Args:
            self: (todo): write your description
            index: (int): write your description
        """
        return self.dataset[index]

    @lru_cache(maxsize=8)
    def collater(self, samples):
        """
        Collater samples.

        Args:
            self: (todo): write your description
            samples: (array): write your description
        """
        return self.dataset.collater(samples)
