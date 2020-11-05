# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from . import BaseWrapperDataset


class SortDataset(BaseWrapperDataset):

    def __init__(self, dataset, sort_order):
        """
        Initialize the dataset.

        Args:
            self: (todo): write your description
            dataset: (todo): write your description
            sort_order: (str): write your description
        """
        super().__init__(dataset)
        if not isinstance(sort_order, (list, tuple)):
            sort_order = [sort_order]
        self.sort_order = sort_order

        assert all(len(so) == len(dataset) for so in sort_order)

    def ordered_indices(self):
        """
        Return the ordered list of the indices.

        Args:
            self: (todo): write your description
        """
        return np.lexsort(self.sort_order)
