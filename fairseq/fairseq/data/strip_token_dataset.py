# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from . import BaseWrapperDataset


class StripTokenDataset(BaseWrapperDataset):

    def __init__(self, dataset, id_to_strip):
        """
        Initialize the id_to_strip.

        Args:
            self: (todo): write your description
            dataset: (todo): write your description
            id_to_strip: (str): write your description
        """
        super().__init__(dataset)
        self.id_to_strip = id_to_strip

    def __getitem__(self, index):
        """
        Return the item corresponding to index

        Args:
            self: (todo): write your description
            index: (int): write your description
        """
        item = self.dataset[index]
        return item[item.ne(self.id_to_strip)]
