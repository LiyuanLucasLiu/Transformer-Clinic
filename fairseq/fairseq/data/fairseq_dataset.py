# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch.utils.data


class FairseqDataset(torch.utils.data.Dataset):
    """A dataset that provides helpers for batching."""

    def __getitem__(self, index):
        """
        Get an item from the given index.

        Args:
            self: (todo): write your description
            index: (int): write your description
        """
        raise NotImplementedError

    def __len__(self):
        """
        Returns the number of bytes.

        Args:
            self: (todo): write your description
        """
        raise NotImplementedError

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch suitable for forwarding with a Model
        """
        raise NotImplementedError

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        raise NotImplementedError

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        raise NotImplementedError

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        return np.arange(len(self))

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return False

    def attr(self, attr: str, index: int):
        """
        Returns the value of the given index.

        Args:
            self: (todo): write your description
            attr: (str): write your description
            index: (int): write your description
        """
        return getattr(self, attr, None)

    def prefetch(self, indices):
        """Prefetch the data required for this epoch."""
        raise NotImplementedError

    def set_epoch(self, epoch):
        """
        Set the epoch for the given epoch.

        Args:
            self: (todo): write your description
            epoch: (todo): write your description
        """
        pass
