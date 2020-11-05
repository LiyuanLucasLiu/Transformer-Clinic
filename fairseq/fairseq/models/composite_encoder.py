# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.models import FairseqEncoder


class CompositeEncoder(FairseqEncoder):
    """
    A wrapper around a dictionary of :class:`FairseqEncoder` objects.

    We run forward on each encoder and return a dictionary of outputs. The first
    encoder's dictionary is used for initialization.

    Args:
        encoders (dict): a dictionary of :class:`FairseqEncoder` objects.
    """

    def __init__(self, encoders):
        """
        Initialize the encoders.

        Args:
            self: (todo): write your description
            encoders: (list): write your description
        """
        super().__init__(next(iter(encoders.values())).dictionary)
        self.encoders = encoders
        for key in self.encoders:
            self.add_module(key, self.encoders[key])

    def forward(self, src_tokens, src_lengths):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): lengths of each source sentence of shape
                `(batch)`

        Returns:
            dict:
                the outputs from each Encoder
        """
        encoder_out = {}
        for key in self.encoders:
            encoder_out[key] = self.encoders[key](src_tokens, src_lengths)
        return encoder_out

    def reorder_encoder_out(self, encoder_out, new_order):
        """Reorder encoder output according to new_order."""
        for key in self.encoders:
            encoder_out[key] = self.encoders[key].reorder_encoder_out(encoder_out[key], new_order)
        return encoder_out

    def max_positions(self):
        """
        The maximum positions of all positions.

        Args:
            self: (todo): write your description
        """
        return min([self.encoders[key].max_positions() for key in self.encoders])

    def upgrade_state_dict(self, state_dict):
        """
        Return a dictionary to dict.

        Args:
            self: (todo): write your description
            state_dict: (dict): write your description
        """
        for key in self.encoders:
            self.encoders[key].upgrade_state_dict(state_dict)
        return state_dict
