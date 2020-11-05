# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq import file_utils
from fairseq.data.encoders import register_bpe


@register_bpe('fastbpe')
class fastBPE(object):

    @staticmethod
    def add_args(parser):
        """
        Add command line arguments.

        Args:
            parser: (todo): write your description
        """
        # fmt: off
        parser.add_argument('--bpe-codes', type=str,
                            help='path to fastBPE BPE')
        # fmt: on

    def __init__(self, args):
        """
        Initialize bpe file.

        Args:
            self: (todo): write your description
        """
        if args.bpe_codes is None:
            raise ValueError('--bpe-codes is required for --bpe=subword_nmt')
        codes = file_utils.cached_path(args.bpe_codes)
        try:
            import fastBPE
            self.bpe = fastBPE.fastBPE(codes)
            self.bpe_symbol = "@@ "
        except ImportError:
            raise ImportError('Please install fastBPE with: pip install fastBPE')

    def encode(self, x: str) -> str:
        """
        Encodes the given x and bpe.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return self.bpe.apply([x])[0]

    def decode(self, x: str) -> str:
        """
        Decode a bpe string.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return (x + ' ').replace(self.bpe_symbol, '').rstrip()
