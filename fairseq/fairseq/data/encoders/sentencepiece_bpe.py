# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq import file_utils
from fairseq.data.encoders import register_bpe


@register_bpe('sentencepiece')
class SentencepieceBPE(object):

    @staticmethod
    def add_args(parser):
        """
        Add command line arguments.

        Args:
            parser: (todo): write your description
        """
        # fmt: off
        parser.add_argument('--sentencepiece-vocab', type=str,
                            help='path to sentencepiece vocab')
        # fmt: on

    def __init__(self, args):
        """
        Initialize the sentence.

        Args:
            self: (todo): write your description
        """
        vocab = file_utils.cached_path(args.sentencepiece_vocab)
        try:
            import sentencepiece as spm
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(vocab)
        except ImportError:
            raise ImportError('Please install sentencepiece with: pip install sentencepiece')

    def encode(self, x: str) -> str:
        """
        Encode a string x.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return ' '.join(self.sp.EncodeAsPieces(x))

    def decode(self, x: str) -> str:
        """
        Decode a string.

        Args:
            self: (todo): write your description
            x: (str): write your description
        """
        return x.replace(' ', '').replace('\u2581', ' ').strip()
