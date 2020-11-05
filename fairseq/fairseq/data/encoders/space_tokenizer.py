# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re

from fairseq.data.encoders import register_tokenizer


@register_tokenizer('space')
class SpaceTokenizer(object):

    def __init__(self, source_lang=None, target_lang=None):
        """
        Init the language.

        Args:
            self: (todo): write your description
            source_lang: (str): write your description
            target_lang: (str): write your description
        """
        self.space_tok = re.compile(r"\s+")

    def encode(self, x: str) -> str:
        """
        Encode x.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return self.space_tok.sub(' ', x)

    def decode(self, x: str) -> str:
        """
        Decode the given string.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return x
