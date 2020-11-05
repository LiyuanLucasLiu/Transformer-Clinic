# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.data import Dictionary


class MaskedLMDictionary(Dictionary):
    """
    Dictionary for Masked Language Modelling tasks. This extends Dictionary by
    adding the mask symbol.
    """
    def __init__(
        self,
        pad='<pad>',
        eos='</s>',
        unk='<unk>',
        mask='<mask>',
    ):
        """
        Initialize the mask.

        Args:
            self: (todo): write your description
            pad: (todo): write your description
            eos: (int): write your description
            unk: (todo): write your description
            mask: (array): write your description
        """
        super().__init__(pad, eos, unk)
        self.mask_word = mask
        self.mask_index = self.add_symbol(mask)
        self.nspecial = len(self.symbols)

    def mask(self):
        """Helper to get index of mask symbol"""
        return self.mask_index


class BertDictionary(MaskedLMDictionary):
    """
    Dictionary for BERT task. This extends MaskedLMDictionary by adding support
    for cls and sep symbols.
    """
    def __init__(
        self,
        pad='<pad>',
        eos='</s>',
        unk='<unk>',
        mask='<mask>',
        cls='<cls>',
        sep='<sep>'
    ):
        """
        Initialize symbol.

        Args:
            self: (todo): write your description
            pad: (todo): write your description
            eos: (int): write your description
            unk: (todo): write your description
            mask: (array): write your description
            cls: (todo): write your description
            sep: (str): write your description
        """
        super().__init__(pad, eos, unk, mask)
        self.cls_word = cls
        self.sep_word = sep
        self.cls_index = self.add_symbol(cls)
        self.sep_index = self.add_symbol(sep)
        self.nspecial = len(self.symbols)

    def cls(self):
        """Helper to get index of cls symbol"""
        return self.cls_index

    def sep(self):
        """Helper to get index of sep symbol"""
        return self.sep_index
