# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.modules import TransformerSentenceEncoderLayer
from fairseq.modules.sparse_multihead_attention import SparseMultiheadAttention


class SparseTransformerSentenceEncoderLayer(TransformerSentenceEncoderLayer):
    """
    Implements a Sprase Transformer Encoder Layer (see SparseMultiheadAttention)
    """

    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: float = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = 'relu',
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        export: bool = False,
        is_bidirectional: bool = True,
        stride: int = 32,
        expressivity: int = 8,
    ) -> None:
        """
        Initialize embeddings.

        Args:
            self: (todo): write your description
            embedding_dim: (int): write your description
            ffn_embedding_dim: (int): write your description
            num_attention_heads: (int): write your description
            dropout: (str): write your description
            attention_dropout: (todo): write your description
            activation_dropout: (str): write your description
            activation_fn: (str): write your description
            add_bias_kv: (str): write your description
            add_zero_attn: (todo): write your description
            export: (str): write your description
            is_bidirectional: (bool): write your description
            stride: (int): write your description
            expressivity: (todo): write your description
        """

        super().__init__(
            embedding_dim, ffn_embedding_dim, num_attention_heads, dropout,
            attention_dropout, activation_dropout, activation_fn, add_bias_kv,
            add_zero_attn, export
        )

        self.self_attn = SparseMultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=True,
            is_bidirectional=is_bidirectional,
            stride=stride,
            expressivity=expressivity,
        )
