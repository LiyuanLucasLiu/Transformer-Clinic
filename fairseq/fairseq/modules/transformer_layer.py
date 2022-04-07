# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.modules import LayerNorm, MultiheadAttention
import math

import functools

import os
import json
import numpy as np
encoder_ratio = 1.0 # encoder_ratio is the ratio w.r.t. un-adjusted output from PreLN
decoder_ratio = 1.0 # decoder_ratio is the ratio w.r.t. un-adjusted output from PreLN
tmp_file = 0

ratio_list = None
residual_components = None

class TransformerEncoderLayer(nn.Module):

    def __init__(self, args, LayerNum=None):
        super().__init__()
        global tmp_file 
        
        self.args = args
        if not hasattr(self.args, 'mixed_precision'):
            self.args.mixed_precision = False
        if not hasattr(self.args, 'plot_variance'):
            self.args.plot_variance = False
        if not hasattr(self.args, 'plot_gradient'):
            self.args.plot_gradient = False
        if not hasattr(self.args, 'plot_stability'):
            self.args.plot_stability = False

        self.normalize_before = args.encoder_normalize_before
        self.embed_dim = args.encoder_embed_dim

        self.layer_num = LayerNum
        # if LayerNum is not None and not self.normalize_before:
        if 'adaptive' in args.init_type:
            assert not self.normalize_before

            self.self_attn = MultiheadAttention(
                self.embed_dim, args.encoder_attention_heads,
                dropout=args.attention_dropout, self_attention=True
            )

            self.fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
            self.fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)

            if 'adaptive-profiling' == args.init_type:
                if not tmp_file:
                    tmp_file = open('profile.ratio.init', 'w')
                self.attention_ratio_change = nn.Parameter(torch.ones(self.embed_dim))
                self.fc_ratio_change = nn.Parameter(torch.ones(self.embed_dim))
            else:
                if not tmp_file:
                    tmp_file = open('profile.ratio.init', 'r')

                layer_iter, next_value = [float(tup) for tup in tmp_file.readline().split()]
                print('layer_num: {}, layer_iter: {}'.format(self.layer_num, layer_iter))
                assert layer_iter == 2 * self.layer_num + 1
                print('encoder attn ratio: {}'.format(next_value))
                self.attention_ratio_change = nn.Parameter(torch.ones(self.embed_dim))
                self.attention_ratio_change.data.fill_(next_value)

                layer_iter, next_value = [float(tup) for tup in tmp_file.readline().split()]
                print('layer_num: {}, layer_iter: {}'.format(self.layer_num, layer_iter))
                assert layer_iter == 2 * self.layer_num + 2
                print('encoder ffn ratio: {}'.format(next_value))
                self.fc_ratio_change = nn.Parameter(torch.ones(self.embed_dim))
                self.fc_ratio_change.data.fill_(next_value)

            self.self_attn_layer_norm = LayerNorm(self.embed_dim) 
            self.final_layer_norm = LayerNorm(self.embed_dim)

        else:

            self.self_attn = MultiheadAttention(
                self.embed_dim, args.encoder_attention_heads,
                dropout=args.attention_dropout, self_attention=True,
                fuse_inproject=not('deepnet' in args.init_type),
            )

            self.fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
            self.fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)
            if args.init_type == 'looklinear':
                self.fc1.weight.data[int(args.encoder_ffn_embed_dim / 2):, :] = -self.fc1.weight.data[0: int(args.encoder_ffn_embed_dim / 2), :]
                self.fc2.weight.data[:, int(args.encoder_ffn_embed_dim / 2):] = -self.fc2.weight.data[:, 0: int(args.encoder_ffn_embed_dim / 2)]

            if args.init_type != 'rezero':
                self.self_attn_layer_norm = LayerNorm(self.embed_dim)
                self.final_layer_norm = LayerNorm(self.embed_dim)
            else:
                self.self_attn_layer_norm = None
                self.final_layer_norm = None

            if 'rezero' in args.init_type:
                self.rezero_weight = nn.Parameter(torch.Tensor([0]))
            else:
                self.rezero_weight = None
            
            if 'deepnet' in args.init_type:           
                self.deepnet = True     
                self.attention_ratio_change = 0.81 * (self.args.encoder_layers ** 4 * self.args.decoder_layers) ** (1./16)
                self.fc_ratio_change = 0.81 * (self.args.encoder_layers ** 4 * self.args.decoder_layers) ** (1./16)
                
                beta=0.87 * (self.args.encoder_layers ** 4 * self.args.decoder_layers) ** (-1./16)
                nn.init.xavier_normal_(self.fc1.weight, gain=beta)
                nn.init.xavier_normal_(self.fc2.weight, gain=beta)
                nn.init.xavier_normal_(self.self_attn.v_proj_weight, gain=beta)
                nn.init.xavier_normal_(self.self_attn.out_proj.weight, gain=beta)
                
                nn.init.xavier_normal_(self.self_attn.q_proj_weight, gain=1.)
                nn.init.xavier_normal_(self.self_attn.k_proj_weight, gain=1.)
            else:
                assert args.init_type == 'default'
                self.deepnet = False

        if self.args.plot_stability:
            self.x0_hat = None
            self.x1_hat = None
            if self.layer_num == self.args.encoder_layers - 1:
                self.x_final = None

        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu')
        )
        self.activation_dropout = getattr(args, 'activation_dropout', 0)
        if self.activation_dropout == 0:
            self.activation_dropout = getattr(args, 'relu_dropout', 0)

        if args.fp16:
            self.in_type=torch.half
        else:
            self.in_type=torch.float

    def upgrade_state_dict_named(self, state_dict, name):
        layer_norm_map = {
            '0': 'self_attn_layer_norm',
            '1': 'final_layer_norm'
        }
        for old, new in layer_norm_map.items():
            for m in ('weight', 'bias'):
                k = '{}.layer_norms.{}.{}'.format(name, old, m)
                if k in state_dict:
                    state_dict[
                        '{}.{}.{}'.format(name, new, m)
                    ] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, encoder_padding_mask, attn_mask=None):

        not_initialized = ('adaptive-profiling' == self.args.init_type) and (1.0 == self.attention_ratio_change.min()) and self.training

        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)

        if self.args.plot_stability:
            if self.x0_hat is None:
                std = x.float().std(dim = -1, keepdim = True)
                self.x0_hat = (x / std).data
            else:
                std = x.float().std(dim = -1, keepdim = True)
                x0star_hat = (x / std).data
                diff = self.x0_hat - x0star_hat
                print('{} {}'.format(self.layer_num * 2, diff.norm().item()))

        # if self.args.mixed_precision: 
        #     x = x.type(self.in_type)
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.bool(), -1e8)
        x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # if self.args.mixed_precision:
        #     x = x.float()
        if 'adaptive' in self.args.init_type:
            if not_initialized:
                global encoder_ratio, tmp_file
                tmp_layer_ind = self.layer_num * 2 + 1
                tmp_ratio = encoder_ratio
                tmp_file.write('{} {}\n'.format(tmp_layer_ind, tmp_ratio))
                self.attention_ratio_change.data.fill_(tmp_ratio)
                print ('encoder attn ratio: {}'.format(tmp_ratio))
                input_std = np.var( (residual*self.attention_ratio_change) .clone().cpu().float().data.contiguous().view(-1).numpy())
                output_std = np.var(x.clone().cpu().float().data.contiguous().view(-1).numpy())
                encoder_ratio = np.sqrt(input_std + output_std)
            x0 = x + residual * self.attention_ratio_change
        elif self.deepnet:
            x0 = x + residual * self.attention_ratio_change
        elif self.rezero_weight is not None:
            x0 = residual + self.rezero_weight * x
        else:
            x0 = residual + x
        if self.args.plot_variance:
            if not self.args.encoder_normalize_before:
                global ratio_list, residual_components
                if ratio_list is None:
                    ratio_list = list()
                    residual_components = [residual.float()]
                ratio_list.append([tup.std(dim=-1).mean().item() for tup in residual_components])
                residual_components.append(x.float())
                if 'adaptive' in self.args.init_type:
                    for ind in range(len(residual_components) - 1):
                        residual_components[ind] = residual_components[ind] * self.self_attn_layer_norm.weight * self.attention_ratio_change.data
                    residual_components[-1] = residual_components[-1] * self.self_attn_layer_norm.weight
                else:
                    for ind in range(len(residual_components)):
                        residual_components[ind] = residual_components[ind] * self.self_attn_layer_norm.weight

            else:
                if ratio_list is None:
                    ratio_list = list()
                    residual_components = [residual.float()]
                ratio_list.append([tup.std(dim=-1).mean().item() for tup in residual_components ])
                residual_components.append(x.float())

        x0 = self.maybe_layer_norm(self.self_attn_layer_norm, x0, after=True)
        residual = x0
        if self.args.plot_gradient:
            x0.register_hook(lambda grad: print('{} encoder attn: {}'.format(self.layer_num, grad.norm().item())))
        x = self.maybe_layer_norm(self.final_layer_norm, x0, before=True)

        if self.args.plot_stability:
            if self.x1_hat is None:
                std = x.float().std(dim = -1, keepdim = True)
                self.x1_hat = (x / std).data
            else:
                std = x.float().std(dim = -1, keepdim = True)
                x1star_hat = (x / std).data
                diff = self.x1_hat - x1star_hat
                print('{} {}'.format(self.layer_num * 2 + 1, diff.norm().item()))

        # if self.args.mixed_precision: 
        #     x = x.type(self.in_type)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # if self.args.mixed_precision:
        #     x = x.float()
        if 'adaptive' in self.args.init_type:
            if not_initialized:
                tmp_layer_ind = self.layer_num * 2 + 2
                tmp_ratio = encoder_ratio
                tmp_file.write('{} {}\n'.format(tmp_layer_ind, tmp_ratio))
                self.fc_ratio_change.data.fill_(tmp_ratio)
                print ('encoder ffn ratio: {}'.format(tmp_ratio))
                input_std = np.var( (residual*self.fc_ratio_change) .clone().cpu().float().data.contiguous().view(-1).numpy())
                output_std = np.var(x.clone().cpu().float().data.contiguous().view(-1).numpy())
                encoder_ratio = np.sqrt(input_std + output_std)
            x1 = x + residual * self.fc_ratio_change
        elif self.deepnet:
            x1 = x + residual * self.fc_ratio_change
        elif self.rezero_weight is not None:
            x1 = residual + self.rezero_weight * x
        else:
            x1 = residual + x

        if self.args.plot_variance:
            if not self.args.encoder_normalize_before:
                ratio_list.append([tup.std(dim=-1).mean().item() for tup in residual_components])
                residual_components.append(x.float())
                if 'adaptive' in self.args.init_type:
                    for ind in range(len(residual_components) - 1):
                        residual_components[ind] = residual_components[ind] * self.final_layer_norm.weight * self.fc_ratio_change.data
                    residual_components[-1] = residual_components[-1] * self.final_layer_norm.weight
                else:
                    for ind in range(len(residual_components) - 1):
                        residual_components[ind] = residual_components[ind] * self.final_layer_norm.weight

            else:
                std = residual.float().std(dim=-1, keepdim=True)
                ratio_list.append([ tup.std(dim=-1).mean().item() for tup in residual_components ])
                residual_components.append(x.float())

            if self.layer_num == self.args.encoder_layers - 1:
                if self.args.encoder_normalize_before:
                    for ind in range(len(residual_components)):
                        residual_components[ind] = residual_components[ind]
                ratio_list.append([tup.std(dim=-1).mean().item() for tup in residual_components])
                json.dump(ratio_list, open('variance_log.json', 'w'))

        x1 = self.maybe_layer_norm(self.final_layer_norm, x1, after=True)
        if self.args.plot_gradient:
            x1.register_hook(lambda grad: print('{} encoder ffn: {}'.format(self.layer_num, grad.norm().item())))
        if self.args.plot_stability and self.layer_num == self.args.encoder_layers - 1:

            if self.x_final is None:
                self.x_final = x1.data
            else:
                x_finalstar_hat = x1.data
                diff = (self.x_final - x_finalstar_hat).norm().item()
                print('final {}'.format(diff))
                name = self.args.restore_file
                with open('{}.log'.format(name), 'a') as fout:
                    fout.write("{} {}\n".format(self.args.encoder_layers, diff))
        return x1

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        if self.args.init_type == 'rezero':
            return x

        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

class TransformerDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False, LayerNum=None):
        super().__init__()

        global tmp_file

        self.args = args
        if not hasattr(self.args, 'mixed_precision'):
            self.args.mixed_precision = False
        if not hasattr(self.args, 'plot_variance'):
            self.args.plot_variance = False
        if not hasattr(self.args, 'plot_gradient'):
            self.args.plot_gradient = False

        self.normalize_before = args.decoder_normalize_before
        self.embed_dim = args.decoder_embed_dim
        self.cross_self_attention = getattr(args, 'cross_self_attention', False)

        self.layer_num = LayerNum
        if 'adaptive' in args.init_type:
            assert not self.normalize_before

            self.self_attn = MultiheadAttention(
                embed_dim=self.embed_dim,
                num_heads=args.decoder_attention_heads,
                dropout=args.attention_dropout,
                add_bias_kv=add_bias_kv,
                add_zero_attn=add_zero_attn,
                self_attention=not self.cross_self_attention
            )

            assert not no_encoder_attn
            self.encoder_attn = MultiheadAttention(
                self.embed_dim,
                args.decoder_attention_heads,
                kdim=getattr(args, 'encoder_embed_dim', None),
                vdim=getattr(args, 'encoder_embed_dim', None),
                dropout=args.attention_dropout,
                encoder_decoder_attention=True
            )

            self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
            self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)
            
            if 'adaptive-profiling' == args.init_type:
                if not tmp_file:
                    tmp_file = open('profile.ratio.init', 'w')
                self.self_ratio_change = nn.Parameter(torch.ones(self.embed_dim))
                self.encoder_ratio_change = nn.Parameter(torch.ones(self.embed_dim))
                self.fc_ratio_change = nn.Parameter(torch.ones(self.embed_dim))
            else:
                if not tmp_file:
                    tmp_file = open('profile.ratio.init', 'r')

                layer_iter, next_value = [float(tup) for tup in tmp_file.readline().split()]
                print('layer_num: {}, layer_iter: {}'.format(self.layer_num, layer_iter))
                assert layer_iter == 3 * self.layer_num + 1
                print('decoder self ratio: {}'.format(next_value))
                self.self_ratio_change = nn.Parameter(torch.ones(self.embed_dim))
                self.self_ratio_change.data.fill_(next_value)

                layer_iter, next_value = [float(tup) for tup in tmp_file.readline().split()]
                print('layer_num: {}, layer_iter: {}'.format(self.layer_num, layer_iter))
                assert layer_iter == 3 * self.layer_num + 2
                print('decoder en ratio: {}'.format(next_value))
                self.encoder_ratio_change = nn.Parameter(torch.ones(self.embed_dim))
                self.encoder_ratio_change.data.fill_(next_value)

                layer_iter, next_value = [float(tup) for tup in tmp_file.readline().split()]
                print('layer_num: {}, layer_iter: {}'.format(self.layer_num, layer_iter))
                assert layer_iter == 3 * self.layer_num + 3
                print('decoder ffn ratio: {}'.format(next_value))
                self.fc_ratio_change = nn.Parameter(torch.ones(self.embed_dim))
                self.fc_ratio_change.data.fill_(next_value)

            export = getattr(args, 'char_inputs', False)
            self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export) 
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export) 
            self.final_layer_norm = LayerNorm(self.embed_dim, export=export) 
        else:
            self.self_attn = MultiheadAttention(
                embed_dim=self.embed_dim,
                num_heads=args.decoder_attention_heads,
                dropout=args.attention_dropout,
                add_bias_kv=add_bias_kv,
                add_zero_attn=add_zero_attn,
                self_attention=not self.cross_self_attention,
                fuse_inproject=not('deepnet' in args.init_type),
            )

            assert not no_encoder_attn
            self.encoder_attn = MultiheadAttention(
                self.embed_dim,
                args.decoder_attention_heads,
                kdim=getattr(args, 'encoder_embed_dim', None),
                vdim=getattr(args, 'encoder_embed_dim', None),
                dropout=args.attention_dropout,
                encoder_decoder_attention=True,
                fuse_inproject=not('deepnet' in args.init_type),
            )
            
            self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
            self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)
            if args.init_type == 'looklinear':
                self.fc1.weight.data[int(args.decoder_ffn_embed_dim / 2):, :] = -self.fc1.weight.data[0: int(args.decoder_ffn_embed_dim / 2), :]
                self.fc2.weight.data[:, int(args.decoder_ffn_embed_dim / 2):] = -self.fc2.weight.data[:, 0: int(args.decoder_ffn_embed_dim / 2)]

            export = getattr(args, 'char_inputs', False)

            if args.init_type != 'rezero':
                self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)
                if no_encoder_attn:
                    self.encoder_attn = None
                    self.encoder_attn_layer_norm = None
                else:
                    self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)
                self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
            else:
                self.self_attn_layer_norm = None
                self.encoder_attn_layer_norm = None
                self.final_layer_norm = None

            if 'rezero' in args.init_type:
                self.rezero_weight = nn.Parameter(torch.Tensor([0]))
            else:
                self.rezero_weight = None

            if 'deepnet' in args.init_type:           
                self.deepnet = True     

                self.self_ratio_change = (3 * self.args.decoder_layers) ** (1./4)
                self.encoder_ratio_change = (3 * self.args.decoder_layers) ** (1./4)
                self.fc_ratio_change = (3 * self.args.decoder_layers) ** (1./4)
                        
                beta=(12 * self.args.decoder_layers) ** (-1./4)
                nn.init.xavier_normal_(self.fc1.weight, gain=beta)
                nn.init.xavier_normal_(self.fc2.weight, gain=beta)
                nn.init.xavier_normal_(self.self_attn.v_proj_weight, gain=beta)
                nn.init.xavier_normal_(self.self_attn.out_proj.weight, gain=beta)
                nn.init.xavier_normal_(self.encoder_attn.v_proj_weight, gain=beta)
                nn.init.xavier_normal_(self.encoder_attn.out_proj.weight, gain=beta)
                
                nn.init.xavier_normal_(self.encoder_attn.q_proj_weight, gain=1.)
                nn.init.xavier_normal_(self.encoder_attn.k_proj_weight, gain=1.)
                nn.init.xavier_normal_(self.self_attn.q_proj_weight, gain=1.)
                nn.init.xavier_normal_(self.self_attn.k_proj_weight, gain=1.)
            else:
                assert args.init_type == 'default'
                self.deepnet = False

        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu')
        )
        self.activation_dropout = getattr(args, 'activation_dropout', 0)
        if self.activation_dropout == 0:
            self.activation_dropout = getattr(args, 'relu_dropout', 0)


        self.need_attn = True

        self.onnx_trace = False

        if args.fp16:
            self.in_type=torch.half
        else:
            self.in_type=torch.float

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(
        self,
        x,
        encoder_out=None,
        encoder_padding_mask=None,
        incremental_state=None,
        prev_self_attn_state=None,
        prev_attn_state=None,
        self_attn_mask=None,
        self_attn_padding_mask=None,
        need_attn=False,
        need_head_weights=False,
    ):
        not_initialized = ('adaptive-profiling' == self.args.init_type) and (1.0 == self.self_ratio_change.min()) and self.training

        if need_head_weights:
            need_attn = True

        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)

        # if self.args.mixed_precision: 
        #     x = x.type(self.in_type)
        if prev_self_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state = {"prev_key": prev_key, "prev_value": prev_value}
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        if self.cross_self_attention and not (incremental_state is not None and "prev_key" in self.self_attn._get_input_buffer(incremental_state)):
            if self_attn_mask is not None:
                self_attn_mask = torch.cat((x.new(x.size(0), encoder_out.size(0)).zero_(), self_attn_mask), dim=1)
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    encoder_padding_mask = self_attn_padding_mask.new(encoder_out.size(1), encoder_out.size(0)).zero_()
                self_attn_padding_mask = torch.cat((encoder_padding_mask, self_attn_padding_mask), dim=1)
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x
        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)

        # if self.args.mixed_precision:
        #     x = x.float()
        if 'adaptive' in self.args.init_type:
            if not_initialized:
                global decoder_ratio, tmp_file
                tmp_layer_ind = self.layer_num * 3 + 1
                tmp_ratio = decoder_ratio
                tmp_file.write('{} {}\n'.format(tmp_layer_ind, tmp_ratio))
                self.self_ratio_change.data.fill_(tmp_ratio)
                print ('decoder self attn ratio: {}'.format(tmp_ratio))
                input_std = np.var( (residual*self.self_ratio_change).clone().cpu().float().data.contiguous().view(-1).numpy())
                output_std = np.var(x.clone().cpu().float().data.contiguous().view(-1).numpy())
                decoder_ratio = np.sqrt(input_std + output_std)
            x0 = x + residual * self.self_ratio_change
        elif self.deepnet:
            x0 = x + residual * self.self_ratio_change
        elif self.rezero_weight is not None:
            x0 = residual + self.rezero_weight * x
        else:
            x0 = residual + x
        x0 = self.maybe_layer_norm(self.self_attn_layer_norm, x0, after=True)
        if self.args.plot_gradient:
            x0.register_hook(lambda grad: print('{} decoder s-att: {}'.format(self.layer_num, grad.norm().item())))
        x = x0
        if self.encoder_attn is not None:
            residual = x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x0, before=True)

            # if self.args.mixed_precision: 
            #     x = x.type(self.in_type)
            if prev_attn_state is not None:
                if incremental_state is None:
                    incremental_state = {}
                prev_key, prev_value = prev_attn_state[:2]
                saved_state = {"prev_key": prev_key, "prev_value": prev_value}
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)
            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = F.dropout(x, p=self.dropout, training=self.training)

            # if self.args.mixed_precision:
            #     x = x.float()
            if 'adaptive' in self.args.init_type:
                if not_initialized:
                    tmp_layer_ind = self.layer_num * 3 + 2
                    tmp_ratio = decoder_ratio
                    tmp_file.write('{} {}\n'.format(tmp_layer_ind, tmp_ratio))
                    self.encoder_ratio_change.data.fill_(tmp_ratio)
                    print ('decoder encoder attn ratio: {}'.format(tmp_ratio))
                    input_std = np.var( (residual*self.encoder_ratio_change).clone().cpu().float().data.contiguous().view(-1).numpy())
                    output_std = np.var(x.clone().cpu().float().data.contiguous().view(-1).numpy())
                    decoder_ratio = np.sqrt(input_std + output_std)
                x1 = x + residual * self.encoder_ratio_change
            elif self.deepnet:
                x1 = x + residual * self.encoder_ratio_change
            elif self.rezero_weight is not None:
                x1 = residual + self.rezero_weight * x
            else:
                x1 = residual + x
            x1 = self.maybe_layer_norm(self.encoder_attn_layer_norm, x1, after=True)
            if self.args.plot_gradient:
                x1.register_hook(lambda grad: print('{} decoder e-att: {}'.format(self.layer_num, grad.norm().item())))
            x = x1
        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)

        # if self.args.mixed_precision: 
        #     x = x.type(self.in_type)
        bx = self.fc1(x)
        hx = self.activation_fn(bx)
        x = F.dropout(hx, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # if self.args.mixed_precision:
        #     x = x.float()
        if 'adaptive' in self.args.init_type:
            if not_initialized:
                tmp_layer_ind = self.layer_num * 3 + 3
                tmp_ratio = decoder_ratio
                tmp_file.write('{} {}\n'.format(tmp_layer_ind, tmp_ratio))
                self.fc_ratio_change.data.fill_(tmp_ratio)
                print ('decoder ffn ratio: {}'.format(tmp_ratio))
                input_var = np.var( (residual * self.fc_ratio_change) .clone().cpu().float().data.contiguous().view(-1).numpy())
                output_var = np.var(x.clone().cpu().float().data.contiguous().view(-1).numpy())
                decoder_ratio = np.sqrt(input_var + output_var)
            x2 = x + residual * self.fc_ratio_change
        elif self.deepnet:
            x2 = x + residual * self.fc_ratio_change
        elif self.rezero_weight is not None:
            x2 = residual + self.rezero_weight * x
        else:
            x2 = residual + x
        x2 = self.maybe_layer_norm(self.final_layer_norm, x2, after=True)
        if self.args.plot_gradient:
            x2.register_hook(lambda grad: print('{} decoder ffn: {}'.format(self.layer_num, grad.norm().item())))

        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            if self_attn_padding_mask is not None:
                self_attn_state = saved_state["prev_key"], saved_state["prev_value"], saved_state["prev_key_padding_mask"]
            else:
                self_attn_state = saved_state["prev_key"], saved_state["prev_value"]
            return x2, attn, self_attn_state

        return x2, attn

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        if self.args.init_type == 'rezero':
            return x
        
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m
