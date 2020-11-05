#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import numpy as np
import torch
from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion
from examples.speech_recognition.data.replabels import pack_replabels

from wav2letter.criterion import ASGLoss, CriterionScaleMode


@register_criterion("asg_loss")
class ASGCriterion(FairseqCriterion):
    @staticmethod
    def add_args(parser):
        """
        Add command line arguments.

        Args:
            parser: (todo): write your description
        """
        group = parser.add_argument_group("ASG Loss")
        group.add_argument(
            "--asg-transitions-init",
            help="initial diagonal value of transition matrix",
            type=float,
            default=0.0,
        )
        group.add_argument(
            "--max-replabel", help="maximum # of replabels", type=int, default=2
        )
        group.add_argument(
            "--linseg-updates",
            help="# of training updates to use LinSeg initialization",
            type=int,
            default=0,
        )
        group.add_argument(
            "--hide-linseg-messages",
            help="hide messages about LinSeg initialization",
            action="store_true",
        )

    def __init__(self, args, task):
        """
        Initialize the target state.

        Args:
            self: (todo): write your description
            task: (str): write your description
        """
        super().__init__(args, task)
        self.tgt_dict = task.target_dictionary
        self.eos = self.tgt_dict.eos()
        self.silence = (
            self.tgt_dict.index(args.silence_token)
            if args.silence_token in self.tgt_dict
            else None
        )
        self.max_replabel = args.max_replabel

        num_labels = len(self.tgt_dict)
        self.asg = ASGLoss(num_labels, scale_mode=CriterionScaleMode.TARGET_SZ_SQRT)
        self.asg.trans = torch.nn.Parameter(
            args.asg_transitions_init * torch.eye(num_labels), requires_grad=True
        )

        self.linseg_progress = torch.nn.Parameter(
            torch.tensor([0], dtype=torch.int), requires_grad=False
        )
        self.linseg_maximum = args.linseg_updates
        self.linseg_message_state = "none" if args.hide_linseg_messages else "start"

    def linseg_step(self):
        """
        Initialize progress.

        Args:
            self: (todo): write your description
        """
        if not self.training:
            return False
        if self.linseg_progress.item() < self.linseg_maximum:
            if self.linseg_message_state == "start":
                print("| using LinSeg to initialize ASG")
                self.linseg_message_state = "finish"
            self.linseg_progress.add_(1)
            return True
        elif self.linseg_message_state == "finish":
            print("| finished LinSeg initialization")
            self.linseg_message_state = "none"
        return False

    def replace_eos_with_silence(self, tgt):
        """
        Replace eos with eos.

        Args:
            self: (todo): write your description
            tgt: (todo): write your description
        """
        if tgt[-1] != self.eos:
            return tgt
        elif self.silence is None or (len(tgt) > 1 and tgt[-2] == self.silence):
            return tgt[:-1]
        else:
            return tgt[:-1] + [self.silence]

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        net_output = model(**sample["net_input"])
        emissions = net_output["encoder_out"].transpose(0, 1).contiguous()
        B = emissions.size(0)
        T = emissions.size(1)
        device = emissions.device

        target = torch.IntTensor(B, T)
        target_size = torch.IntTensor(B)
        using_linseg = self.linseg_step()

        for b in range(B):
            initial_target_size = sample["target_lengths"][b].item()
            if initial_target_size == 0:
                raise ValueError("target size cannot be zero")

            tgt = sample["target"][b, :initial_target_size].tolist()
            tgt = self.replace_eos_with_silence(tgt)
            tgt = pack_replabels(tgt, self.tgt_dict, self.max_replabel)
            tgt = tgt[:T]

            if using_linseg:
                tgt = [tgt[t * len(tgt) // T] for t in range(T)]

            target[b][: len(tgt)] = torch.IntTensor(tgt)
            target_size[b] = len(tgt)

        loss = self.asg.forward(emissions, target.to(device), target_size.to(device))

        if reduce:
            loss = torch.sum(loss)

        sample_size = (
            sample["target"].size(0) if self.args.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": utils.item(loss.data) if reduce else loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        agg_output = {
            "loss": loss_sum / nsentences,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }
        return agg_output
