#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Adapted from https://github.com/elbayadm/attn2d/blob/master/examples/waitk
# Implementation of the papers:
#   *Efficient Wait-k Models for Simultaneous Machine Translation
#       http://www.interspeech2020.org/uploadfile/pdf/Tue-1-1-2.pdf

import logging
from fairseq import checkpoint_utils

from fairseq.models import (
    register_model,
    register_model_architecture
)
from fairseq.models.transformer import (
    base_architecture
)
# user
from .waitk_transformer import (
    CausalTransformerEncoder,
    WaitkTransformerModel,
)
from .sinkhorn_encoder import (
    SinkhornCascadedEncoder,
)

logger = logging.getLogger(__name__)


@register_model("sinkhorn_waitk")
class SinkhornWaitkTransformerModel(WaitkTransformerModel):
    """
    Waitk transformer with sinkhorn encoder
    """

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        super(SinkhornWaitkTransformerModel,
              SinkhornWaitkTransformerModel).add_args(parser)
        parser.add_argument(
            "--non-causal-layers",
            type=int,
            help=(
                'number of layers for non-causal encoder.'
            ),
        )
        parser.add_argument(
            '--sinkhorn-tau',
            type=float,
            required=True,
            help='temperature for gumbel sinkhorn.'
        )
        parser.add_argument(
            "--sinkhorn-iters",
            type=int,
            required=True,
            help=(
                'iters of sinkhorn normalization to perform.'
            ),
        )
        parser.add_argument(
            "--sinkhorn-noise-factor",
            type=float,
            required=True,
            help=(
                'represents how many gumbel randomness in training.'
            ),
        )
        parser.add_argument(
            "--sinkhorn-bucket-size",
            type=int,
            required=True,
            help=(
                'number of elements to group before performing sinkhorn sorting.'
            ),
        )
        parser.add_argument(
            "--sinkhorn-energy",
            type=str,
            required=True,
            choices=["dot", "cos", "l2"],
            help=(
                'type of energy function to use to calculate attention. available: dot, cos, L2'
            ),
        )

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        encoder = CausalTransformerEncoder(args, src_dict, embed_tokens)
        if getattr(args, "load_pretrained_encoder_from", None):
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=args.load_pretrained_encoder_from
            )
            logger.info(
                f"loaded pretrained encoder from: "
                f"{args.load_pretrained_encoder_from}"
            )
        cascade = SinkhornCascadedEncoderSlice(args, encoder)
        return cascade

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        """ changed encoder forward to forward_train """
        encoder_out = self.encoder.forward_train(src_tokens=src_tokens, src_lengths=src_lengths)
        x, extra = self.decoder(
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            features_only=True,
        )
        extra["decoder_states"] = x
        extra["attn"] = encoder_out["attn"]
        extra["log_alpha"] = encoder_out["log_alpha"]
        logits = self.decoder.output_projection(x)
        return logits, extra


class SinkhornCascadedEncoderSlice(SinkhornCascadedEncoder):
    def slice_encoder_out(self, encoder_out, context_size):
        return self.causal_encoder.slice_encoder_out(encoder_out, context_size)

    def forward_train(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return super().forward_causal(*args, **kwargs)


@register_model_architecture(
    "sinkhorn_waitk", "sinkhorn_waitk"
)
def sinkhorn_waitk(args):
    args.waitk = getattr(args, 'waitk', 60000)  # default is wait-until-end

    args.max_source_positions = getattr(args, "max_source_positions", 1024)
    args.max_target_positions = getattr(args, "max_target_positions", 1024)

    args.non_causal_layers = getattr(args, "non_causal_layers", 2)
    args.dropout = getattr(args, "dropout", 0.1)

    args.share_decoder_input_output_embed = True
    args.upsample_ratio = 1
    args.delay = 1
    base_architecture(args)


@register_model_architecture(
    "sinkhorn_waitk", "sinkhorn_waitk_small"
)
def sinkhorn_waitk_small(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 5)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_normalize_before = getattr(
        args, "encoder_normalize_before", True)

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 256)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_layers = getattr(args, "decoder_layers", 5)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_normalize_before = getattr(
        args, "decoder_normalize_before", True)

    args.non_causal_layers = getattr(args, "non_causal_layers", 3)

    sinkhorn_waitk(args)
