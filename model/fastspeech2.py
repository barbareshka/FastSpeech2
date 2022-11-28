import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor
from utils.tools import get_mask_from_lengths


class FastSpeech2(nn.Module):
    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        self.config = model_config

        self.encoder = Encoder(model_config)
        self.adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_lin = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()
        self.speaker_emb = None

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = None
        if mel_lens is not None:
            mel_masks = get_mask_from_lengths(mel_lens, max_mel_len)
        encoded = self.encoder(texts, src_masks)

        (
            out, p_pred, e_pred, log_d_pred, d_rounded, mel_lens, mel_masks,
        ) = self.adaptor(
            encoded, src_masks, mel_masks, max_mel_len, p_targets, e_targets,
            d_targets, p_control, e_control, d_control,
        )

        out, mel_masks = self.decoder(out, mel_masks)
        out = self.mel_lin(out)

        p_out = self.postnet(out) + out

        return (
            out, p_out, p_pred, e_pred, log_d_pred, d_rounded,
            src_masks, mel_masks, src_lens, mel_lens,
        )
