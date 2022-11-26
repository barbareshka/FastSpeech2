import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from .support import VarianceAdaptor, Encoder, Decoder, PostNet
from .funcs import get_mask_from_lengths


class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )

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

        encoded_output = self.encoder(texts, src_masks)

        if self.speaker_emb is not None:
            encoded_output = encoded_output + self.speaker_emb(speakers).unsqueeze(1).expand(-1, max_src_len, -1)

        (
            adapted_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            adapted_mel_lens,
            adapted_mel_masks,
        ) = self.variance_adaptor(
            encoded_output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        decoded_output, decoded_mel_masks = self.decoder(adapted, adapted_mel_masks)
        linear_output = self.mel_linear(decoded_output)

        postnet_output = self.postnet(linear_output) + linear_output

        return (
            linear_output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )
