import copy
import collections
import json
import math
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.tools import get_mask_from_lengths, pad

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Conv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        out = x.contiguous().transpose(1, 2)
        out = self.conv(out)
        out = out.contiguous().transpose(1, 2)
        return out


class VarianceAdaptor(nn.Module):

    def __init__(self, preprocess_config, model_config):
        super(VarianceAdaptor, self).__init__()
        self.d_pred = VariancePredictor(model_config)
        self.p_pred = VariancePredictor(model_config)
        self.e_pred = VariancePredictor(model_config)
        self.lr = LengthRegulator()
        self.p_fl = preprocess_config["preprocessing"]["pitch"]["feature"]
        self.e_fl = preprocess_config["preprocessing"]["energy"]["feature"]
        assert self.p_fl in ["phoneme_level", "frame_level"]
        assert self.e_fl in ["phoneme_level", "frame_level"]
        p_quant = model_config["variance_embedding"]["pitch_quantization"]
        e_quant = model_config["variance_embedding"]["energy_quantization"]
        n_bins = model_config["variance_embedding"]["n_bins"]
        assert p_quant in ["linear", "log"]
        assert e_quant in ["linear", "log"]

        with open(
            os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            e_min, e_max = stats["energy"][:2]
            p_min, p_max = stats["pitch"][:2]      
        if e_quant == "log":
            self.e_bins = nn.Parameter(
                torch.exp(torch.linspace(np.log(e_min), np.log(e_max), n_bins - 1)), 
                requires_grad=False,
            )
        else:
            self.e_bins = nn.Parameter(torch.linspace(e_min, e_max, n_bins - 1), requires_grad=False)
        if p_quant == "log":
            self.p_bins = nn.Parameter(
                torch.exp(torch.linspace(np.log(p_min), np.log(p_max), n_bins - 1)),
                requires_grad=False,
            )
        else:
            self.p_bins = nn.Parameter(torch.linspace(p_min, p_max, n_bins - 1), requires_grad=False)
        self.p_emb = nn.Embedding(n_bins, model_config["transformer"]["encoder_hidden"])
        self.e_emb = nn.Embedding(n_bins, model_config["transformer"]["encoder_hidden"])

    def get_pitch_embedding(self, x, target, mask, control):
        pred = self.p_pred(x, mask)
        if target is None:
            pred = pred * control
            emb = self.p_emb(torch.bucketize(pred, self.p_bins))
        else:
            emb = self.p_emb(torch.bucketize(target, self.p_bins))
        return pred, emb

    def get_energy_embedding(self, x, target, mask, control):
        pred = self.e_pred(x, mask)
        if target is None:
            pred = pred * control
            emb = self.e_emb(torch.bucketize(pred, self.e_bins))
        else:
            emb = self.e_emb(torch.bucketize(target, self.e_bins))
            
        return pred, emb

    def forward(
        self,
        x,
        src_mask,
        mel_mask=None,
        max_len=None,
        pitch_target=None,
        energy_target=None,
        duration_target=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):

        log_d_pred = self.d_pred(x, src_mask)
        if self.p_fl == "phoneme_level":
            p_pred, p_emb = self.get_pitch_embedding(x, pitch_target, src_mask, p_control)
            x += p_emb
        if self.e_fl == "phoneme_level":
            e_pred, e_emb = self.get_energy_embedding(x, energy_target, src_mask, p_control)
            x += e_emb

        if duration_target is not None:
            x, mel_len = self.lr(x, duration_target, max_len)
            d_rounded = duration_target
        else:
            d_rounded = torch.clamp((torch.round(torch.exp(log_d_pred) - 1) * d_control), min=0)
            x, mel_len = self.lr(x, d_rounded, max_len)
            mel_mask = get_mask_from_lengths(mel_len)

        if self.p_fl == "frame_level":
            p_pred, p_emb = self.get_pitch_embedding(x, pitch_target, mel_mask, p_control)
            x += p_emb
        if self.e_fl == "frame_level":
            e_pred, e_emb = self.get_energy_embedding(x, energy_target, mel_mask, p_control)
            x += e_emb

        return (
            x,
            p_pred,
            e_pred,
            log_d_pred,
            d_rounded,
            mel_len,
            mel_mask,
        )


class LengthRegulator(nn.Module):
    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        out = list()
        mel_len = list()
        for b, tar in zip(x, duration):
            expanded = self.expand(b, tar)
            out.append(expanded)
            mel_len.append(expanded.shape[0])
        if max_len is None:
            out = pad(out)
        else:
            out = pad(out, max_len)
        return out, torch.LongTensor(mel_len).to(device)

    def expand(self, b, pred):
        out = list()
        for i, j in enumerate(b):
            sz = predicted[i].item()
            out.append(j.expand(max(int(sz), 0), -1))
        out = torch.cat(out, 0)
        return out

    def forward(self, x, duration, max_len):
        out, mel_len = self.LR(x, duration, max_len)
        return out, mel_len


class VariancePredictor(nn.Module):
    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()

        self.input_sz = model_config["transformer"]["encoder_hidden"]
        self.filter_sz = model_config["variance_predictor"]["filter_size"]
        self.kernel = model_config["variance_predictor"]["kernel_size"]
        self.conv_sz = model_config["variance_predictor"]["filter_size"]
        self.dropout = model_config["variance_predictor"]["dropout"]

        self.conv_layer = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_sz,
                            self.filter_sz,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_sz)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_sz,
                            self.filter_sz,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_sz)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.ll = nn.Linear(self.conv_sz, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.ll(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out
