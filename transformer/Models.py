import numpy as np
import torch
import torch.nn as nn

import transformer.Constants as Constants
from .Layers import FFTBlock
from text.symbols import symbols


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    def angle(pos, idx):
        return pos / np.power(10000, 2 * (idx // 2) / d_hid)

    def angle_vec(pos):
        return [angle(pos, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    if padding_idx is not None:
        sinusoid_table[padding_idx] = 0.0
    return torch.FloatTensor(sinusoid_table)


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        n_pos = config["max_seq_len"] + 1
        n_vocab = len(symbols) + 1
        n_layers = config["transformer"]["encoder_layer"]
        n_head = config["transformer"]["encoder_head"]
        word_vec = config["transformer"]["encoder_hidden"]
        d_k = d_v = (
            config["transformer"]["encoder_hidden"]
            // config["transformer"]["encoder_head"]
        )
        d_model = config["transformer"]["encoder_hidden"]
        d_in = config["transformer"]["conv_filter_size"]
        kernel_sz = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["encoder_dropout"]

        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model

        self.src_word_emb = nn.Embedding(n_vocab, word_vec, padding_idx=Constants.PAD)
        self.position_enc = nn.Parameter(get_sinusoid_encoding_table(n_pos, word_vec).unsqueeze(0), requires_grad=False)

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_in, kernel_sz, dropout=dropout
                )
                for i in range(n_layers)
            ]
        )

    def forward(self, src_seq, mask, return_attns=False):
        attn_list = []
        b, max_len = src_seq.shape[0], src_seq.shape[1]
        attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
        if not self.training and src_seq.shape[1] > self.max_seq_len:
            out = self.src_word_emb(src_seq) + get_sinusoid_encoding_table(
                src_seq.shape[1], self.d_model
            )[: src_seq.shape[1], :].unsqueeze(0).expand(b, -1, -1).to(src_seq.device)
        else:
            out = self.src_word_emb(src_seq) + self.position_enc[:, :max_len, :].expand(b, -1, -1)

        for layer in self.layer_stack:
            out, attn = layer(out, mask=mask, slf_attn_mask=attn_mask)
            if return_attns:
                attn_list += [attn]

        return out


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        n_pos = config["max_seq_len"] + 1
        word_vec = config["transformer"]["decoder_hidden"]
        n_layers = config["transformer"]["decoder_layer"]
        n_head = config["transformer"]["decoder_head"]
        d_k = d_v = (
            config["transformer"]["decoder_hidden"]
            // config["transformer"]["decoder_head"]
        )
        d_model = config["transformer"]["decoder_hidden"]
        d_in = config["transformer"]["conv_filter_size"]
        kernel_sz = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["decoder_dropout"]

        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model

        self.position_enc = nn.Parameter(get_sinusoid_encoding_table(n_pos, word_vec).unsqueeze(0), requires_grad=False)

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_in, kernel_sz, dropout=dropout
                )
                for i in range(n_layers)
            ]
        )

    def forward(self, enc_seq, mask, return_attns=False):
        attn_list = []
        b, max_len = enc_seq.shape[0], enc_seq.shape[1]
        if not self.training and enc_seq.shape[1] > self.max_seq_len:
            attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            out = enc_seq + get_sinusoid_encoding_table(
                enc_seq.shape[1], self.d_model
            )[: enc_seq.shape[1], :].unsqueeze(0).expand(b, -1, -1).to(enc_seq.device)
        else:
            max_len = min(max_len, self.max_seq_len)
            attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            out = enc_seq[:, :max_len, :] + self.position_enc[
                :, :max_len, :
            ].expand(b, -1, -1)
            mask = mask[:, :max_len]
            attn_mask = attn_mask[:, :, :max_len]

        for layer in self.layer_stack:
            out, attn = layer(out, mask=mask, slf_attn_mask=attn_mask)
            if return_attns:
                attn_list += [dec_slf_attn]

        return out, mask
