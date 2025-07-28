import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from ..modules.merge_tfm_encoder import TransformerEncoderFusionLayer, TransformerEncoder
from torch import Tensor
from typing import Optional, Any, Union, Callable

class OnlineTransformerDADiarization(nn.Module):
    def __init__(self, n_speakers, in_size, n_units, n_heads, enc_n_layers, dec_n_layers, dropout, has_mask, max_seqlen, dec_dim_feedforward, conv_delay=9, mask_delay=0, decom_kernel_size=64):
        """ Self-attention-based diarization model.

        Args:
          in_size (int): Dimension of input feature vector
          n_units (int): Number of units in a self-attention block
          n_heads (int): Number of attention heads
          n_layers (int): Number of transformer-encoder layers
          dropout (float): dropout ratio
        """
        super(OnlineTransformerDADiarization, self).__init__()
        self.n_speakers = n_speakers
        self.delay = conv_delay
        self.enc = MaskedTransformerEncoderModel(
            in_size, n_heads, n_units, enc_n_layers, dropout=dropout, has_mask=has_mask, max_seqlen=max_seqlen, mask_delay=mask_delay
        )
        self.dec = MaskedTransformerDecoderModel(
            in_size, n_heads, n_units, dec_n_layers, dim_feedforward=dec_dim_feedforward, dropout=dropout, has_mask=has_mask, max_seqlen=max_seqlen, mask_delay=mask_delay
        )
        self.cnn = nn.Conv1d(n_units, n_units, kernel_size=2 * conv_delay + 1, padding=9)

    def forward(self, src, tgt, ilens):
        n_speakers = [t.shape[1] for t in tgt]
        max_nspks = max(n_speakers)
        # emb: (B, T, E)
        emb = self.enc(src)
        B, T, D = emb.shape
        emb  = [e[:ilen] for e, ilen in zip(emb, ilens)]
        emb = nn.utils.rnn.pad_sequence(emb, padding_value=0, batch_first=True)
        emb: Tensor = self.cnn(emb.transpose(1,2)).transpose(1,2) # (B, T, D)
        emb = emb / torch.norm(emb, dim=-1, keepdim=True)
        attractors = self.dec(emb, max_nspks)
        attractors = attractors / torch.norm(attractors, dim=-1, keepdim=True)

        # Calculate emb consistency loss (cosine similarity)
        attn_map = emb.matmul(emb.transpose(-1, -2))
        # att_norm: (B, T, 1)
        attn_norm = torch.norm(emb, dim=-1, keepdim=True)
        attn_norm = attn_norm.matmul(attn_norm.transpose(-1, -2))
        attn_map = attn_map / (attn_norm + 1e-6)
        tgt_pad = [F.pad(t, (0, max_nspks-t.shape[1]), "constant", 0) for t in tgt]
        tgt_pad = nn.utils.rnn.pad_sequence(tgt_pad, padding_value=0, batch_first=True)
        label_map = tgt_pad.matmul(tgt_pad.transpose(-1, -2))
        tgt_norm = torch.norm(tgt_pad, dim=-1, keepdim=True)
        tgt_norm = tgt_norm.matmul(tgt_norm.transpose(-1, -2))
        label_map = label_map / (tgt_norm + 1e-6)
        emb_consis_loss = F.mse_loss(attn_map, label_map)

        # output: (B, T, C)
        output = torch.matmul(emb.unsqueeze(dim=-2), attractors.transpose(-1, -2)).squeeze(dim=-2)
        output = [out[:ilen, :n_spk] for out, ilen, n_spk in zip(output, ilens, n_speakers)]

        emb = [e[:ilen] for e, ilen in zip(emb, ilens)]
        attractors = [attr[:ilen, 1:n_spk] for attr, ilen, n_spk in zip(attractors, ilens, n_speakers)]
        return output, emb_consis_loss, emb, attractors
    
    def test(self, src, ilens, max_nspks=6):
        # emb: (B, T, E)
        emb = self.enc(src)
        B, T, D = emb.shape
        emb  = [e[:ilen] for e, ilen in zip(emb, ilens)]
        emb = nn.utils.rnn.pad_sequence(emb, padding_value=0, batch_first=True)
        emb: Tensor = self.cnn(emb.transpose(1,2)).transpose(1,2) # (B, T, D)
        emb = emb / torch.norm(emb, dim=-1, keepdim=True)
        attractors = self.dec(emb, max_nspks)
        attractors = attractors / torch.norm(attractors, dim=-1, keepdim=True)
        
        # output: (B, T, C)
        output = torch.matmul(emb.unsqueeze(dim=-2), attractors.transpose(-1, -2)).squeeze(dim=-2)
        output = [out[:ilen] for out, ilen in zip(output, ilens)]

        emb = [e[:ilen] for e, ilen in zip(emb, ilens)]
        attractors = [attr[:ilen] for attr, ilen in zip(attractors, ilens)]
        return output, emb, attractors


class MaskedTransformerDecoderModel(nn.Module):
    def __init__(self, in_size, n_heads, n_units, n_layers, dim_feedforward, dropout=0.5, has_mask=False, max_seqlen = 500, has_pos=False, mask_delay=0):
        super(MaskedTransformerDecoderModel, self).__init__()
        self.in_size = in_size
        self.n_heads = n_heads
        self.n_units = n_units
        self.n_layers = n_layers
        self.has_pos = has_pos
        self.has_mask = has_mask
        self.max_seqlen = max_seqlen
        self.mask_delay = mask_delay

        self.encoder = nn.Linear(in_size, n_units)
        self.encoder_norm = nn.LayerNorm(n_units)

        self.pos_enc = PositionalEncoding(n_units, dropout)
        self.convert = nn.Linear(n_units * 2, n_units)
        decoder_layers = TransformerEncoderFusionLayer(n_units, n_heads, dim_feedforward, dropout, batch_first=True)
        self.attractor_decoder = TransformerEncoder(decoder_layers, n_layers)

    def _generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device), diagonal=-self.mask_delay) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, emb: Tensor, max_nspks: int, activation: Optional[Callable]=None):
        pos_enc = self.pos_enc(emb, max_nspks) # (B, T, C, D)
        attractors_init: Tensor = self.convert(torch.cat([emb.unsqueeze(dim=2).repeat(1, 1, max_nspks, 1), pos_enc], dim=-1))
        
        t_mask = self._generate_square_subsequent_mask(emb.shape[1], emb.device)
        attractors = self.attractor_decoder(attractors_init, t_mask)
        return attractors

class MaskedTransformerEncoderModel(nn.Module):
    def __init__(self, in_size, n_heads, n_units, n_layers, dim_feedforward=2048, dropout=0.5, has_mask=False, max_seqlen = 500, has_pos=False, mask_delay=0):
        """ Self-attention-based diarization model.

        Args:
          n_speakers (int): Number of speakers in recording
          in_size (int): Dimension of input feature vector
          n_heads (int): Number of attention heads
          n_units (int): Number of units in a self-attention block
          n_layers (int): Number of transformer-encoder layers
          dropout (float): dropout ratio
        """
        super(MaskedTransformerEncoderModel, self).__init__()
        self.in_size = in_size
        self.n_heads = n_heads
        self.n_units = n_units
        self.n_layers = n_layers
        self.has_pos = has_pos
        self.has_mask = has_mask
        self.max_seqlen = max_seqlen
        self.mask_delay = mask_delay

        self.bn = nn.BatchNorm1d(in_size)
        self.encoder = nn.Linear(in_size, n_units)
        self.encoder_norm = nn.LayerNorm(n_units)
        if self.has_pos:
            self.pos_encoder = PositionalEncoding(n_units, dropout)
        encoder_layers = TransformerEncoderLayer(n_units, n_heads, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device), diagonal=-self.mask_delay) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.bias.data.zero_()
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, activation=None):

        ilens = [x.shape[0] for x in src]
        src = nn.utils.rnn.pad_sequence(src, padding_value=-1, batch_first=True)
        src = self.bn(src.transpose(1, 2)).transpose(1, 2)

        src_mask = None
        if self.has_mask:
            src_mask = self._generate_square_subsequent_mask(src.shape[1], src.device)

        # src: (B, T, E)
        src = self.encoder(src)
        src = self.encoder_norm(src)
        # src: (T, B, E)
        src = src.transpose(0, 1)
        if self.has_pos:
            # src: (T, B, E)
            src = self.pos_encoder(src)
        # output: (T, B, E)
        output = self.transformer_encoder(src, src_mask)
        # output: (B, T, E)
        output = output.transpose(0, 1)

        if activation:
            output = activation(output)

        return output

class PositionalEncoding(nn.Module):
    """Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor, max_nspks):
        # Add positional information to each time step of x
        pe = self.pe[:, :max_nspks, :]
        pe = pe.unsqueeze(dim=0).repeat(x.shape[0], x.shape[1], 1, 1) # (B, T, C, D)
        x = x.unsqueeze(dim=2).repeat(1, 1, max_nspks, 1)
        # x = x + pe
        return pe
