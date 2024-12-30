import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import Tensor

import numpy as np

class TransformerEDADiarization(nn.Module):
    def __init__(self, n_speakers, in_size, n_units, n_heads, n_layers, dropout,
                 attractor_loss_ratio=1.0,
                 attractor_encoder_dropout=0.1,
                 attractor_decoder_dropout=0.1):
        """ Self-attention-based diarization model.

        Args:
          in_size (int): Dimension of input feature vector
          n_units (int): Number of units in a self-attention block
          n_heads (int): Number of attention heads
          n_layers (int): Number of transformer-encoder layers
          dropout (float): dropout ratio
          attractor_loss_ratio (float)
          attractor_encoder_dropout (float)
          attractor_decoder_dropout (float)
        """
        super(TransformerEDADiarization, self).__init__()
        self.n_speakers = n_speakers
        self.enc = TransformerModel(
            in_size, n_heads, n_units, n_layers, dropout=dropout
        )
        self.eda = EncoderDecoderAttractor(
            n_units,
            encoder_dropout=attractor_encoder_dropout,
            decoder_dropout=attractor_decoder_dropout,
        )
        self.attractor_loss_ratio = attractor_loss_ratio

    def forward(self, src, tgt, ilens):
        n_speakers = [t.shape[1] for t in tgt]
        # emb: (B, T, E)
        emb = self.enc(src)
        # emb  = [e[:ilen] for e, ilen in zip(emb, ilens)]
        # emb = nn.utils.rnn.pad_sequence(emb, padding_value=0, batch_first=True)
        # attractors: (B, C, E)
        attractor_loss, attractors = self.eda(emb, n_speakers)
        output = torch.bmm(emb, attractors.transpose(1, 2))
        output = [out[:ilen, :n_spk] for out, ilen, n_spk in zip(output, ilens, n_speakers)]
        # output: [(T', C')]
        # output = [y[:ilen, :n_spk] for y, ilen, n_spk in zip(output_bch, ilens, n_speakers)]
        # Version for fixed speakers
        return output, self.attractor_loss_ratio * attractor_loss, emb, attractors[:, :-1, :]
    
    def test(self, src, ilens, **kwargs):
        n_spk = kwargs.get('n_spk')
        th = kwargs.get('th')
        # emb: (B, T, E)
        emb = self.enc(src)
        # emb  = [e[:ilen] for e, ilen in zip(emb, ilens)]
        # emb = nn.utils.rnn.pad_sequence(emb, padding_value=0, batch_first=True)
        # attractors: (B, C, E) probs: (B, C)
        order = np.arange(emb.shape[1])
        np.random.shuffle(order)
        # print(emb.shape)
        attractors, probs = self.eda.estimate(emb[:, order, :])
        output = torch.bmm(emb, attractors.transpose(1, 2))
        output_active = []
        for p, y, ilen in zip(probs, output, ilens):
            if n_spk is not None:
                output_active.append(y[:ilen, :n_spk])
            elif th is not None:
                silence = torch.where(p < th)[0]
                n_spk = silence[0] if silence.size else None
                # n_spk = min(n_spk, 5)
                output_active.append(y[:ilen, :n_spk])
        return output_active, emb, attractors[:, :-1, :]


class EncoderDecoderAttractor(nn.Module):
    def __init__(self, n_units, encoder_dropout=0.1, decoder_dropout=0.1):
        super(EncoderDecoderAttractor, self).__init__()
        self.encoder = nn.LSTM(n_units, n_units, 1, batch_first=True, dropout=encoder_dropout)
        self.decoder = nn.LSTM(n_units, n_units, 1, batch_first=True, dropout=decoder_dropout)
        self.counter = nn.Linear(n_units, 1)
        self.n_units = n_units
    
    def eda_forward(self, xs, zeros):
        _, (hn, cn) = self.encoder(xs)
        # c_zeros = torch.zeros(hn.shape, dtype=xs.dtype, device=xs.device)
        # attractors, _ = self.decoder(zeros, (hn, c_zeros))
        attractors, _ = self.decoder(zeros, (hn, cn))
        return attractors
    
    def estimate(self, xs, max_n_speakers=15):
        r"""
        Calculate attractors from embedding sequences
        without prior knowledge of number of speakers

        Args:
        xs:
         
        """
        # zeros: (B, C, E)
        zeros = torch.zeros((xs.shape[0], max_n_speakers, xs.shape[-1]), dtype=xs.dtype, device=xs.device)
        attractors = self.eda_forward(xs, zeros)
        probs = torch.sigmoid(self.counter(attractors).squeeze(dim=-1))
        return attractors, probs

    def forward(self, xs, n_speakers) -> tuple[(Tensor, Tensor)]:
        r"""
        Calculate attractors from embedding sequences with given number of speakers

        Args:
        xs: (B, T, E)
        n_speakers: list of number of speakers in batch
        """
        # zeros: (B, C, E)
        zeros = torch.zeros((xs.shape[0], max(n_speakers) + 1, xs.shape[-1]), dtype=xs.dtype, device=xs.device)
        attractors = self.eda_forward(xs, zeros)
        # attractors: (B, C, E)
        labels = torch.cat([torch.tensor([[1] * n_spk + [0]], dtype=torch.float32, device=xs.device) for n_spk in n_speakers], dim=1)
        logit = torch.cat([(self.counter(att[:n_spk+1, :])).reshape(-1, n_spk + 1) for (att, n_spk) in zip(attractors, n_speakers)], dim=1)
        loss = F.binary_cross_entropy_with_logits(logit, labels)

        # attractors = [att[:n_spk, :] for (att, n_spk) in zip(attractors, n_speakers)]
        # attractors: (B, C, E)
        return loss, attractors


class TransformerModel(nn.Module):
    def __init__(self, in_size, n_heads, n_units, n_layers, dim_feedforward=2048, dropout=0.5, has_pos=False):
        """ Self-attention-based diarization model.

        Args:
          n_speakers (int): Number of speakers in recording
          in_size (int): Dimension of input feature vector
          n_heads (int): Number of attention heads
          n_units (int): Number of units in a self-attention block
          n_layers (int): Number of transformer-encoder layers
          dropout (float): dropout ratio
        """
        super(TransformerModel, self).__init__()
        self.in_size = in_size
        self.n_heads = n_heads
        self.n_units = n_units
        self.n_layers = n_layers
        self.has_pos = has_pos

        self.src_mask = None
        self.encoder = nn.Linear(in_size, n_units)
        self.encoder_norm = nn.LayerNorm(n_units)
        if self.has_pos:
            self.pos_encoder = PositionalEncoding(n_units, dropout)
        encoder_layers = TransformerEncoderLayer(n_units, n_heads, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.bias.data.zero_()
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, has_mask=False, activation=None):
        self.src_mask = None

        ilens = [x.shape[0] for x in src]
        src = nn.utils.rnn.pad_sequence(src, padding_value=-1, batch_first=True)

        # src: (B, T, E)
        src = self.encoder(src)
        src = self.encoder_norm(src)
        # src: (T, B, E)
        src = src.transpose(0, 1)
        if self.has_pos:
            # src: (T, B, E)
            src = self.pos_encoder(src)
        # output: (T, B, E)
        output = self.transformer_encoder(src, self.src_mask)
        # output: (B, T, E)
        output = output.transpose(0, 1)

        if activation:
            output = activation(output)

        # output = [out[:ilen] for out, ilen in zip(output, ilens)]

        return output

    def get_attention_weight(self, src):
        # NOTE: NOT IMPLEMENTED CORRECTLY!!!
        attn_weight = []
        def hook(module, input, output):
            # attn_output, attn_output_weights = multihead_attn(query, key, value)
            # output[1] are the attention weights
            attn_weight.append(output[1])
            
        handles = []
        for l in range(self.n_layers):
            handles.append(self.transformer_encoder.layers[l].self_attn.register_forward_hook(hook))

        self.eval()
        with torch.no_grad():
            self.forward(src)

        for handle in handles:
            handle.remove()
        self.train()

        return torch.stack(attn_weight)


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
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional information to each time step of x
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


if __name__ == "__main__":
    import torch
    model = TransformerModel(5, 40, 4, 512, 2, 0.1)
    input = torch.randn(8, 500, 40)
    print("Model output:", model(input).size())
    print("Model attention:", model.get_attention_weight(input).size())
    print("Model attention sum:", model.get_attention_weight(input)[0][0][0].sum())
