import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..conformer.encoder import ConformerEncoder
from ..modules.merge_retnet_layer import TransformerEncoderFusionLayer
from torch import Tensor
from typing import Optional, Any, Union, Callable
from collections import deque

# import numpy as np
# torch.set_printoptions(threshold=np.inf)

class OnlineConformerRetentionDADiarization(nn.Module):
    def __init__(
            self, 
            n_speakers,
            in_size,
            n_units,
            n_heads,
            enc_n_layers,
            dec_n_layers,
            dropout,
            max_seqlen,
            recurrent_chunk_size: int = 500,
            feed_forward_expansion_factor: int = 8,
            dec_dim_feedforward: int = 2048,
            conv_expansion_factor: int = 2,
            conv_kernel_size: int = 16,
            half_step_residual: bool = True,
            conv_delay=9,
            mask_delay=0):
        """ Self-attention-based diarization model.

        Args:
          in_size (int): Dimension of input feature vector
          n_units (int): Number of units in a self-attention block
          n_heads (int): Number of attention heads
          n_layers (int): Number of transformer-encoder layers
          dropout (float): dropout ratio
        """
        super(OnlineConformerRetentionDADiarization, self).__init__()
        self.n_speakers = n_speakers
        self.n_units = n_units
        self.delay = conv_delay
        self.max_seqlen = max_seqlen
        self.recurrent_chunk_size = recurrent_chunk_size
        self.enc = EmbeddingEncoderModule(
            in_size=in_size,
            n_units=n_units,
            n_heads=n_heads,
            n_layers=enc_n_layers,
            recurrent_chunk_size=recurrent_chunk_size,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor=conv_expansion_factor,
            dropout=dropout,
            conv_kernel_size=conv_kernel_size,
            half_step_residual=half_step_residual,
            max_seqlen=max_seqlen,
        )
        self.dec = MaskedTransformerDecoderModel(
            in_size,
            n_heads=n_heads,
            n_units=n_units,
            n_layers=dec_n_layers,
            recurrent_chunk_size=recurrent_chunk_size,
            dim_feedforward=dec_dim_feedforward,
            dropout=dropout,
            max_seqlen=max_seqlen,
            mask_delay=mask_delay,
        )
        self.cnn = nn.Conv1d(n_units, n_units, kernel_size=2 * conv_delay + 1, padding=conv_delay)

    def forward(self, src, tgt, ilens):
        n_speakers = [t.shape[1] for t in tgt]
        max_nspks = max(n_speakers)
        # emb: (B, T, E)
        emb = self.enc(src)
        B, T, D = emb.shape
        emb  = [e[:ilen] for e, ilen in zip(emb, ilens)]
        emb = nn.utils.rnn.pad_sequence(emb, padding_value=0, batch_first=True)
        seq_len = emb.shape[1]
        pad_seqlen = math.ceil(seq_len / self.recurrent_chunk_size) * self.recurrent_chunk_size
        # pad_seqlen = max(math.ceil(seq_len / self.recurrent_chunk_size) * self.recurrent_chunk_size, self.max_seqlen)
        emb = F.pad(emb, (0, 0, 0, pad_seqlen - emb.shape[1]))
        emb: Tensor = self.cnn(emb.transpose(1,2)).transpose(1,2) # (B, T, D)
        emb = emb / torch.norm(emb, dim=-1, keepdim=True)
        attractors = self.dec(emb, max_nspks)
        attractors = attractors / torch.norm(attractors, dim=-1, keepdim=True)

        # Calculate emb consistency loss (cosine similarity)
        len_masks = [torch.ones(len, device=src[0].device) for len in ilens]
        len_masks = nn.utils.rnn.pad_sequence(len_masks, padding_value=0, batch_first=True)
        len_masks = len_masks.unsqueeze(dim=-1)
        len_square = [(len * len) for len in ilens]
        sum_len_square = sum(len_square)
        emb = emb[:, :seq_len, :] * len_masks
        # emb = emb[:, :T, :]
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
        # emb_consis_loss = F.mse_loss(attn_map, label_map, reduction='none')
        # emb_consis_loss = emb_consis_loss.masked_fill((label_map == 0) & (attn_map <= 0), 0.0)
        # emb_consis_loss = emb_consis_loss.sum() / sum_len_square
        emb_consis_loss = F.mse_loss(attn_map, label_map, reduction='sum') / sum_len_square

        # output: (B, T, C)
        emb = F.pad(emb, (0, 0, 0, pad_seqlen - emb.shape[1]))
        output = torch.matmul(emb.unsqueeze(dim=-2), attractors.transpose(-1, -2)).squeeze(dim=-2)
        output = [out[:ilen, :n_spk] for out, ilen, n_spk in zip(output, ilens, n_speakers)]

        emb = [e[:ilen] for e, ilen in zip(emb, ilens)]
        attractors = [attr[:ilen, 1:n_spk] for attr, ilen, n_spk in zip(attractors, ilens, n_speakers)]
        return output, emb_consis_loss, emb, attractors
    
    # def test(self, src, ilens, tgt, cos_oup_file, rec, max_nspks=6):
    def test(self, src, ilens, max_nspks=6):
        # emb: (B, T, E)
        emb = self.enc(src)
        B, T, D = emb.shape
        emb  = [e[:ilen] for e, ilen in zip(emb, ilens)]
        emb = nn.utils.rnn.pad_sequence(emb, padding_value=0, batch_first=True)
        seq_len = emb.shape[1]
        # pad_seqlen = max(math.ceil(seq_len / self.recurrent_chunk_size) * self.recurrent_chunk_size, self.max_seqlen)
        pad_seqlen = math.ceil(seq_len / self.recurrent_chunk_size) * self.recurrent_chunk_size
        emb = F.pad(emb, (0, 0, 0, pad_seqlen - emb.shape[1]))
        emb: Tensor = self.cnn(emb.transpose(1,2)).transpose(1,2) # (B, T, D)
        emb = emb / torch.norm(emb, dim=-1, keepdim=True)
        attractors = self.dec(emb, max_nspks)
        attractors = attractors / torch.norm(attractors, dim=-1, keepdim=True)
        
        # output: (B, T, C)
        emb = F.pad(emb, (0, 0, 0, pad_seqlen - emb.shape[1]))
        output = torch.matmul(emb.unsqueeze(dim=-2), attractors.transpose(-1, -2)).squeeze(dim=-2)
        output = [out[:ilen] for out, ilen in zip(output, ilens)]

        emb = [e[:ilen] for e, ilen in zip(emb, ilens)]
        attractors = [attr[:ilen] for attr, ilen in zip(attractors, ilens)]
        return output, emb, attractors



class StreamingConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=19):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0)  # No internal padding
        self.buffer = deque(maxlen=kernel_size)
        self.center = kernel_size // 2
        self.t = 0
    
    def forward(self, x_t):
        """
        Args:
            x_t: (batch_size, in_channels, 1)
        Returns:
            output: (batch_size, out_channels, 1)
        """
        self.t += 1
        B, C, _ = x_t.shape
        self.buffer.append(x_t)
        
        # if not enougn frames, return None or zero output
        if len(self.buffer) < self.kernel_size:
            left = self.kernel_size - len(self.buffer)
            padded_buffer = [torch.zeros_like(x_t)] * left + list(self.buffer)
        else:
            padded_buffer = list(self.buffer)
        
        # concat into shape (batch_size, in_channels, kernel_size)
        x_window = torch.cat(padded_buffer, dim=2) # (B, C, T)
        y_all = self.conv(x_window)  # (B, out_channels, T_out) = (B, C_out, 1)
        assert y_all.shape[-1] == 1, "Output should be a single time frame"
        
        if self.t >= self.center + 1:
            return y_all # (B, C_out, 1)
        else:
            return None
        

class MaskedTransformerDecoderModel(nn.Module):
    def __init__(self, in_size, n_heads, n_units, n_layers, recurrent_chunk_size, dim_feedforward, dropout=0.5, max_seqlen = 500, has_pos=False, mask_delay=0):
        super(MaskedTransformerDecoderModel, self).__init__()
        self.in_size = in_size
        self.n_heads = n_heads
        self.n_units = n_units
        self.n_layers = n_layers
        self.has_pos = has_pos
        self.max_seqlen = max_seqlen
        self.mask_delay = mask_delay

        self.encoder = nn.Linear(in_size, n_units)
        self.encoder_norm = nn.LayerNorm(n_units)

        self.pos_enc = PositionalEncoding(n_units, dropout)
        self.convert = nn.Linear(n_units * 2, n_units)
        self.layers = nn.ModuleList([
            TransformerEncoderFusionLayer(n_units, n_heads, recurrent_chunk_size, dim_feedforward, dropout, batch_first=True)
            for _ in range(n_layers)
        ])

    def _generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device), diagonal=-self.mask_delay) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, emb: Tensor, max_nspks: int, activation: Optional[Callable]=None):
        pos_enc = self.pos_enc(emb, max_nspks)
        attractors: Tensor = self.convert(torch.cat([emb.unsqueeze(dim=2).repeat(1, 1, max_nspks, 1), pos_enc], dim=-1))
        for layer in self.layers:
            attractors = layer(attractors)
        return attractors

    def forward_recurrent(self, emb: Tensor, max_nspks: int) -> Tensor:
        pos_enc = self.pos_enc(emb, max_nspks)
        attractors_init: Tensor = self.convert(torch.cat([emb.unsqueeze(dim=2).repeat(1, 1, max_nspks, 1), pos_enc], dim=-1))
        T = emb.shape[1]
        ret_states = [dict() for _ in self.layers]
        frames = []
        for t in range(T):
            x = attractors_init[:, [t], :, :]
            for i, layer in enumerate(self.layers):
                x = layer.forward_one_step(x, t, ret_states[i])
            frames.append(x)
        return torch.cat(frames, dim=1)

    def forward_one_step(self, emb_t: Tensor, t: int, max_nspks: int, ret_states: list) -> Tensor:
        # emb_t: (B, 1, D)
        # pos_enc.pe: (1, max_len, D), we need (B, 1, C, D)
        pe = self.pos_enc.pe[:, :max_nspks, :]  # (1, C, D)
        pos_enc_t = pe.unsqueeze(1).repeat(emb_t.shape[0], 1, 1, 1)  # (B, 1, C, D)
        attractor_t = self.convert(torch.cat([emb_t.unsqueeze(dim=2).repeat(1, 1, max_nspks, 1), pos_enc_t], dim=-1))  # (B, 1, C, D)
        for i, layer in enumerate(self.layers):
            attractor_t = layer.forward_one_step(attractor_t, t, ret_states[i])
        return attractor_t


class EmbeddingEncoderModule(nn.Module):
    def __init__(
            self,
            in_size,
            n_units,
            n_heads,
            n_layers,
            recurrent_chunk_size,
            feed_forward_expansion_factor: int = 8,
            conv_expansion_factor: int = 2,
            dropout: float = 0.1,
            conv_kernel_size: int = 16,
            half_step_residual: bool = True,
            max_seqlen: int = 500,
    ):
        super(EmbeddingEncoderModule, self).__init__()
        self.max_seqlen = max_seqlen
        self.recurrent_chunk_size = recurrent_chunk_size
        self.encoder = ConformerEncoder(
            input_dim=in_size,
            encoder_dim=n_units,
            num_layers=n_layers,
            num_attention_heads=n_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor=conv_expansion_factor,
            feed_forward_dropout_p=dropout,
            attention_dropout_p=dropout,
            conv_dropout_p=dropout,
            conv_kernel_size=conv_kernel_size,
            half_step_residual=half_step_residual,
            recurrent_chunk_size=recurrent_chunk_size,
        )
    
    def forward(self, src):
        src = nn.utils.rnn.pad_sequence(src, padding_value=0, batch_first=True)
        seq_len = src.shape[1]
        pad_seqlen = math.ceil(seq_len / self.recurrent_chunk_size) * self.recurrent_chunk_size
        src = F.pad(src, (0, 0, 0, pad_seqlen - src.shape[1]))
        output = self.encoder(src)
        return output

    def forward_recurrent(self, src):
        src = nn.utils.rnn.pad_sequence(src, padding_value=0, batch_first=True)
        return self.encoder.forward_recurrent(src)

    def forward_one_step(self, x_t: Tensor, t: int, ret_states: list, conv_caches: list) -> Tensor:
        # x_t: (B, 1, D_in)
        return self.encoder.forward_one_step(x_t, t, ret_states, conv_caches)

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






if __name__ == '__main__':
    print("=" * 60)
    print("Test 1: EmbeddingEncoderModule")
    print("=" * 60)

    enc = EmbeddingEncoderModule(
        in_size=80, n_units=64, n_heads=4, n_layers=2,
        recurrent_chunk_size=10, feed_forward_expansion_factor=4,
        conv_expansion_factor=2, dropout=0.0, conv_kernel_size=7,
        half_step_residual=True, max_seqlen=500,
    ).eval()

    torch.manual_seed(42)
    T, D_in = 30, 80
    src_tensor = torch.randn(T, D_in)
    src = [src_tensor]

    with torch.no_grad():
        # Batch forward
        out_batch = enc(src)
        print(f"Batch forward output shape: {out_batch.shape}")

        # Frame-by-frame forward_one_step
        B, D_out = 1, 64
        conv_kernel_size = 7
        ret_states = [dict() for _ in range(2)]
        conv_caches = [torch.zeros(B, D_out, conv_kernel_size - 1) for _ in range(2)]

        frames = []
        for t in range(T):
            x_t = src_tensor[t:t+1].unsqueeze(0)  # (1, 1, D_in)
            out_t = enc.forward_one_step(x_t, t, ret_states, conv_caches)
            frames.append(out_t)

        out_streaming = torch.cat(frames, dim=1)
        print(f"Streaming forward_one_step output shape: {out_streaming.shape}")

        match = torch.allclose(out_batch, out_streaming, atol=1e-5)
        max_diff = (out_batch - out_streaming).abs().max().item()
        print(f"forward == forward_one_step: {match}")
        print(f"Max diff: {max_diff:.2e}")

    print("\n" + "=" * 60)
    print("Test 2: MaskedTransformerDecoderModel")
    print("=" * 60)

    dec = MaskedTransformerDecoderModel(
        in_size=64, n_heads=4, n_units=64, n_layers=2,
        recurrent_chunk_size=10, dim_feedforward=128, dropout=0.0,
    ).eval()

    torch.manual_seed(42)
    T, D_in = 30, 64
    max_nspks = 3
    emb = torch.randn(1, T, D_in)

    with torch.no_grad():
        # Batch forward
        out_batch = dec.forward(emb, max_nspks)
        print(f"Batch forward output shape: {out_batch.shape}")

        # Frame-by-frame forward_one_step
        ret_states = [dict() for _ in range(2)]
        frames = []
        for t in range(T):
            emb_t = emb[:, [t], :]  # (1, 1, D_in)
            out_t = dec.forward_one_step(emb_t, t, max_nspks, ret_states)
            frames.append(out_t)

        out_streaming = torch.cat(frames, dim=1)
        print(f"Streaming forward_one_step output shape: {out_streaming.shape}")

        match = torch.allclose(out_batch, out_streaming, atol=1e-5)
        max_diff = (out_batch - out_streaming).abs().max().item()
        print(f"forward == forward_one_step: {match}")
        print(f"Max diff: {max_diff:.2e}")

    print("\n" + "=" * 60)
    print("Test 3: OnlineConformerRetentionDADiarization end-to-end")
    print("=" * 60)

    model = OnlineConformerRetentionDADiarization(
        n_speakers=3, in_size=80, n_units=64, n_heads=4,
        enc_n_layers=2, dec_n_layers=2, dropout=0.0,
        max_seqlen=500, recurrent_chunk_size=10,
        feed_forward_expansion_factor=4, dec_dim_feedforward=128,
        conv_expansion_factor=2, conv_kernel_size=7,
        half_step_residual=True, conv_delay=4, mask_delay=0,
    ).eval()

    torch.manual_seed(42)
    B, T, D_in = 1, 30, 80
    max_nspks = 3
    src_tensor = torch.randn(T, D_in)
    src = [src_tensor]
    ilens = [T]

    with torch.no_grad():
        # Batch inference via test()
        output_batch_all, emb_batch_all, attractor_batch_all = model.test(src, ilens, max_nspks=max_nspks)
        output_batch = output_batch_all[0]
        emb_batch = emb_batch_all[0]
        attractor_batch = attractor_batch_all[0]
        print(f"Batch test() output shape: {output_batch.shape}")

        # Reference: manually replicate streaming pipeline in batch mode
        # 1. Encoder (no padding needed, processes T frames)
        emb_ref = model.enc(src)  # (1, T, D)
        # 2. Causal conv1d equivalent to StreamingConv1d:
        #    left-pad with (kernel_size-1) zeros, no right pad → output T frames
        #    then append `center` zero frames on the right to flush → output T+center frames total
        #    but StreamingConv1d suppresses first `center` outputs → net output = T frames
        kernel_size = 2 * model.delay + 1
        skip = model.delay  # center = conv_delay
        # Pad encoder output: left=(kernel_size-1) zeros for causal, right=center zeros for flush
        emb_ref_padded = F.pad(emb_ref.transpose(1, 2), (kernel_size - 1, skip))  # (1, D, T+kernel_size-1+skip)
        emb_ref_conv_full = F.conv1d(emb_ref_padded, model.cnn.weight, model.cnn.bias)  # (1, D, T+skip)
        emb_ref_conv_full = emb_ref_conv_full.transpose(1, 2)  # (1, T+skip, D)
        # Drop first `skip` frames (StreamingConv1d returns None for those)
        emb_ref_conv = emb_ref_conv_full[:, skip:, :]  # (1, T, D)
        emb_ref_conv = emb_ref_conv / torch.norm(emb_ref_conv, dim=-1, keepdim=True)
        # 3. Decoder on all T frames
        attractor_ref = model.dec(emb_ref_conv, max_nspks)
        attractor_ref = attractor_ref / torch.norm(attractor_ref, dim=-1, keepdim=True)
        # 4. Output
        output_ref = torch.matmul(emb_ref_conv.unsqueeze(dim=-2), attractor_ref.transpose(-1, -2)).squeeze(dim=-2)
        output_ref = output_ref[0]  # (T, C)

        # Streaming inference: main loop + flush
        # Build StreamingConv1d externally, copy weights from model.cnn (simulates loading from ckpt)
        kernel_size_cnn = 2 * model.delay + 1
        streaming_cnn = StreamingConv1d(model.n_units, model.n_units, kernel_size=kernel_size_cnn)
        streaming_cnn.conv.weight.data.copy_(model.cnn.weight.data)
        streaming_cnn.conv.bias.data.copy_(model.cnn.bias.data)
        streaming_cnn.eval()

        # Init states externally
        n_enc_layers = len(model.enc.encoder.layers)
        n_dec_layers = len(model.dec.layers)
        conv_kernel_size_enc = model.enc.encoder._conv_kernel_size
        enc_states = {
            'ret_states': [dict() for _ in range(n_enc_layers)],
            'conv_caches': [torch.zeros(B, model.n_units, conv_kernel_size_enc - 1) for _ in range(n_enc_layers)]
        }
        dec_states = [dict() for _ in range(n_dec_layers)]

        outputs_streaming = []
        embs_streaming = []
        attractors_streaming = []
        dec_t = 0

        # Main loop: real input frames
        for t in range(T):
            x_t = src_tensor[t:t+1].unsqueeze(0)  # (1, 1, D_in)
            emb_t = model.enc.forward_one_step(x_t, t, enc_states['ret_states'], enc_states['conv_caches'])
            emb_t_conv = streaming_cnn(emb_t.transpose(1, 2))
            if emb_t_conv is not None:
                emb_t_conv = emb_t_conv.transpose(1, 2)
                emb_t_conv = emb_t_conv / torch.norm(emb_t_conv, dim=-1, keepdim=True)
                embs_streaming.append(emb_t_conv.squeeze(0))
                attractor_t = model.dec.forward_one_step(emb_t_conv, dec_t, max_nspks, dec_states)
                attractor_t = attractor_t / torch.norm(attractor_t, dim=-1, keepdim=True)
                attractors_streaming.append(attractor_t.squeeze(0))
                y_t = torch.matmul(emb_t_conv.unsqueeze(dim=-2), attractor_t.transpose(-1, -2)).squeeze(dim=-2)
                outputs_streaming.append(y_t.squeeze(0))
                dec_t += 1

        # Flush: send `center` zero frames to drain the last `center` real frames from conv
        for _ in range(model.delay):
            emb_zero = torch.zeros(B, 1, model.n_units)
            emb_t_conv = streaming_cnn(emb_zero.transpose(1, 2))
            if emb_t_conv is not None:
                emb_t_conv = emb_t_conv.transpose(1, 2)
                emb_t_conv = emb_t_conv / torch.norm(emb_t_conv, dim=-1, keepdim=True)
                embs_streaming.append(emb_t_conv.squeeze(0))
                attractor_t = model.dec.forward_one_step(emb_t_conv, dec_t, max_nspks, dec_states)
                attractor_t = attractor_t / torch.norm(attractor_t, dim=-1, keepdim=True)
                attractors_streaming.append(attractor_t.squeeze(0))
                y_t = torch.matmul(emb_t_conv.unsqueeze(dim=-2), attractor_t.transpose(-1, -2)).squeeze(dim=-2)
                outputs_streaming.append(y_t.squeeze(0))
                dec_t += 1

        output_streaming = torch.cat(outputs_streaming, dim=0)
        embs_streaming_cat = torch.cat(embs_streaming, dim=0)
        attractors_streaming_cat = torch.cat(attractors_streaming, dim=0)

        print(f"Reference (causal batch, with flush) output shape: {output_ref.shape}")
        print(f"Streaming (with flush) output shape: {output_streaming.shape}")

        output_ref_valid = output_ref
        emb_ref_valid = emb_ref_conv[0]
        attractor_ref_valid = attractor_ref[0]

        print(f"\n  Encoder+Conv output comparison (frames {skip}~{T}):")
        emb_match = torch.allclose(emb_ref_valid, embs_streaming_cat, atol=1e-4)
        emb_diff = (emb_ref_valid - embs_streaming_cat).abs().max().item()
        print(f"    Match: {emb_match}, Max diff: {emb_diff:.2e}")

        print(f"  Decoder output comparison:")
        attr_match = torch.allclose(attractor_ref_valid, attractors_streaming_cat, atol=1e-4)
        attr_diff = (attractor_ref_valid - attractors_streaming_cat).abs().max().item()
        print(f"    Match: {attr_match}, Max diff: {attr_diff:.2e}")

        print(f"  Final output comparison:")
        if output_streaming.shape[0] == output_ref_valid.shape[0]:
            match = torch.allclose(output_ref_valid, output_streaming, atol=1e-4)
            max_diff = (output_ref_valid - output_streaming).abs().max().item()
            print(f"    Causal-batch vs Streaming: {match}")
            print(f"    Max diff: {max_diff:.2e}")
        else:
            print(f"    Shape mismatch: ref={output_ref_valid.shape}, streaming={output_streaming.shape}")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
