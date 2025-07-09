import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..modules.streaming_tfm import StreamingEmbeddingEncoder, StreamingTransformerEncoderLayer, StreamingConv1d, StreamingAttractorDecoder, StreamingAttractorDecoderLayer
from torch import Tensor
from typing import Optional, Any, Union, Callable

class StreamingTransformerEDADiarization(nn.Module):
    def __init__(self, in_size, n_units, n_heads, enc_n_layers, dec_n_layers, dropout, has_mask, max_seqlen, dec_dim_feedforward, conv_delay=9, mask_delay=0, decom_kernel_size=64):
        """ Self-attention-based diarization model.

        Args:
          in_size (int): Dimension of input feature vector
          n_units (int): Number of units in a self-attention block
          n_heads (int): Number of attention heads
          n_layers (int): Number of transformer-encoder layers
          dropout (float): dropout ratio
        """
        super(StreamingTransformerEDADiarization, self).__init__()
        self.delay = conv_delay
        self.n_units = n_units
        self.enc = StreamingEmbeddingEncoder(
            in_size, n_units, n_heads, enc_n_layers, dim_feedforward=dec_dim_feedforward, dropout=dropout
            )
        self.cnn = StreamingConv1d(n_units, n_units, kernel_size=2 * conv_delay + 1)
        self.dec = StreamingAttractorDecoder(
            n_units, n_heads, dec_n_layers, dim_feedforward=dec_dim_feedforward, dropout=dropout
        )
    
    def test(self, x_t: Tensor, max_nspks: int=6, dummy_conv_input=False):
        """
        Args:
          x_t (Tensor): input tensor (B, 1, D), input feature vector at time t
          max_nspks (int): maximum number of speakers
          dummy_conv_input (bool): whether to use dummy input for conv1d
        Returns:
          y_t (Tensor): output tensor (B, 1, S), predicted speaker at time t
        """
        # --------- 1. Embedding encoder ---------
        if dummy_conv_input:
            emb_t = torch.zeros(1, 1, self.n_units, device=x_t.device) # (B, 1, D)
        else:
            emb_t = self.enc(x_t) # (B, 1, D)
        
        # --------- 2. Conv1d ---------
        emb_t = self.cnn(emb_t.transpose(1, 2)) # (B, D, 1)
        if emb_t is None:
            return None
        emb_t = emb_t.transpose(1, 2) # (B, 1, D)
        emb_t = emb_t / torch.norm(emb_t, dim=-1, keepdim=True)
        
        # --------- 3. Attractor decoder ---------
        attractor_t = self.dec(emb_t, max_nspks)
        attractor_t = attractor_t / torch.norm(attractor_t, dim=-1, keepdim=True)
        
        # --------- 4. Output ---------
        y_t = torch.matmul(emb_t.unsqueeze(dim=-2), attractor_t.transpose(-1, -2)).squeeze(dim=-2) # (B, 1, S)
        
        return y_t

