# This is a frame-by-frame streaming inference implementation
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from .merge_tfm_encoder import TransformerEncoderFusionLayer
import math

class IncrementalSelfAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
    
    def forward(self, query, key, value, past_key=None, past_value=None):
        """
        Args:
            query: (batch_size, 1, d_model)
            key: (batch_size, 1, d_model)
            value: (batch_size, 1, d_model)
            past_key: (batch_size, seq_len, d_model)
            past_value: (batch_size, seq_len, d_model)
        Returns:
            output: (batch_size, 1, d_model)
            new_key: (batch_size, seq_len+1, d_model)
            new_value: (batch_size, seq_len+1, d_model)
        """
        if past_key is not None and past_value is not None:
            comb_key = torch.cat([past_key, key], dim=1)
            comb_value = torch.cat([past_value, value], dim=1)
        else:
            comb_key = key
            comb_value = value
        # (batch_size, 1, d_model)
        attn_output, _ = self.attention(query, comb_key, comb_value)
        
        return attn_output, comb_key, comb_value

class StreamingTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=None):
        super().__init__()
        self.self_attn = IncrementalSelfAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)        
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation if activation else nn.ReLU()

    def forward(self, x, past_key=None, past_value=None):
        """
        Args:
            x: (batch_size, 1, d_model)
            past_key: (batch_size, seq_len, d_model)
            past_value: (batch_size, seq_len, d_model)
        Returns:
            output: (batch_size, 1, d_model)
            new_key: (batch_size, seq_len+1, d_model)
            new_value: (batch_size, seq_len+1, d_model)
        """
        query = x
        key, value = x, x
        attn_output, new_key, new_value = self._sa_block(query, key, value, past_key, past_value)
        attn_output = self.norm1(attn_output + x)
        
        ff_output = self._ff_block(attn_output)
        output = self.norm2(ff_output + attn_output)
        
        return output, new_key, new_value
    
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
    
    def _sa_block(self, query: Tensor, key, value, past_key=None, past_value=None) -> Tensor:
        x, new_key, new_value = self.self_attn(query, key, value, past_key, past_value)
        return self.dropout1(x), new_key, new_value


class StreamingEmbeddingEncoder(nn.Module):
    def __init__(self, in_size, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1, activation=F.relu):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_size)
        self.proj = nn.Linear(in_size, d_model)
        self.proj_norm = nn.LayerNorm(d_model)
        
        self.layers = nn.ModuleList([
            StreamingTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
            )
            for _ in range(num_layers)
        ])
        self.cache = [{} for _ in range(num_layers)] # cache key and value for each layer
        
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.1
        self.proj.bias.data.zero_()
        self.proj.weight.data.uniform_(-initrange, initrange)

    def forward(self, x: Tensor):
        """
        Args:
            x: (batch_size, 1, d_model)
        Returns:
            output: (batch_size, 1, d_model)
        """
        x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        x = self.proj_norm(self.proj(x))
        
        for i, layer in enumerate(self.layers):
            cache = self.cache[i]
            past_key = cache.get('key', None)
            past_value = cache.get('value', None)
            # print(f"Layer {i} - past_key: {past_key.shape if past_key is not None else None}")
            # print(f"Layer {i} - past_value: {past_value.shape if past_value is not None else None}")
            x, new_key, new_value = layer(x, past_key, past_value)

            self.cache[i]["key"] = new_key.detach()  # Detach to save memory
            self.cache[i]["value"] = new_value.detach()
        
        return x


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


class StreamingAttractorDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu):
        super().__init__()
        self.temp_attn = IncrementalSelfAttention(d_model, nhead)
        self.spk_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = activation if activation else nn.ReLU()
    
    def forward(self, x: Tensor, past_key=None, past_value=None):
        """
        x: (batch_size, 1, n_spks, d_model)
        past_key/value: for temporal attention, shape (B*n_spks, seq_len, D)
        Returns:
            output: (batch_size, 1, n_spks, d_model)
            new_key/value: updated tenporal key/value, shape (B*n_spks, seq_len+1, D)
        """
        B, T1, S, D = x.shape
        assert T1 == 1, "Input should be a single time frame"
        
        # ---------- 1. Temporal attention (along time axix) ----------
        x_temp = x.transpose(1, 2).reshape(B*S, 1, D)
        query = key = value = x_temp
        attn_out, new_key, new_value = self._temp_sa_block(query, key, value, past_key, past_value)
        temp_out = self.norm1(attn_out + x_temp)
        temp_out = temp_out.reshape(B, S, 1, D).transpose(1, 2)  # (B, 1, S, D)
        
        # ---------- 2. Speaker attention (along speaker axis) ----------
        query = key = value = temp_out.squeeze(1)  # (B, S, D)
        spk_out = self._spk_sa_block(query, key, value)
        spk_out = self.norm2(spk_out + query)
        spk_out = spk_out.unsqueeze(1)  # (B, 1, S, D)
        
        # ---------- 3. Feedforward ----------
        ff_out = self._ff_block(spk_out)
        output = self.norm3(ff_out + spk_out)
        
        return output, new_key, new_value
        
    def _temp_sa_block(self, query: Tensor, key, value, past_key=None, past_value=None) -> Tensor:
        x, new_key, new_value = self.temp_attn(query, key, value, past_key, past_value)
        return self.dropout1(x), new_key, new_value
    
    def _spk_sa_block(self, query: Tensor, key, value) -> Tensor:
        x, _ = self.spk_attn(query, key, value)
        return self.dropout2(x)
    
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class StreamingAttractorDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1, activation=F.relu):
        super().__init__()
        self.pos_enc = PositionalEncoding(d_model, dropout)
        self.convert = nn.Linear(2 * d_model, d_model)
        
        self.layers = nn.ModuleList([
            StreamingAttractorDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
            )
            for _ in range(num_layers)
        ])
        self.cache = [{} for _ in range(num_layers)] # cache key and value for each layer
    
    def forward(self, emb: Tensor, max_nspks: int):
        """
        Args:
            emb: (batch_size, 1, d_model)
        Returns:
            output: (batch_size, 1, n_spks, d_model)
        """
        pos_enc = self.pos_enc(emb, max_nspks) # (batch_size, 1, n_spks, d_model)
        x = self.convert(torch.cat([emb.unsqueeze(dim=2).repeat(1, 1, max_nspks, 1), pos_enc], dim=-1))
        
        for i, layer in enumerate(self.layers):
            cache = self.cache[i]
            past_key = cache.get('key', None)
            past_value = cache.get('value', None)
            # print(f"Layer {i} - past_key: {past_key.shape if past_key is not None else None}")
            # print(f"Layer {i} - past_value: {past_value.shape if past_value is not None else None}")
            x, new_key, new_value = layer(x, past_key, past_value)

            self.cache[i]["key"] = new_key.detach()  # Detach to save memory
            self.cache[i]["value"] = new_value.detach()
        
        return x

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
        pe = pe.unsqueeze(dim=0).repeat(x.shape[0], 1, 1, 1) # (B, 1, S, D)
        # x = x.unsqueeze(dim=2).repeat(1, 1, max_nspks, 1)
        # x = x + pe
        return pe
    
