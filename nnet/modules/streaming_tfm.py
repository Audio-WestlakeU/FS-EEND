# This is a frame-by-frame transformer encoder
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

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


class StreamingTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1, activation=F.relu):
        super().__init__()
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

    def forward(self, x):
        """
        Args:
            x: (batch_size, 1, d_model)
        Returns:
            output: (batch_size, 1, d_model)
        """
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

def _generate_square_subsequent_mask(sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device), diagonal=-0) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

def check_with_standard_tfm_encoder(standard_tfm, incremental_tfm):

    for std_layer, incr_layer in zip(standard_tfm.layers, incremental_tfm.layers):
        incr_layer.self_attn.attention.in_proj_weight.data.copy_(std_layer.self_attn.in_proj_weight.data)
        incr_layer.self_attn.attention.in_proj_bias.data.copy_(std_layer.self_attn.in_proj_bias.data)
        incr_layer.self_attn.attention.out_proj.weight.data.copy_(std_layer.self_attn.out_proj.weight.data)
        incr_layer.self_attn.attention.out_proj.bias.data.copy_(std_layer.self_attn.out_proj.bias.data)

        incr_layer.linear1.weight.data.copy_(std_layer.linear1.weight.data)
        incr_layer.linear1.bias.data.copy_(std_layer.linear1.bias.data)
        incr_layer.linear2.weight.data.copy_(std_layer.linear2.weight.data)
        incr_layer.linear2.bias.data.copy_(std_layer.linear2.bias.data)

        incr_layer.norm1.weight.data.copy_(std_layer.norm1.weight.data)
        incr_layer.norm1.bias.data.copy_(std_layer.norm1.bias.data)
        incr_layer.norm2.weight.data.copy_(std_layer.norm2.weight.data)
        incr_layer.norm2.bias.data.copy_(std_layer.norm2.bias.data)

if __name__ == '__main__':
    inp_seq = torch.rand(4, 10, 64)
    
    encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=2048, dropout=0.1, activation=F.relu, batch_first=True)
    standard_tfm = nn.TransformerEncoder(encoder_layer, num_layers=2)
    incremental_tfm = StreamingTransformerEncoder(d_model=64, nhead=4, num_layers=2)
    check_with_standard_tfm_encoder(standard_tfm, incremental_tfm)
    
    incremental_tfm.eval()
    standard_tfm.eval()
    incremental_output = []

    inp_mask = _generate_square_subsequent_mask(inp_seq.shape[1], device=inp_seq.device)
    standard_out = standard_tfm(inp_seq, inp_mask)

    for t in range(inp_seq.shape[1]):
        frame_output = incremental_tfm(inp_seq[:, t:t+1, :])  # 保证输出符合预期
        # print(f"Frame {t}: {frame_output.shape}")
        incremental_output.append(frame_output)
    incremental_output = torch.cat(incremental_output, dim=1)
    print(f"Incremental output: {incremental_output.shape}", f"Standard output: {standard_out.shape}")  # 打印输出的形状
    print(incremental_output)
    print(standard_out)
    print(torch.allclose(incremental_output, standard_out, atol=1e-6, rtol=1e-5))

