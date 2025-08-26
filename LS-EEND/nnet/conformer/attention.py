# Copyright (c) 2021, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified by Di Liang @ Audio Lab of Westlake University 2024

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from torch.nn.modules.activation import MultiheadAttention

from ..modules.retention import RetNetRelPos, MultiScaleRetention
from .modules import Linear


class MergeMultiHeadedSelfRetentionModule(nn.Module):
    """
    Inputs: inputs, mask

        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked
        - **inputs** (batch, spk, time, dim): Tensor containing input vector
    Returns:
        - **outputs** (batch, time, dim): Tensor produces by relative multi headed self attention module.
    """
    def __init__(self, d_model: int, num_heads: int, recurrent_chunk_size: int = 500, dropout_p: float = 0.1, batch_first: bool = True):
        super(MergeMultiHeadedSelfRetentionModule, self).__init__()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        # attention: mha -> retention
        self.ret_pos = RetNetRelPos(d_model, num_heads, recurrent_chunk_size=recurrent_chunk_size)
        self.self_attn1 = MultiScaleRetention(d_model, num_heads, value_factor=1)
        self.self_attn2 = MultiheadAttention(d_model, num_heads, dropout=dropout_p, batch_first=batch_first)
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.dropout2 = nn.Dropout(p=dropout_p)

    
    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None):
        B, C, T, D = inputs.shape
        inputs = inputs.reshape(B*C, T, D)
        outputs: Tensor = inputs + self._sa_block1(self.layer_norm1(inputs))
        
        outputs = outputs.reshape(B, C, T, D).transpose(1, 2).reshape(B*T, C, D)
        outputs = outputs + self._sa_block2(self.layer_norm2(outputs), None, None)
        outputs = outputs.reshape(B, T, C, D).transpose(1, 2)
        return outputs
    
    def _sa_block1(self, x: Tensor) -> Tensor:
        rp = self.ret_pos(slen=x.shape[1], chunkwise_recurrent=True)
        x = self.self_attn1(x, rel_pos=rp, chunkwise_recurrent=True)
        return self.dropout1(x)
    
    def _sa_block2(self, x: Tensor, attn_mask: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        x = self.self_attn2(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout2(x)

class MultiHeadedSelfRetentionModule(nn.Module):
    """
    Conformer employ multi-headed self-attention (MHSA) while integrating an important technique from Transformer-XL,
    the relative sinusoidal positional encoding scheme. The relative positional encoding allows the self-attention
    module to generalize better on different input length and the resulting encoder is more robust to the variance of
    the utterance length. Conformer use prenorm residual units with dropout which helps training
    and regularizing deeper models.

    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout

    Inputs: inputs, mask
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked

    Returns:
        - **outputs** (batch, time, dim): Tensor produces by relative multi headed self attention module.
    """
    def __init__(self, d_model: int, num_heads: int, recurrent_chunk_size: int = 500, dropout_p: float = 0.1):
        super(MultiHeadedSelfRetentionModule, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        # attention: mha -> retention
        self.ret_pos = RetNetRelPos(d_model, num_heads, recurrent_chunk_size=recurrent_chunk_size)
        self.self_attn = MultiScaleRetention(d_model, num_heads, value_factor=1)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None):
        inputs = self.layer_norm(inputs)
        rp = self.ret_pos(slen=inputs.shape[1], chunkwise_recurrent=True)
        outputs = self.self_attn(inputs, rel_pos=rp, chunkwise_recurrent=True)

        # if inference, recurrent forward
        # state = dict()
        # outputs = []
        # for t in range(inputs.shape[1]):
        #     yr = self.self_attn(x=inputs[:, [t], :], rel_pos=self.ret_pos.forward(slen=t, activate_recurrent=True), incremental_state=state)
        #     outputs.append(yr)
        # outputs = torch.concat(outputs, dim=1)

        return self.dropout(outputs)
