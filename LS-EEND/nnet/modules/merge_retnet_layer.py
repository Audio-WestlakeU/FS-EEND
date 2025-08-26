import copy
from typing import Optional, Any, Union, Callable

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.modules.activation import MultiheadAttention
from .retention import RetNetRelPos, MultiScaleRetention
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm

class TransformerEncoderFusionLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectively. Otherwise it's done after. Default: ``False`` (after).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)

    Fast path:
        forward() will use a special optimized implementation if all of the following
        conditions are met:

        - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor
          argument ``requires_grad``
        - training is disabled (using ``.eval()``)
        - batch_first is ``True`` and the input is batched (i.e., ``src.dim() == 3``)
        - activation is one of: ``"relu"``, ``"gelu"``, ``torch.functional.relu``, or ``torch.functional.gelu``
        - at most one of ``src_mask`` and ``src_key_padding_mask`` is passed
        - if src is a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_, neither ``src_mask``
          nor ``src_key_padding_mask`` is passed
        - the two ``LayerNorm`` instances have a consistent ``eps`` value (this will naturally be the case
          unless the caller has manually modified one without modifying the other)

        If the optimized implementation is in use, a
        `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be
        passed for ``src`` to represent padding more efficiently than using a padding
        mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ will be
        returned, and an additional speedup proportional to the fraction of the input that
        is padding can be expected.
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, recurrent_chunk_size: int = 500, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderFusionLayer, self).__init__()
        # self.self_attn1 = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
        #                                     **factory_kwargs)
        self.ret_pos1 = RetNetRelPos(d_model, nhead, recurrent_chunk_size=recurrent_chunk_size)
        self.self_attn1 = MultiScaleRetention(d_model, nhead, value_factor=1)
        
        self.self_attn2 = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm11 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm12 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm21 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm22 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout11 = Dropout(dropout)
        self.dropout21 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super(TransformerEncoderFusionLayer, self).__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu


    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        if src_key_padding_mask is not None:
            _skpm_dtype = src_key_padding_mask.dtype
            if _skpm_dtype != torch.bool and not torch.is_floating_point(src_key_padding_mask):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported")
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        why_not_sparsity_fast_path = ''
        if not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif self.training:
            why_not_sparsity_fast_path = "training is enabled"
        elif not self.self_attn1.batch_first :
            why_not_sparsity_fast_path = "self_attn.batch_first was not True"
        elif not self.self_attn1._qkv_same_embed_dim :
            why_not_sparsity_fast_path = "self_attn._qkv_same_embed_dim was not True"
        elif not self.activation_relu_or_gelu:
            why_not_sparsity_fast_path = "activation_relu_or_gelu was not True"
        elif not (self.norm11.eps == self.norm12.eps):
            why_not_sparsity_fast_path = "norm1.eps is not equal to norm2.eps"
        elif src_mask is not None:
            why_not_sparsity_fast_path = "src_mask is not supported for fastpath"
        elif src.is_nested and src_key_padding_mask is not None:
            why_not_sparsity_fast_path = "src_key_padding_mask is not supported with NestedTensor input for fastpath"
        elif self.self_attn1.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"

        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                self.self_attn1.in_proj_weight,
                self.self_attn1.in_proj_bias,
                self.self_attn1.out_proj.weight,
                self.self_attn1.out_proj.bias,
                self.self_attn2.in_proj_weight,
                self.self_attn2.in_proj_bias,
                self.self_attn2.out_proj.weight,
                self.self_attn2.out_proj.bias,
                self.norm11.weight,
                self.norm11.bias,
                self.norm12.weight,
                self.norm12.bias,
                self.norm21.weight,
                self.norm21.bias,
                self.norm22.weight,
                self.norm22.bias,
                self.linear1.weight,
                self.linear1.bias,
                self.linear2.weight,
                self.linear2.bias,
            )

            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif not all((x.is_cuda or 'cpu' in str(x.device)) for x in tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument is neither CUDA nor CPU"
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                              "input/output projection weights or biases requires_grad")

            if not why_not_sparsity_fast_path:
                return torch._transformer_encoder_layer_fwd(
                    src,
                    self.self_attn1.embed_dim,
                    self.self_attn1.num_heads,
                    self.self_attn1.in_proj_weight,
                    self.self_attn1.in_proj_bias,
                    self.self_attn1.out_proj.weight,
                    self.self_attn1.out_proj.bias,
                    self.self_attn2.embed_dim,
                    self.self_attn2.num_heads,
                    self.self_attn2.in_proj_weight,
                    self.self_attn2.in_proj_bias,
                    self.self_attn2.out_proj.weight,
                    self.self_attn2.out_proj.bias,
                    self.activation_relu_or_gelu == 2,
                    self.norm_first,
                    self.norm11.eps,
                    self.norm11.weight,
                    self.norm11.bias,
                    self.norm12.weight,
                    self.norm12.bias,
                    self.norm21.eps,
                    self.norm21.weight,
                    self.norm21.bias,
                    self.norm22.weight,
                    self.norm22.bias,
                    self.linear1.weight,
                    self.linear1.bias,
                    self.linear2.weight,
                    self.linear2.bias,
                    # TODO: if src_mask and src_key_padding_mask merge to single 4-dim mask
                    src_mask if src_mask is not None else src_key_padding_mask,
                    1 if src_key_padding_mask is not None else
                    0 if src_mask is not None else
                    None,
                )


        # Self-attn on time-frame dim
        B, T, C, D = src.shape
        x = src.transpose(1, 2).reshape(B*C, T, D)
        if self.norm_first:
            x = x + self._sa_block1(self.norm11(x), src_mask, src_key_padding_mask)
            # x = x + self._ff_block(self.norm12(x))
        else:
            x = self.norm11(x + self._sa_block1(x, src_mask, src_key_padding_mask))
            # x = self.norm12(x + self._ff_block(x))
        x = x.reshape(B, C, T, D).transpose(1, 2).reshape(B*T, C, D)
        
        # Self-attention on spk dim
        if self.norm_first:
            x = x + self._sa_block2(self.norm21(x), None, None)
            x = x + self._ff_block(self.norm22(x))
        else:
            x = self.norm21(x + self._sa_block2(x, None, None))
            x = self.norm22(x + self._ff_block(x))
        x = x.reshape(B, T, C, D)

        return x

    # self-attention block
    def _sa_block1(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        # x = self.self_attn1(x, x, x,
        #                    attn_mask=attn_mask,
        #                    key_padding_mask=key_padding_mask,
        #                    need_weights=False)[0]
        
        rp = self.ret_pos1(slen=x.shape[1], chunkwise_recurrent=True)
        x = self.self_attn1(x, rel_pos=rp, chunkwise_recurrent=True)

        # if inference, recurrent forward
        # state = dict()
        # outputs = []
        # for t in range(x.shape[1]):
        #     yr = self.self_attn1(x=x[:, [t], :], rel_pos=self.ret_pos1.forward(slen=t, activate_recurrent=True), incremental_state=state)
        #     outputs.append(yr)
        # outputs = torch.concat(outputs, dim=1)
        # return self.dropout11(outputs)
        
        return self.dropout11(x)
    
    # self-attention2 block
    def _sa_block2(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn2(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout21(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))