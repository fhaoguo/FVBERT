# -*- coding: utf-8 -*-
# @Author     :fenghaoguo
# @Time       :2022/5/17 22:47
# @FileName   :transformer.py
# @Description:
import copy
import operator
from collections import OrderedDict, abc as container_abcs
from itertools import chain
from typing import Any, Dict, Iterable, Iterator, Optional, Union
from typing import Callable

import torch
from torch import Tensor
from torch._jit_internal import _copy_to_script_wrapper
from torch.nn import functional as F

from .attention import MultiheadAttention
from .dropout import Dropout
from .init import xavier_uniform_
from .layer_norm import LayerNorm
from .linear import Linear
from .module import Module


class TransformerBlock(Module):
    """
    Bidirectional Encoder = Left part of Transformer
    MultiHead_Attention + Feed_Forward with sublayer connection
    """
    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 12,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 custom_encoder: Optional[Any] = None,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerBlock, self).__init__()

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                    activation, layer_norm_eps, batch_first, norm_first,
                                                    **factory_kwargs)
            encoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.batch_first = batch_first

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        is_batched = src.dim() == 3
        if not self.batch_first and src.size(1) != tgt.size(1) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
        elif self.batch_first and src.size(0) != tgt.size(0) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")

        if src.size(-1) != self.d_model or tgt.size(-1) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        return self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer m2."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


# class Transformer(Module):
#     r"""A transformer m2. User is able to modify the attributes as needed. The architecture
#     is based on the paper "Attention Is All You Need". Ashish Vaswani, Noam Shazeer,
#     Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and
#     Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information
#     Processing Systems, pages 6000-6010. Users can build the BERT(https://arxiv.org/abs/1810.04805)
#     m2 with corresponding parameters.
#     Args:
#         d_model: the number of expected features in the encoder/decoder inputs (default=512).
#         nhead: the number of heads in the multiheadattention models (default=8).
#         num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
#         num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
#         dim_feedforward: the dimension of the feedforward network m2 (default=2048).
#         dropout: the dropout value (default=0.1).
#         activation: the activation function of encoder/decoder intermediate layer, can be a string
#             ("relu" or "gelu") or a unary callable. Default: relu
#         custom_encoder: custom encoder (default=None).
#         custom_decoder: custom decoder (default=None).
#         layer_norm_eps: the eps value in layer normalization components (default=1e-5).
#         batch_first: If ``True``, then the input and output tensors are provided
#             as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
#         norm_first: if ``True``, encoder and decoder layers will perform LayerNorms before
#             other attention and feedforward operations, otherwise after. Default: ``False`` (after).
#     Examples::
#         >>> transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
#         >>> src = torch.rand((10, 32, 512))
#         >>> tgt = torch.rand((20, 32, 512))
#         >>> out = transformer_model(src, tgt)
#     Note: A full example to apply nn.Transformer module for the word language m2 is available in
#     https://github.com/pytorch/examples/tree/master/word_language_model
#     """
#
#     def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
#                  num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
#                  activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
#                  custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None,
#                  layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
#                  device=None, dtype=None) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super(Transformer, self).__init__()
#
#         if custom_encoder is not None:
#             self.encoder = custom_encoder
#         else:
#             encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
#                                                     activation, layer_norm_eps, batch_first, norm_first,
#                                                     **factory_kwargs)
#             encoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
#             self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
#
#         if custom_decoder is not None:
#             self.decoder = custom_decoder
#         else:
#             decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
#                                                     activation, layer_norm_eps, batch_first, norm_first,
#                                                     **factory_kwargs)
#             decoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
#             self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
#
#         self._reset_parameters()
#
#         self.d_model = d_model
#         self.nhead = nhead
#
#         self.batch_first = batch_first
#
#     def forward(self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
#                 memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
#                 tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
#         r"""Take in and process masked source/target sequences.
#         Args:
#             src: the sequence to the encoder (required).
#             tgt: the sequence to the decoder (required).
#             src_mask: the additive mask for the src sequence (optional).
#             tgt_mask: the additive mask for the tgt sequence (optional).
#             memory_mask: the additive mask for the encoder output (optional).
#             src_key_padding_mask: the ByteTensor mask for src keys per batch (optional).
#             tgt_key_padding_mask: the ByteTensor mask for tgt keys per batch (optional).
#             memory_key_padding_mask: the ByteTensor mask for memory keys per batch (optional).
#         Shape:
#             - src: :math:`(S, E)` for unbatched input, :math:`(S, N, E)` if `batch_first=False` or
#               `(N, S, E)` if `batch_first=True`.
#             - tgt: :math:`(T, E)` for unbatched input, :math:`(T, N, E)` if `batch_first=False` or
#               `(N, T, E)` if `batch_first=True`.
#             - src_mask: :math:`(S, S)` or :math:`(N\cdot\text{num\_heads}, S, S)`.
#             - tgt_mask: :math:`(T, T)` or :math:`(N\cdot\text{num\_heads}, T, T)`.
#             - memory_mask: :math:`(T, S)`.
#             - src_key_padding_mask: :math:`(S)` for unbatched input otherwise :math:`(N, S)`.
#             - tgt_key_padding_mask: :math:`(T)` for unbatched input otherwise :math:`(N, T)`.
#             - memory_key_padding_mask: :math:`(S)` for unbatched input otherwise :math:`(N, S)`.
#             Note: [src/tgt/memory]_mask ensures that position i is allowed to attend the unmasked
#             positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
#             while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
#             are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
#             is provided, it will be added to the attention weight.
#             [src/tgt/memory]_key_padding_mask provides specified elements in the key to be ignored by
#             the attention. If a ByteTensor is provided, the non-zero positions will be ignored while the zero
#             positions will be unchanged. If a BoolTensor is provided, the positions with the
#             value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
#             - output: :math:`(T, E)` for unbatched input, :math:`(T, N, E)` if `batch_first=False` or
#               `(N, T, E)` if `batch_first=True`.
#             Note: Due to the multi-head attention architecture in the transformer m2,
#             the output sequence length of a transformer is same as the input sequence
#             (i.e. target) length of the decode.
#             where S is the source sequence length, T is the target sequence length, N is the
#             batch size, E is the feature number
#         Examples:
#             >>> output = transformer_model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
#         """
#
#         is_batched = src.dim() == 3
#         if not self.batch_first and src.size(1) != tgt.size(1) and is_batched:
#             raise RuntimeError("the batch number of src and tgt must be equal")
#         elif self.batch_first and src.size(0) != tgt.size(0) and is_batched:
#             raise RuntimeError("the batch number of src and tgt must be equal")
#
#         if src.size(-1) != self.d_model or tgt.size(-1) != self.d_model:
#             raise RuntimeError("the feature number of src and tgt must be equal to d_model")
#
#         memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
#
#         output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
#                               tgt_key_padding_mask=tgt_key_padding_mask,
#                               memory_key_padding_mask=memory_key_padding_mask)
#         return output
#
#     @staticmethod
#     def generate_square_subsequent_mask(sz: int) -> Tensor:
#         r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
#             Unmasked positions are filled with float(0.0).
#         """
#         return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
#
#     def _reset_parameters(self):
#         r"""Initiate parameters in the transformer m2."""
#
#         for p in self.parameters():
#             if p.dim() > 1:
#                 xavier_uniform_(p)


class TransformerEncoder(Module):
    r"""TransformerEncoder is a stack of N encoder layers
    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.
        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


# class TransformerDecoder(Module):
#     r"""TransformerDecoder is a stack of N decoder layers
#     Args:
#         decoder_layer: an instance of the TransformerDecoderLayer() class (required).
#         num_layers: the number of sub-decoder-layers in the decoder (required).
#         norm: the layer normalization component (optional).
#     Examples::
#         >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
#         >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
#         >>> memory = torch.rand(10, 32, 512)
#         >>> tgt = torch.rand(20, 32, 512)
#         >>> out = transformer_decoder(tgt, memory)
#     """
#     __constants__ = ['norm']
#
#     def __init__(self, decoder_layer, num_layers, norm=None):
#         super(TransformerDecoder, self).__init__()
#         self.layers = _get_clones(decoder_layer, num_layers)
#         self.num_layers = num_layers
#         self.norm = norm
#
#     def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
#                 memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
#                 memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
#         r"""Pass the inputs (and mask) through the decoder layer in turn.
#         Args:
#             tgt: the sequence to the decoder (required).
#             memory: the sequence from the last layer of the encoder (required).
#             tgt_mask: the mask for the tgt sequence (optional).
#             memory_mask: the mask for the memory sequence (optional).
#             tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
#             memory_key_padding_mask: the mask for the memory keys per batch (optional).
#         Shape:
#             see the docs in Transformer class.
#         """
#         output = tgt
#
#         for mod in self.layers:
#             output = mod(output, memory, tgt_mask=tgt_mask,
#                          memory_mask=memory_mask,
#                          tgt_key_padding_mask=tgt_key_padding_mask,
#                          memory_key_padding_mask=memory_key_padding_mask)
#
#         if self.norm is not None:
#             output = self.norm(output)
#
#         return output


class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network m2 (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).
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
        - norm_first is ``False`` (this restriction may be loosened in the future)
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

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward m2
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu:
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu:
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

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

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        if (not self.norm_first and not self.training and
            self.self_attn.batch_first and src.dim() == 3 and self.self_attn._qkv_same_embed_dim and
            self.activation_relu_or_gelu and self.norm1.eps == self.norm2.eps and
            ((src_mask is None and src_key_padding_mask is None)
             if src.is_nested
             else (src_mask is None or src_key_padding_mask is None))):
            tensor_args = (
                src,
                self.self_attn.in_proj_weight,
                self.self_attn.in_proj_bias,
                self.self_attn.out_proj.weight,
                self.self_attn.out_proj.bias,
                self.norm1.weight,
                self.norm1.bias,
                self.norm2.weight,
                self.norm2.bias,
                self.linear1.weight,
                self.linear1.bias,
                self.linear2.weight,
                self.linear2.bias,
            )
            if (not torch.overrides.has_torch_function(tensor_args) and
                    # We have to use a list comprehension here because TorchScript
                    # doesn't support generator expressions.
                    all([(x.is_cuda or 'cpu' in str(x.device)) for x in tensor_args]) and
                    (not torch.is_grad_enabled() or all([not x.requires_grad for x in tensor_args]))):
                return torch._transformer_encoder_layer_fwd(
                    src,
                    self.self_attn.embed_dim,
                    self.self_attn.num_heads,
                    self.self_attn.in_proj_weight,
                    self.self_attn.in_proj_bias,
                    self.self_attn.out_proj.weight,
                    self.self_attn.out_proj.bias,
                    self.activation_relu_or_gelu == 2,
                    False,  # norm_first, currently not supported
                    self.norm1.eps,
                    self.norm1.weight,
                    self.norm1.bias,
                    self.norm2.weight,
                    self.norm2.bias,
                    self.linear1.weight,
                    self.linear1.bias,
                    self.linear2.weight,
                    self.linear2.bias,
                    src_mask if src_mask is not None else src_key_padding_mask,
                )
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


# class TransformerDecoderLayer(Module):
#     r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
#     This standard decoder layer is based on the paper "Attention Is All You Need".
#     Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
#     Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
#     Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
#     in a different way during application.
#     Args:
#         d_model: the number of expected features in the input (required).
#         nhead: the number of heads in the multiheadattention models (required).
#         dim_feedforward: the dimension of the feedforward network m2 (default=2048).
#         dropout: the dropout value (default=0.1).
#         activation: the activation function of the intermediate layer, can be a string
#             ("relu" or "gelu") or a unary callable. Default: relu
#         layer_norm_eps: the eps value in layer normalization components (default=1e-5).
#         batch_first: If ``True``, then the input and output tensors are provided
#             as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
#         norm_first: if ``True``, layer norm is done prior to self attention, multihead
#             attention and feedforward operations, respectivaly. Otherwise it's done after.
#             Default: ``False`` (after).
#     Examples::
#         >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
#         >>> memory = torch.rand(10, 32, 512)
#         >>> tgt = torch.rand(20, 32, 512)
#         >>> out = decoder_layer(tgt, memory)
#     Alternatively, when ``batch_first`` is ``True``:
#         >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
#         >>> memory = torch.rand(32, 10, 512)
#         >>> tgt = torch.rand(32, 20, 512)
#         >>> out = decoder_layer(tgt, memory)
#     """
#     __constants__ = ['batch_first', 'norm_first']
#
#     def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
#                  activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
#                  layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
#                  device=None, dtype=None) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super(TransformerDecoderLayer, self).__init__()
#         self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
#                                             **factory_kwargs)
#         self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
#                                                  **factory_kwargs)
#         # Implementation of Feedforward m2
#         self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
#         self.dropout = Dropout(dropout)
#         self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)
#
#         self.norm_first = norm_first
#         self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
#         self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
#         self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
#         self.dropout1 = Dropout(dropout)
#         self.dropout2 = Dropout(dropout)
#         self.dropout3 = Dropout(dropout)
#
#         # Legacy string support for activation function.
#         if isinstance(activation, str):
#             self.activation = _get_activation_fn(activation)
#         else:
#             self.activation = activation
#
#     def __setstate__(self, state):
#         if 'activation' not in state:
#             state['activation'] = F.relu
#         super(TransformerDecoderLayer, self).__setstate__(state)
#
#     def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
#                 tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
#         r"""Pass the inputs (and mask) through the decoder layer.
#         Args:
#             tgt: the sequence to the decoder layer (required).
#             memory: the sequence from the last layer of the encoder (required).
#             tgt_mask: the mask for the tgt sequence (optional).
#             memory_mask: the mask for the memory sequence (optional).
#             tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
#             memory_key_padding_mask: the mask for the memory keys per batch (optional).
#         Shape:
#             see the docs in Transformer class.
#         """
#         # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
#
#         x = tgt
#         if self.norm_first:
#             x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
#             x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
#             x = x + self._ff_block(self.norm3(x))
#         else:
#             x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
#             x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
#             x = self.norm3(x + self._ff_block(x))
#
#         return x
#
#     # self-attention block
#     def _sa_block(self, x: Tensor,
#                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
#         x = self.self_attn(x, x, x,
#                            attn_mask=attn_mask,
#                            key_padding_mask=key_padding_mask,
#                            need_weights=False)[0]
#         return self.dropout1(x)
#
#     # multihead attention block
#     def _mha_block(self, x: Tensor, mem: Tensor,
#                    attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
#         x = self.multihead_attn(x, mem, mem,
#                                 attn_mask=attn_mask,
#                                 key_padding_mask=key_padding_mask,
#                                 need_weights=False)[0]
#         return self.dropout2(x)
#
#     # feed forward block
#     def _ff_block(self, x: Tensor) -> Tensor:
#         x = self.linear2(self.dropout(self.activation(self.linear1(x))))
#         return self.dropout3(x)


class ModuleList(Module):
    r"""Holds submodules in a list.
    :class:`~torch.nn.ModuleList` can be indexed like a regular Python list, but
    modules it contains are properly registered, and will be visible by all
    :class:`~torch.nn.Module` methods.
    Args:
        modules (iterable, optional): an iterable of modules to add
    Example::
        class MyModule(nn.Module):
            def __init__.py(self):
                super(MyModule, self).__init__.py()
                self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])
            def forward(self, x):
                # ModuleList can act as an iterable, or be indexed using ints
                for i, l in enumerate(self.linears):
                    x = self.linears[i // 2](x) + l(x)
                return x
    """

    _modules: Dict[str, Module]  # type: ignore[assignment]

    def __init__(self, modules: Optional[Iterable[Module]] = None) -> None:
        super(ModuleList, self).__init__()
        if modules is not None:
            self += modules

    def _get_abs_string_index(self, idx):
        """Get the absolute index for the list of modules"""
        idx = operator.index(idx)
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        return str(idx)

    @_copy_to_script_wrapper
    def __getitem__(self, idx: int) -> Union[Module, 'ModuleList']:
        if isinstance(idx, slice):
            return self.__class__(list(self._modules.values())[idx])
        else:
            return self._modules[self._get_abs_string_index(idx)]

    def __setitem__(self, idx: int, module: Module) -> None:
        idx = self._get_abs_string_index(idx)
        return setattr(self, str(idx), module)

    def __delitem__(self, idx: Union[int, slice]) -> None:
        if isinstance(idx, slice):
            for k in range(len(self._modules))[idx]:
                delattr(self, str(k))
        else:
            delattr(self, self._get_abs_string_index(idx))
        # To preserve numbering, self._modules is being reconstructed with modules after deletion
        str_indices = [str(i) for i in range(len(self._modules))]
        self._modules = OrderedDict(list(zip(str_indices, self._modules.values())))

    @_copy_to_script_wrapper
    def __len__(self) -> int:
        return len(self._modules)

    @_copy_to_script_wrapper
    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def __iadd__(self, modules: Iterable[Module]) -> 'ModuleList':
        return self.extend(modules)

    def __add__(self, other: Iterable[Module]) -> 'ModuleList':
        combined = ModuleList()
        for i, module in enumerate(chain(self, other)):
            combined.add_module(str(i), module)
        return combined

    @_copy_to_script_wrapper
    def __dir__(self):
        keys = super(ModuleList, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def insert(self, index: int, module: Module) -> None:
        r"""Insert a given module before a given index in the list.
        Args:
            index (int): index to insert.
            module (nn.Module): module to insert
        """
        for i in range(len(self._modules), index, -1):
            self._modules[str(i)] = self._modules[str(i - 1)]
        self._modules[str(index)] = module

    def append(self, module: Module) -> 'ModuleList':
        r"""Appends a given module to the end of the list.
        Args:
            module (nn.Module): module to append
        """
        self.add_module(str(len(self)), module)
        return self

    def extend(self, modules: Iterable[Module]) -> 'ModuleList':
        r"""Appends modules from a Python iterable to the end of the list.
        Args:
            modules (iterable): iterable of modules to append
        """
        if not isinstance(modules, container_abcs.Iterable):
            raise TypeError("ModuleList.extend should be called with an "
                            "iterable, but got " + type(modules).__name__)
        offset = len(self)
        for i, module in enumerate(modules):
            self.add_module(str(offset + i), module)
        return self


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


if __name__ == '__main__':
    run_code = 0
