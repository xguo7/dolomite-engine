import os
from typing import Tuple

import torch
import torch.nn.functional as F

from ....utils import is_flash_attention_available


if is_flash_attention_available():
    from flash_attn.flash_attn_interface import flash_attn_varlen_func


_USE_PYTORCH_NATIVE_FLASH_KERNEL: bool = bool(os.getenv("PYTORCH_NATIVE_FLASH_KERNEL", False))


class _FlashAttentionVarlenTorch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: torch.Tensor,
        max_seqlen_k: torch.Tensor,
        dropout_p: float,
        softmax_scale: float,
        causal: bool,
    ) -> torch.Tensor:
        attention_output, log_sum_exp, philox_seed, philox_offset, _ = torch.ops.aten._flash_attention_forward(
            query=query,
            key=key,
            value=value,
            cum_seq_q=cu_seqlens_q,
            cum_seq_k=cu_seqlens_k,
            max_q=max_seqlen_q,
            max_k=max_seqlen_k,
            dropout_p=dropout_p,
            is_causal=causal,
            return_debug_mask=False,
            scale=softmax_scale,
        )

        ctx.save_for_backward(
            query,
            key,
            value,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            attention_output,
            log_sum_exp,
            philox_seed,
            philox_offset,
        )

        ctx.dropout_p = dropout_p
        ctx.causal = causal
        ctx.softmax_scale = softmax_scale

        return attention_output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            query,
            key,
            value,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            causal,
            softmax_scale,
            attention_output,
            log_sum_exp,
            philox_seed,
            philox_offset,
        ) = ctx.saved_tensors

        query_grad, key_grad, value_grad = torch.ops.aten._flash_attention_backward(
            grad_output,
            query,
            key,
            value,
            attention_output,
            log_sum_exp,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            causal,
            philox_seed,
            philox_offset,
            softmax_scale,
        )

        return query_grad, key_grad, value_grad


def flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: torch.Tensor,
    max_seqlen_k: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    if _USE_PYTORCH_NATIVE_FLASH_KERNEL:
        attention_output = _FlashAttentionVarlenTorch.apply(
            query, key, value, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, causal
        )
    else:
        attention_output = flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
        )

    return attention_output


def unpad_input(hidden_states, attention_mask):
    """
    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
    Return:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz), the indices of non-masked tokens from the flattened input sequence.
        cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        max_seqlen_in_batch: int
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    # TD [2022-03-04] We don't want to index with a bool mask, because Pytorch will expand the
    # bool mask, then call nonzero to get the indices, then index with those. The indices is @dim
    # times larger than it needs to be, wasting memory. It's faster and more memory-efficient to
    # index with integer indices. Moreover, torch's index is a bit slower than it needs to be,
    # so we write custom forward and backward to make it a bit faster.
    return (
        index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def get_unpad_data(attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )
