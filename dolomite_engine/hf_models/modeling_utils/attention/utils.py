from typing import Tuple

import torch
import torch.nn.functional as F

from ....utils import is_flash_attention_available


if is_flash_attention_available():
    from flash_attn.flash_attn_interface import flash_attn_varlen_func


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
            dropout_p,
            causal,
            softmax_scale,
            attention_output,
            log_sum_exp,
            philox_seed,
            philox_offset,
        )

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
    # if is_flash_attention_available():
    #     attention_output = flash_attn_varlen_func(
    #         query,
    #         key,
    #         value,
    #         cu_seqlens_q=cu_seqlens_q,
    #         cu_seqlens_k=cu_seqlens_k,
    #         max_seqlen_q=max_seqlen_q,
    #         max_seqlen_k=max_seqlen_k,
    #         dropout_p=dropout_p,
    #         softmax_scale=softmax_scale,
    #         causal=causal,
    #     )
    # else:
    attention_output = _FlashAttentionVarlenTorch.apply(
        query, key, value, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, causal
    )

    return attention_output


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
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


def interleave_query_key_value_tensor_for_mha(
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    interleaved = []
    for i in range(num_heads):
        start_index = i * head_dim
        end_index = start_index + head_dim

        interleaved.append(query_weight[start_index:end_index])
        interleaved.append(key_weight[start_index:end_index])
        interleaved.append(value_weight[start_index:end_index])

    return torch.cat(interleaved)


def split_query_key_value_tensor_for_mha(
    query_key_value_weight: torch.Tensor, num_heads: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    original_shape = query_key_value_weight.shape

    query_key_value_weight = query_key_value_weight.view(num_heads, -1)

    query_weight, key_weight, value_weight = query_key_value_weight.chunk(3, -1)

    query_weight = query_weight.reshape(-1, *original_shape[1:])
    key_weight = key_weight.reshape(-1, *original_shape[1:])
    value_weight = value_weight.reshape(-1, *original_shape[1:])

    return query_weight, key_weight, value_weight


def interleave_query_key_value_tensor_for_gqa(
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
    num_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> torch.Tensor:
    query_heads_per_group = num_heads // num_key_value_heads

    interleaved = []
    for i in range(num_key_value_heads):
        start_index = i * query_heads_per_group * head_dim
        end_index = start_index + query_heads_per_group * head_dim
        interleaved.append(query_weight[start_index:end_index])

        start_index = i * head_dim
        end_index = start_index + head_dim
        interleaved.append(key_weight[start_index:end_index])
        interleaved.append(value_weight[start_index:end_index])

    return torch.cat(interleaved)


def split_query_key_value_tensor_for_gqa(
    query_key_value_weight: torch.Tensor, num_heads: int, num_key_value_heads: int, head_dim: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query_heads_per_group = num_heads // num_key_value_heads
    original_shape = query_key_value_weight.shape

    query_key_value_weight = query_key_value_weight.view(num_key_value_heads, (query_heads_per_group + 2), -1)

    query_weight, key_weight, value_weight = query_key_value_weight.split((query_heads_per_group, 1, 1), 1)

    query_weight = query_weight.reshape(-1, *original_shape[1:])
    key_weight = key_weight.reshape(-1, *original_shape[1:])
    value_weight = value_weight.reshape(-1, *original_shape[1:])

    return query_weight, key_weight, value_weight


def interleave_query_key_value_tensor_for_mqa(
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
) -> torch.Tensor:
    # [:] for converting slice to tensor
    return torch.cat([query_weight[:], key_weight[:], value_weight[:]])


def split_query_key_value_tensor_for_mqa(
    query_key_value_weight: torch.Tensor, num_heads: int, head_dim: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return query_key_value_weight.split((num_heads * head_dim, head_dim, head_dim))


def repeat_key_value(x: torch.Tensor, num_heads: int, num_key_value_heads: int) -> torch.Tensor:
    num_groups = num_heads // num_key_value_heads

    if num_groups == 1:
        return x

    if num_key_value_heads == 1:
        return x.expand(-1, num_heads, -1, -1)

    return x.repeat_interleave(num_groups, dim=1)
