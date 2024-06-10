import torch

from ...enums import AttentionHeadType, PositionEmbeddingType
from ...modeling_utils import is_glu
from ..gpt_dolomite import GPTDolomiteConfig


def unshard(
    config: GPTDolomiteConfig, tensor_parallel_state_dicts: list[dict], tensor_parallel_embeddings: bool
) -> dict:
    attention_head_type = AttentionHeadType(config.attention_head_type)
    position_embedding_type = PositionEmbeddingType(config.position_embedding_type)

    # word embeddings
    output_state_dict = _get_embeddings_or_lm_head(
        tensor_parallel_state_dicts, tensor_parallel_embeddings, "transformer.wte.weight"
    )

    # positional embeddings if using learned positional embeddings
    if position_embedding_type == PositionEmbeddingType.learned_absolute:
        output_state_dict.update(
            _get_embeddings_or_lm_head(
                tensor_parallel_state_dicts, tensor_parallel_embeddings, "transformer.wpe.weight"
            )
        )

    # layers
    for layer_idx in range(config.n_layer):
        prefix = f"transformer.h.{layer_idx}."

        # attention
        output_state_dict.update(
            _get_attention_weights(
                tensor_parallel_state_dicts,
                attention_head_type=attention_head_type,
                add_bias=config.add_bias,
                key=prefix,
            )
        )

        # mlp
        output_state_dict.update(
            _get_mlp_weights(
                tensor_parallel_state_dicts, is_glu=is_glu(config.activation_function), add_bias=config.add_bias
            )
        )

    if not config.tie_word_embeddings:
        output_state_dict.update(
            _get_embeddings_or_lm_head(
                tensor_parallel_state_dicts,
                tensor_parallel_embeddings=tensor_parallel_embeddings,
                key="lm_head.weight",
            )
        )


def _get_embeddings_or_lm_head(
    tensor_parallel_state_dicts: list[dict], tensor_parallel_embeddings: bool, key: str
) -> dict:
    output = (
        _concatenate_tensors_from_state_dicts(tensor_parallel_state_dicts, key=key, dim=0)
        if tensor_parallel_embeddings
        else _get_once_from_state_dicts_with_check(tensor_parallel_state_dicts, key=key)
    )
    return {key: output}


def _get_attention_weights(
    tensor_parallel_state_dicts: list[dict], attention_head_type: AttentionHeadType, add_bias: bool, key: str
) -> dict:
    output = {
        key
        + "c_proj.weight": _concatenate_tensors_from_state_dicts(
            tensor_parallel_state_dicts, key=key + "c_proj.weight", dim=1
        )
    }
    if add_bias:
        output[key + "c_proj.bias"] = _get_once_from_state_dicts_with_check(
            tensor_parallel_state_dicts, key=key + "c_proj.bias"
        )

    if attention_head_type == AttentionHeadType.mha:
        pass
    elif attention_head_type == AttentionHeadType.gqa:
        pass
    elif attention_head_type == AttentionHeadType.mqa:
        q_weight = _concatenate_tensors_from_state_dicts(tensor_parallel_state_dicts, key=key + "q_attn.weight", dim=0)
        kv_weight = _get_once_from_state_dicts_with_check(tensor_parallel_state_dicts, key + "kv_attn.weight")

        output[key + "c_attn.weight"] = torch.cat([q_weight, kv_weight])

        if add_bias:
            q_bias = _concatenate_tensors_from_state_dicts(tensor_parallel_state_dicts, key=key + "q_attn.bias", dim=0)
            kv_bias = _get_once_from_state_dicts_with_check(tensor_parallel_state_dicts, key + "kv_attn.bias")

            output[key + "c_attn.bias"] = torch.cat([q_bias, kv_bias])

    return output


def _get_mlp_weights(tensor_parallel_state_dicts: list[dict], is_glu: bool, add_bias: bool, key: str) -> dict:
    output = {
        key
        + "c_proj.weight": _concatenate_tensors_from_state_dicts(
            tensor_parallel_state_dicts, key=key + "c_proj.weight", dim=1
        )
    }
    if add_bias:
        output[key + "c_proj.bias"] = _get_once_from_state_dicts_with_check(
            tensor_parallel_state_dicts, key=key + "c_proj.bias"
        )

    if is_glu:
        pass
    else:
        output[key + "c_fc.weight"] = _concatenate_tensors_from_state_dicts(
            tensor_parallel_state_dicts, key=key + "c_fc.weight", dim=0
        )
        if add_bias:
            output[key + "c_fc.bias"] = _concatenate_tensors_from_state_dicts(
                tensor_parallel_state_dicts, key=key + "c_fc.bias", dim=0
            )

    return output


def _concatenate_tensors_from_state_dicts(tensor_parallel_state_dicts: list[dict], key: str, dim: int) -> torch.Tensor:
    tensor_list = [state_dict[key] for state_dict in tensor_parallel_state_dicts]
    tensor = torch.cat(tensor_list, dim=dim)
    return tensor


def _get_once_from_state_dicts_with_check(
    tensor_parallel_state_dicts: list[dict], key: str, check_equal: bool = True
) -> torch.Tensor:
    output = tensor_parallel_state_dicts[0][key]
    if check_equal:
        for state_dict in tensor_parallel_state_dicts[1:]:
            assert output.equal(state_dict[key])
    return output
