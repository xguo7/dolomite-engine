from collections import defaultdict
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from transformers import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

from ...enums import AttentionHeadType, PositionEmbeddingType
from ...modeling_utils import ParameterizedEmbedding, get_normalization_function
from ..gpt_dolomite import GPTDolomiteConfig, GPTDolomiteModel, GPTDolomitePreTrainedModel
from .config import GPTCrossLayerConfig
from .layer import GPTCrossLayerBlock


class GPTCrossLayerPreTrainedModel(GPTDolomitePreTrainedModel):
    config_class = GPTCrossLayerConfig
    _no_split_modules = ["GPTCrossLayerBlock"]

    def __init__(self, config: GPTDolomiteConfig, *inputs, **kwargs):
        GPTDolomitePreTrainedModel.__init__(self, config, *inputs, **kwargs)
        self.sharing_pattern = config.sharing_pattern


class GPTCrossLayerModel(GPTCrossLayerPreTrainedModel, GPTDolomiteModel):
    def __init__(self, config: GPTCrossLayerConfig, **kwargs) -> None:
        GPTCrossLayerPreTrainedModel.__init__(self, config, **kwargs)

        self.attention_head_type = AttentionHeadType(config.attention_head_type)
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.m_emb = config.m_emb
        self.initializer_range = config.initializer_range

        assert (
            self.embed_dim % self.num_heads == 0
        ), f"`embed_dim` ({self.embed_dim}) must be divisible by `num_heads` ({self.num_heads})"

        self.head_dim = self.embed_dim // self.num_heads

        self.wte = ParameterizedEmbedding(config.vocab_size, self.embed_dim, std=self.initializer_range)

        self.drop = nn.Identity() if config.embd_pdrop == 0 else nn.Dropout(config.embd_pdrop)

        global_index, local_index = 0, 0
        # layer_index to (global_index, local_index)
        self.layer_map = [(global_index, local_index)]

        # global_index to layer_index
        sub_layer_map = defaultdict(list)
        sub_layer_map[global_index].append(0)

        for layer_idx in range(1, config.n_layer):
            if self.sharing_pattern[layer_idx] != self.sharing_pattern[layer_idx - 1]:
                global_index += 1
                local_index = 0
            else:
                local_index += 1

            self.layer_map.append((global_index, local_index))
            sub_layer_map[global_index].append(layer_idx)

        self.h = nn.ModuleList(
            [
                GPTCrossLayerBlock(
                    config,
                    self.normalization_implementation,
                    self.attention_implementation,
                    self._use_padding_free_transformer,
                    layer_indices=sub_layer_map[i],
                    layer_idx=i,
                )
                for i in sub_layer_map
            ]
        )
        self.ln_f = get_normalization_function(
            config.normalization_function,
            self.embed_dim,
            eps=config.layer_norm_epsilon,
            normalization_implementation=self.normalization_implementation,
        )

        self.position_embedding_type = PositionEmbeddingType(config.position_embedding_type)
        self._setup_positional_encoding()

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor = None,
        past_key_values: List[torch.Tensor] = None,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        use_cache: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
        cu_seqlens: torch.Tensor = None,
        max_seqlen: torch.Tensor = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        (
            output_hidden_states,
            use_cache,
            return_dict,
            input_shape,
            hidden_states,
            attention_mask,
            position_ids,
            rope_cos_sin,
            past_key_values,
        ) = self._prepare_a_bunch_of_stuff(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        output_shape = input_shape + (hidden_states.size(-1),)

        past_key_values = DynamicCache() if use_cache and past_key_values is None else past_key_values
        all_hidden_states = () if output_hidden_states else None
        for block in self.h:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            hidden_states = block(
                hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                rope_cos_sin=rope_cos_sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, past_key_values, all_hidden_states] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
        )

    def get_global_local_idx(self, index: int) -> Tuple[int, int]:
        return self.layer_map[index]
