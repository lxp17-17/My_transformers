# # coding=utf-8
# # Copyright 2023 the HuggingFace Inc. team. All rights reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# """PyTorch Llava model."""

# from dataclasses import dataclass
# from typing import Optional, Union

# import torch
# import torch.utils.checkpoint
# from torch import nn

# from ...activations import ACT2FN
# from ...generation import GenerationMixin
# from ...modeling_flash_attention_utils import FlashAttentionKwargs
# from ...modeling_outputs import BaseModelOutputWithPast, ModelOutput
# from ...modeling_utils import PreTrainedModel
# from ...processing_utils import Unpack
# from ...utils import LossKwargs, auto_docstring, can_return_tuple, is_torchdynamo_compiling, logging
# from ..auto import AutoModel
# from .configuration_llava import LlavaConfig


# logger = logging.get_logger(__name__)


# @dataclass
# class LlavaModelOutputWithPast(BaseModelOutputWithPast):
#     """
#     Base class for Llava outputs, with hidden states and attentions.

#     Args:
#         last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
#             Sequence of hidden-states at the output of the last layer of the model.
#         past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
#             Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
#             `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

#             Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
#             `past_key_values` input) to speed up sequential decoding.
#         hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
#             Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
#             one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

#             Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
#         attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
#             Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
#             sequence_length)`.

#             Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
#             heads.
#         image_hidden_states (`torch.FloatTensor`, *optional*):
#             A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
#             image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
#     """

#     image_hidden_states: Optional[torch.FloatTensor] = None


# @dataclass
# class LlavaCausalLMOutputWithPast(ModelOutput):
#     """
#     Base class for Llava causal language model (or autoregressive) outputs.

#     Args:
#         loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
#             Language modeling loss (for next-token prediction).
#         logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
#             Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
#         past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
#             Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
#             `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

#             Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
#             `past_key_values` input) to speed up sequential decoding.
#         hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
#             Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
#             one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

#             Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
#         attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
#             Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
#             sequence_length)`.

#             Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
#             heads.
#         image_hidden_states (`torch.FloatTensor`, *optional*):
#             A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
#             image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
#     """

#     loss: Optional[torch.FloatTensor] = None
#     logits: Optional[torch.FloatTensor] = None
#     past_key_values: Optional[list[torch.FloatTensor]] = None
#     hidden_states: Optional[tuple[torch.FloatTensor]] = None
#     attentions: Optional[tuple[torch.FloatTensor]] = None
#     image_hidden_states: Optional[torch.FloatTensor] = None


# class LlavaMultiModalProjector(nn.Module):
#     def __init__(self, config: LlavaConfig):
#         super().__init__()
#         # We have hidden_size * the number of vision feature layers
#         num_feature_layers = 1 if isinstance(config.vision_feature_layer, int) else len(config.vision_feature_layer)
#         self.linear_1 = nn.Linear(
#             config.vision_config.hidden_size * num_feature_layers,
#             config.text_config.hidden_size,
#             bias=config.multimodal_projector_bias,
#         )
#         self.act = ACT2FN[config.projector_hidden_act]
#         self.linear_2 = nn.Linear(
#             config.text_config.hidden_size, config.text_config.hidden_size, bias=config.multimodal_projector_bias
#         )

#     def forward(self, image_features):
#         hidden_states = self.linear_1(image_features)
#         hidden_states = self.act(hidden_states)
#         hidden_states = self.linear_2(hidden_states)
#         return hidden_states


# @auto_docstring
# class LlavaPreTrainedModel(PreTrainedModel):
#     config_class = LlavaConfig
#     base_model_prefix = ""
#     supports_gradient_checkpointing = True
#     _skip_keys_device_placement = "past_key_values"
#     _supports_cache_class = True
#     _supports_flash_attn_2 = True
#     _supports_sdpa = True
#     _supports_quantized_cache = True
#     _supports_static_cache = True
#     _supports_flex_attn = True
#     _supports_attention_backend = True

#     def _init_weights(self, module):
#         # important: this ported version of Llava isn't meant for training from scratch - only
#         # inference and fine-tuning - so the proper init weights code has been removed - the original codebase
#         # https://github.com/haotian-liu/LLaVA/tree/main/llava should serve for that purpose
#         std = getattr(self.config, "initializer_range", self.config.get_text_config().initializer_range)

#         if isinstance(module, nn.Linear):
#             module.weight.data.normal_(mean=0.0, std=std)
#             if module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.weight.data.fill_(1.0)
#             module.bias.data.zero_()


# @auto_docstring(
#     custom_intro="""
#     The Llava model which consists of a vision backbone and a language model, without a language modeling head.
#     """
# )
# class LlavaModel(LlavaPreTrainedModel):
#     _checkpoint_conversion_mapping = {"language_model.model": "language_model"}

#     def __init__(self, config: LlavaConfig):
#         super().__init__(config)
#         self.vision_tower = AutoModel.from_config(config.vision_config)

#         self.multi_modal_projector = LlavaMultiModalProjector(config)
#         self.language_model = AutoModel.from_config(config.text_config)
#         self.post_init()

#     def get_input_embeddings(self):
#         return self.language_model.get_input_embeddings()

#     def set_input_embeddings(self, value):
#         self.language_model.set_input_embeddings(value)

#     def get_image_features(
#         self,
#         pixel_values: torch.FloatTensor,
#         vision_feature_layer: Optional[Union[int, list[int]]] = None,
#         vision_feature_select_strategy: Optional[str] = None,
#         **kwargs,
#     ):
#         """
#         Obtains image last hidden states from the vision tower and apply multimodal projection.

#         Args:
#             pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`):
#                The tensors corresponding to the input images.
#             vision_feature_layer (`Union[int, list[int]]`, *optional*):
#                 The index of the layer to select the vision feature. If multiple indices are provided,
#                 the vision feature of the corresponding indices will be concatenated to form the
#                 vision features.
#             vision_feature_select_strategy (`str`, *optional*):
#                 The feature selection strategy used to select the vision feature from the vision backbone.
#                 Can be one of `"default"` or `"full"`
#         Returns:
#             image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
#         """
#         vision_feature_layer = (
#             vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
#         )
#         vision_feature_select_strategy = (
#             vision_feature_select_strategy
#             if vision_feature_select_strategy is not None
#             else self.config.vision_feature_select_strategy
#         )

#         if vision_feature_select_strategy not in ["default", "full"]:
#             raise ValueError(f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}")

#         kwargs = {k: v for k, v in kwargs.items() if v is not None}
#         # this is not memory efficient at all (output_hidden_states=True) will save all the hidden states.
#         image_outputs = self.vision_tower(pixel_values, output_hidden_states=True, **kwargs)

#         # If we have one vision feature layer, return the corresponding hidden states,
#         # otherwise, select the hidden states of each feature layer and concatenate them
#         if isinstance(vision_feature_layer, int):
#             selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
#             if vision_feature_select_strategy == "default":
#                 selected_image_feature = selected_image_feature[:, 1:]
#         else:
#             hs_pool = [image_outputs.hidden_states[layer_idx] for layer_idx in vision_feature_layer]
#             # For default; crop CLS from each hidden state in the hidden state pool
#             if vision_feature_select_strategy == "default":
#                 hs_pool = [hs[:, 1:] for hs in hs_pool]
#             selected_image_feature = torch.cat(hs_pool, dim=-1)

#         image_features = self.multi_modal_projector(selected_image_feature)

#         if "image_sizes" in kwargs:
#             split_sizes = [
#                 (height // self.vision_tower.patch_size) * (width // self.vision_tower.patch_size)
#                 for height, width in kwargs["image_sizes"]
#             ]
#             image_features = torch.split(image_features.squeeze(0), split_sizes)
#         else:
#             image_features = list(image_features)
#         return image_features

#     @can_return_tuple
#     @auto_docstring
#     def forward(
#         self,
#         input_ids: torch.LongTensor = None,
#         pixel_values: torch.FloatTensor = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[list[torch.FloatTensor]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         vision_feature_layer: Optional[Union[int, list[int]]] = None,
#         vision_feature_select_strategy: Optional[str] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         cache_position: Optional[torch.LongTensor] = None,
#         image_sizes: torch.Tensor = None,
#         **kwargs: Unpack[FlashAttentionKwargs],
#     ) -> Union[tuple, LlavaModelOutputWithPast]:
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#         vision_feature_layer = (
#             vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
#         )
#         vision_feature_select_strategy = (
#             vision_feature_select_strategy
#             if vision_feature_select_strategy is not None
#             else self.config.vision_feature_select_strategy
#         )

#         if (input_ids is None) ^ (inputs_embeds is not None):
#             raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

#         if inputs_embeds is None:
#             inputs_embeds = self.get_input_embeddings()(input_ids)

#         if pixel_values is not None:
#             image_features = self.get_image_features(
#                 pixel_values=pixel_values,
#                 vision_feature_layer=vision_feature_layer,
#                 vision_feature_select_strategy=vision_feature_select_strategy,
#                 image_sizes=image_sizes,
#             )
#             image_features = torch.cat(image_features, dim=0)

#             if input_ids is None:
#                 special_image_mask = inputs_embeds == self.get_input_embeddings()(
#                     torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
#                 )
#                 n_image_tokens = (special_image_mask).sum(dim=1).sum(dim=0)[0]
#             else:
#                 special_image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1)
#                 special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
#                 n_image_tokens = (input_ids == self.config.image_token_id).sum()

#             if not is_torchdynamo_compiling() and inputs_embeds[special_image_mask].numel() != image_features.numel():
#                 n_image_tokens = (input_ids == self.config.image_token_id).sum()
#                 n_image_features = image_features.shape[0] * image_features.shape[1]
#                 raise ValueError(
#                     f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
#                 )
#             image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
#             inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

#         outputs = self.language_model(
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=True,
#             cache_position=cache_position,
#             **kwargs,
#         )

#         return LlavaModelOutputWithPast(
#             last_hidden_state=outputs.last_hidden_state,
#             past_key_values=outputs.past_key_values,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#             image_hidden_states=image_features if pixel_values is not None else None,
#         )


# class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


# @auto_docstring(
#     custom_intro="""
#     The LLAVA model which consists of a vision backbone and a language model.
#     """
# )
# class LlavaForConditionalGeneration(LlavaPreTrainedModel, GenerationMixin):
#     _checkpoint_conversion_mapping = {
#         "^language_model.model": "model.language_model",
#         "^vision_tower": "model.vision_tower",
#         "^multi_modal_projector": "model.multi_modal_projector",
#         "^language_model.lm_head": "lm_head",
#     }
#     _tied_weights_keys = ["lm_head.weight"]

#     def __init__(self, config: LlavaConfig):
#         super().__init__(config)
#         self.model = LlavaModel(config)
#         self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
#         self.post_init()

#     def get_input_embeddings(self):
#         return self.model.get_input_embeddings()

#     def set_input_embeddings(self, value):
#         self.model.set_input_embeddings(value)

#     def get_output_embeddings(self) -> nn.Module:
#         return self.lm_head

#     def set_output_embeddings(self, new_embeddings):
#         self.lm_head = new_embeddings

#     def set_decoder(self, decoder):
#         self.model = decoder

#     def get_decoder(self):
#         return self.model

#     def get_image_features(
#         self,
#         pixel_values: torch.FloatTensor,
#         vision_feature_layer: Optional[Union[int, list[int]]] = None,
#         vision_feature_select_strategy: Optional[str] = None,
#         **kwargs,
#     ):
#         return self.model.get_image_features(
#             pixel_values=pixel_values,
#             vision_feature_layer=vision_feature_layer,
#             vision_feature_select_strategy=vision_feature_select_strategy,
#             **kwargs,
#         )

#     # Make modules available throught conditional class for BC
#     @property
#     def language_model(self):
#         return self.model.language_model

#     @property
#     def vision_tower(self):
#         return self.model.vision_tower

#     @property
#     def multi_modal_projector(self):
#         return self.model.multi_modal_projector

#     @can_return_tuple
#     @auto_docstring
#     def forward(
#         self,
#         input_ids: torch.LongTensor = None,
#         pixel_values: torch.FloatTensor = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[list[torch.FloatTensor]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         vision_feature_layer: Optional[Union[int, list[int]]] = None,
#         vision_feature_select_strategy: Optional[str] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         cache_position: Optional[torch.LongTensor] = None,
#         logits_to_keep: Union[int, torch.Tensor] = 0,
#         image_sizes: Optional[torch.Tensor] = None,
#         **kwargs: Unpack[KwargsForCausalLM],
#     ) -> Union[tuple, LlavaCausalLMOutputWithPast]:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
#             Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
#             config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
#             (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

#         Example:

#         ```python
#         >>> from PIL import Image
#         >>> import requests
#         >>> from transformers import AutoProcessor, LlavaForConditionalGeneration

#         >>> model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
#         >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

#         >>> prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
#         >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
#         >>> image = Image.open(requests.get(url, stream=True).raw)

#         >>> inputs = processor(images=image, text=prompt, return_tensors="pt")

#         >>> # Generate
#         >>> generate_ids = model.generate(**inputs, max_new_tokens=15)
#         >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
#         "USER:  \nWhat's the content of the image? ASSISTANT: The image features a busy city street with a stop sign prominently displayed"
#         ```"""
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#         vision_feature_layer = (
#             vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
#         )
#         vision_feature_select_strategy = (
#             vision_feature_select_strategy
#             if vision_feature_select_strategy is not None
#             else self.config.vision_feature_select_strategy
#         )

#         outputs = self.model(
#             input_ids=input_ids,
#             pixel_values=pixel_values,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             vision_feature_layer=vision_feature_layer,
#             vision_feature_select_strategy=vision_feature_select_strategy,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=True,
#             cache_position=cache_position,
#             image_sizes=image_sizes,
#             **kwargs,
#         )

#         hidden_states = outputs[0]
#         # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
#         slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
#         logits = self.lm_head(hidden_states[:, slice_indices, :])

#         loss = None
#         if labels is not None:
#             loss = self.loss_function(
#                 logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
#             )

#         return LlavaCausalLMOutputWithPast(
#             loss=loss,
#             logits=logits,
#             past_key_values=outputs.past_key_values,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#             image_hidden_states=outputs.image_hidden_states,
#         )

#     def prepare_inputs_for_generation(
#         self,
#         input_ids,
#         past_key_values=None,
#         inputs_embeds=None,
#         pixel_values=None,
#         attention_mask=None,
#         cache_position=None,
#         logits_to_keep=None,
#         **kwargs,
#     ):
#         # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

#         model_inputs = super().prepare_inputs_for_generation(
#             input_ids,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             attention_mask=attention_mask,
#             cache_position=cache_position,
#             logits_to_keep=logits_to_keep,
#             **kwargs,
#         )

#         if cache_position[0] == 0:
#             # If we're in cached decoding stage, pixel values should be None because input ids do not contain special image token anymore
#             # Otherwise we need pixel values to be passed to model
#             model_inputs["pixel_values"] = pixel_values

#         return model_inputs


# __all__ = ["LlavaForConditionalGeneration", "LlavaPreTrainedModel", "LlavaModel"]


# coding=utf-8
# Copyright 2023 the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Llava model."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from ...activations import ACT2FN
from ...generation import GenerationMixin
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutputWithPast, ModelOutput
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import LossKwargs, auto_docstring, can_return_tuple, is_torchdynamo_compiling, logging
from ..auto import AutoModel
from .configuration_llava import LlavaConfig


logger = logging.get_logger(__name__)


@dataclass
class LlavaModelOutputWithPast(BaseModelOutputWithPast):
    """
    Base class for Llava outputs, with hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`torch.FloatTensor`, *optional*):
            A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
            image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    image_hidden_states: Optional[torch.FloatTensor] = None


@dataclass
class LlavaCausalLMOutputWithPast(ModelOutput):
    """
    Base class for Llava causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`torch.FloatTensor`, *optional*):
            A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
            image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None


class LlavaMultiModalProjector(nn.Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()
        # We have hidden_size * the number of vision feature layers
        #num_feature_layers是视觉特征的层数，如果vision_feature_layer是int，则num_feature_layers=1，否则num_feature_layers=len(vision_feature_layer)
        num_feature_layers = 1 if isinstance(config.vision_feature_layer, int) else len(config.vision_feature_layer)
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size * num_feature_layers,#这是1024*1=1024，是视觉特征的维度
            config.text_config.hidden_size, # 这是4096，是语言模型的隐藏层维度
            bias=config.multimodal_projector_bias, # 是否使用偏置
        )
        # 激活函数，这里使用GELU
        self.act = ACT2FN[config.projector_hidden_act]
        # 第二层线性层，将4096维的特征投影到4096维
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size, config.text_config.hidden_size, bias=config.multimodal_projector_bias
        )

    def forward(self, image_features):
        #是一个 MLP (多层感知机)！它由 线性层 -> GELU激活 -> 线性层 组成。
        # 这比论文里说的“a simple linear layer”要稍微复杂一点，但本质上就是一个投影网络。
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


@auto_docstring
class LlavaPreTrainedModel(PreTrainedModel):
    config_class = LlavaConfig
    base_model_prefix = ""
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        # important: this ported version of Llava isn't meant for training from scratch - only
        # inference and fine-tuning - so the proper init weights code has been removed - the original codebase
        # https://github.com/haotian-liu/LLaVA/tree/main/llava should serve for that purpose
        std = getattr(self.config, "initializer_range", self.config.get_text_config().initializer_range)

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()


@auto_docstring(
    custom_intro="""
    The Llava model which consists of a vision backbone and a language model, without a language modeling head.
    """
)
class LlavaModel(LlavaPreTrainedModel):
    # _checkpoint_conversion_mapping用于映射检查点的权重名称
    _checkpoint_conversion_mapping = {"language_model.model": "language_model"}

    def __init__(self, config: LlavaConfig):
        # 初始化LlavaModel类，继承自LlavaPreTrainedModel
            #这个初始化过程只创建了模型结构，没有加载预训练权重#
            #  这个初始化过程只创建了模型结构，没有加载预训练权重
            # 要使用预训练权重，需要额外调用from_pretrained()方法
            # 模型设计为可以分别加载：
            # 视觉编码器的预训练权重（通常来自CLIP）
            # 语言模型的预训练权重（通常来自LLaMA）
            # 多模态投影器的权重（需要单独训练或微调）
            # 这种设计允许模型：
            # 灵活地使用不同的预训练权重
            # 支持增量训练和微调
            # 可以分别更新不同组件的权重
        super().__init__(config)
        # 注意它们的初始化方式（是否加载预训练权重）。
        # 从配置中加载视觉模型
         # - hidden_size: 1024 (CLIP ViT-L输出维度)
        # - image_size: 336 (输入图像尺寸)
        # - patch_size: 14 (patch大小)
        self.vision_tower = AutoModel.from_config(config.vision_config)
        # 初始化多模态投影器
        # - 输入维度: 1024 (CLIP输出)
        # - 输出维度: 4096 (LLaMA隐藏层维度)
        # - 激活函数: GELU
        self.multi_modal_projector = LlavaMultiModalProjector(config)  #
        # 从配置中加载语言模型
        # - vocab_size: 32000
        # - hidden_size: 4096，这是一个新初始化的投影层，用于将视觉特征(1024维)投影到语言模型空间(4096维)
        # - num_attention_heads: 32
        # 包括了LLaMA的架构，包括了嵌入层、位置编码、编码器层、输出层等。包含两层线性层和GELU激活函数
        self.language_model = AutoModel.from_config(config.text_config)
        # 执行初始化后的处理
        self.post_init()

    def get_input_embeddings(self):
        # 获取语言模型的输入嵌入层
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        # 设置语言模型的输入嵌入层
        self.language_model.set_input_embeddings(value)

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: Optional[Union[int, List[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        **kwargs,
    ):
        """
        从视觉塔中获取图像的最后隐藏状态并应用多模态投影。

        参数:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`):
               输入图像对应的张量。
            vision_feature_layer (`Union[int, List[int]]`, *optional*):
                选择视觉特征的层的索引。如果提供多个索引，将连接相应索引的视觉特征以形成视觉特征。
            vision_feature_select_strategy (`str`, *optional*):
                用于从视觉骨干中选择视觉特征的特征选择策略。
                可以是 `"default"` 或 `"full"` 之一。
        返回:
            image_features (`torch.Tensor`): 形状为 `(num_images, image_length, embed_dim)` 的图像特征张量。
        """
        # 确定使用的视觉特征层
        vision_feature_layer = (
            # 如果vision_feature_layer为None，则使用config中的vision_feature_layer
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        # 确定使用的特征选择策略
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        # 检查特征选择策略是否有效
        if vision_feature_select_strategy not in ["default", "full"]:
            raise ValueError(f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}")

        # 过滤掉None值的参数
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        # 获取视觉模型的输出，注意这里会保存所有隐藏状态，可能不够内存高效
        image_outputs = self.vision_tower(pixel_values, output_hidden_states=True, **kwargs)
        # #batch_size: 批次大小
        # channels: 3 (RGB图像)
        # height: 336 (默认图像高度)
        # width: 336 (默认图像宽度)

        # 如果只有一个视觉特征层，返回相应的隐藏状态；否则，选择每个特征层的隐藏状态并连接它们，这里是将所有层连接起来
        if isinstance(vision_feature_layer, int):
            selected_image_feature = image_outputs.hidden_states[vision_feature_layer]  #
            if vision_feature_select_strategy == "default":
                selected_image_feature = selected_image_feature[:, 1:]  # 裁剪掉CLS，因为CLS是特殊标记，不用于后续的文本生成
        else:
            hs_pool = [image_outputs.hidden_states[layer_idx] for layer_idx in vision_feature_layer]
            # 对于默认策略，从每个隐藏状态中裁剪CLS
            if vision_feature_select_strategy == "default":
                hs_pool = [hs[:, 1:] for hs in hs_pool]
            selected_image_feature = torch.cat(hs_pool, dim=-1)  # 将所有层连接起来

        # 应用多模态投影器
        image_features = self.multi_modal_projector(selected_image_feature)

        # 如果提供了图像尺寸，按尺寸分割特征
        #         图像被分割成patches:
        # patch_size = 14
        # 每个patch大小: 14×14像素
        # 图像被分割成: (336/14) × (336/14) = 24×24 = 576个patches

        if "image_sizes" in kwargs:
            ## 如果提供了图像尺寸，按尺寸分割特征
            split_sizes = [
                (height // self.vision_tower.patch_size) * (width // self.vision_tower.patch_size)
                for height, width in kwargs["image_sizes"]
            ]
            image_features = torch.split(image_features.squeeze(0), split_sizes)
        else:
            image_features = list(image_features)
        return image_features

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,# 输入的文本ID
        pixel_values: torch.FloatTensor = None, # 输入的图像张量
        attention_mask: Optional[torch.Tensor] = None, # 注意力掩码
        position_ids: Optional[torch.LongTensor] = None, # 位置ID
        past_key_values: Optional[List[torch.FloatTensor]] = None, # 过去的键值
        inputs_embeds: Optional[torch.FloatTensor] = None, # 输入的嵌入
        vision_feature_layer: Optional[Union[int, List[int]]] = None, # 选择视觉特征的层的索引
        vision_feature_select_strategy: Optional[str] = None, # 用于从视觉骨干中选择视觉特征的特征选择策略
        use_cache: Optional[bool] = None,# 是否使用缓存
        output_attentions: Optional[bool] = None, # 是否输出注意力
        output_hidden_states: Optional[bool] = None, # 是否输出隐藏状态
        return_dict: Optional[bool] = None, # 是否返回字典
            cache_position: Optional[torch.LongTensor] = None, # 缓存位置
        image_sizes: torch.Tensor = None, # 图像尺寸
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, LlavaModelOutputWithPast]:
        # 确定是否输出注意力
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 确定是否输出隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 确定是否返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 确定使用的视觉特征层
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        # 确定使用的特征选择策略
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        # 确保input_ids和inputs_embeds中只有一个被指定
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # 如果没有提供inputs_embeds，则通过input_ids获取
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # 如果提供了图像数据，获取图像特征
        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values, # 输入图像对应的张量
                vision_feature_layer=vision_feature_layer, # 选择视觉特征的层的索引
                vision_feature_select_strategy=vision_feature_select_strategy, # 用于从视觉骨干中选择视觉特征的特征选择策略
                image_sizes=image_sizes, # 图像尺寸
            )
            image_features = torch.cat(image_features, dim=0)

            # 处理特殊图像标记
            if input_ids is None:
                special_image_mask = inputs_embeds == self.get_input_embeddings()(
                    torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
                n_image_tokens = (special_image_mask).sum(dim=1).sum(dim=0)[0]
            else:
                special_image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1)
                special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
                n_image_tokens = (input_ids == self.config.image_token_id).sum()

            # 检查图像特征和图像标记是否匹配
            if not is_torchdynamo_compiling() and inputs_embeds[special_image_mask].numel() != image_features.numel():
                n_image_tokens = (input_ids == self.config.image_token_id).sum()
                n_image_features = image_features.shape[0] * image_features.shape[1]
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            # 将图像特征转移到输入嵌入的设备和数据类型
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            # 使用图像特征更新输入嵌入，将图像特征插入到输入嵌入中，用于后续的文本生成
            #融合！！！最终将图像特征插入到文本序列中的特殊图像token位置，实现多模态融合
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        # 获取语言模型的输出
        outputs = self.language_model(
            attention_mask=attention_mask, # 注意力掩码
            position_ids=position_ids, # 位置ID
            past_key_values=past_key_values, # 过去的键值
            inputs_embeds=inputs_embeds, # 输入的嵌入
            use_cache=use_cache, # 是否使用缓存
            output_attentions=output_attentions, # 是否输出注意力
            output_hidden_states=output_hidden_states, # 是否输出隐藏状态
            return_dict=True, # 是否返回字典
            cache_position=cache_position, # 缓存位置
            **kwargs,
        )

        # 返回LlavaModelOutputWithPast对象，包含最后的隐藏状态、过去的键值、隐藏状态、注意力和图像隐藏状态
        return LlavaModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


@auto_docstring(
    custom_intro="""
    The LLAVA model which consists of a vision backbone and a language model.
    """
)
# LlavaForConditionalGeneration类是LLaVA模型的核心类之一，负责条件生成任务。
# 该类继承自LlavaPreTrainedModel和GenerationMixin，结合了预训练模型的特性和生成任务的功能。
class LlavaForConditionalGeneration(LlavaPreTrainedModel, GenerationMixin):
    # _checkpoint_conversion_mapping用于映射检查点的权重名称，确保在加载预训练权重时能够正确匹配。
    _checkpoint_conversion_mapping = {
        "^language_model.model": "model.language_model",
        "^vision_tower": "model.vision_tower",
        "^multi_modal_projector": "model.multi_modal_projector",
        "^language_model.lm_head": "lm_head",
    }
    # _tied_weights_keys用于指定需要共享权重的模块。
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: LlavaConfig):
        # 初始化方法，负责模型的构建。
        super().__init__(config)
        # LlavaModel是LLaVA的核心模型，包含视觉编码器、语言模型和多模态投影层。
        self.model = LlavaModel(config)
        # lm_head是语言模型的输出层，这是线性层，将隐藏状态转换为词汇表大小的向量。
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        # post_init用于执行初始化后的处理。
        self.post_init()

    def get_input_embeddings(self):
        # 获取输入嵌入层，通常是语言模型的嵌入层。
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        # 设置输入嵌入层。
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        # 获取输出嵌入层，即语言模型的输出层。
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        # 设置输出嵌入层。
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        # 设置解码器，通常是语言模型。
        self.model = decoder

    def get_decoder(self):
        # 获取解码器。
        return self.model

    # 通过属性访问语言模型、视觉编码器和多模态投影层。
    @property
    def language_model(self):
        return self.model.language_model

    @property
    def vision_tower(self):
        return self.model.vision_tower

    @property
    def multi_modal_projector(self): # 多模态投影层
        return self.model.multi_modal_projector

    # forward方法是模型接收输入并生成输出的核心逻辑。
    @can_return_tuple
    @auto_docstring
    def forward(
        #这是在LlavaForConditionalGeneration类中定义的forward方法，用于接收多模态输入，生成文本响应。
        self,
        input_ids: torch.LongTensor = None,# 输入的文本ID
        attention_mask: Optional[torch.Tensor] = None, # 注意力掩码
        position_ids: Optional[torch.Tensor] = None, # 位置ID
        pixel_values: Optional[torch.Tensor] = None, # 输入的图像张量
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None, # 过去的键值
        inputs_embeds: Optional[torch.FloatTensor] = None, # 输入的嵌入
        use_cache: Optional[bool] = None, # 是否使用缓存
        output_attentions: Optional[bool] = None, # 是否输出注意力
        output_hidden_states: Optional[bool] = None, # 是否输出隐藏状态
        return_dict: Optional[bool] = None, # 是否返回字典
        cache_position: Optional[torch.Tensor] = None, # 缓存位置
        **kwargs,
    ) -> Union[LlavaModelOutputWithPast, Tuple[torch.FloatTensor]]:
        
        # 视觉特征提取：通过vision_tower处理pixel_values，提取视觉特征。
        visual_features = self.vision_tower(pixel_values)
         # image_features 是 vision_tower 的输出
        print("我加的1：Shape BEFORE projector:", visual_features.shape)
        # 特征投影：通过multi_modal_projector将视觉特征转换为对齐后的image_features。从1024维投影到4096维
        image_features = self.multi_modal_projector(visual_features)
        print("我加的2：Shape AFTER projector:", image_features.shape)
        # 特征拼接/插入：将image_features插入到语言模型的输入序列中。
        # 语言模型推理：通过language_model接收多模态输入，生成文本响应。
        
        outputs = self.language_model(# 语言模型接收多模态输入，生成文本响应。
            input_ids=input_ids,# 输入的文本ID
            attention_mask=attention_mask,# 注意力掩码
            position_ids=position_ids,# 位置ID
            past_key_values=past_key_values,# 过去的键值
            inputs_embeds=inputs_embeds,# 输入的嵌入，将图像特征插入到输入嵌入中，用于后续的文本生成
            use_cache=use_cache,# 是否使用缓存
            output_attentions=output_attentions,# 是否输出注意力
            output_hidden_states=output_hidden_states,# 是否输出隐藏状态
            return_dict=return_dict,# 是否返回字典
            cache_position=cache_position,# 缓存位置
            **kwargs,
        )
        # 损失计算：如果提供了labels，计算损失。
        # 返回模型的输出
        return outputs

    # 检查点转换映射，用于将旧模型的权重映射到新模型
    _checkpoint_conversion_mapping = {
        "^language_model.model": "model.language_model",
        "^vision_tower": "model.vision_tower",
        "^multi_modal_projector": "model.multi_modal_projector",
        "^language_model.lm_head": "lm_head",
    }
    # 绑定权重的键，确保权重共享
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: LlavaConfig):
        # 初始化模型，设置配置
        super().__init__(config)
        # 创建 Llava 模型实例
        self.model = LlavaModel(config)
        # 定义语言模型的输出层
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        # 初始化后处理
        self.post_init()

    def get_input_embeddings(self):
        # 获取输入嵌入层
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        # 设置输入嵌入层
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        # 获取输出嵌入层
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        # 设置输出嵌入层
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        # 设置解码器
        self.model = decoder

    def get_decoder(self):
        # 获取解码器
        return self.model

    # 通过条件类使模块可用以保持向后兼容性
    @property
    def language_model(self):
        # 获取语言模型
        return self.model.language_model

    @property
    def vision_tower(self):
        # 获取视觉塔
        return self.model.vision_tower

    @property
    def multi_modal_projector(self):
        # 获取多模态投影器
        return self.model.multi_modal_projector

    @can_return_tuple
    @auto_docstring
    def forward(
        # 前向传播，接收多模态输入，生成文本响应。
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[Union[int, List[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, LlavaCausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            用于计算掩码语言建模损失的标签。索引应在 `[0, ..., config.vocab_size]` 或 -100 之间（参见 `input_ids` 文档字符串）。索引设置为 `-100` 的标记被忽略（掩码），仅对 `[0, ..., config.vocab_size]` 中的标签标记计算损失。

        示例：

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, LlavaForConditionalGeneration

        >>> model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
        >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

        >>> prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, text=prompt, return_tensors="pt")

        >>> # 生成
        >>> generate_ids = model.generate(**inputs, max_new_tokens=15)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "USER:  \nWhat's the content of the image? ASSISTANT: The image features a busy city street with a stop sign prominently displayed"
        ```"""
        # 设置输出注意力和隐藏状态
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        # 调用模型
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            image_sizes=image_sizes,
            **kwargs,
        )

        hidden_states = outputs[0]
        # 仅计算必要的logits，如果不计算损失则不将其上采样为浮点数
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # 计算损失
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return LlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        attention_mask=None,
        cache_position=None,
        logits_to_keep=None,
        **kwargs,
    ):
        # 重写 -- 在特定情况下我们不希望将图像输入传递给模型

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        if cache_position[0] == 0:
            # 如果我们处于缓存解码阶段，像素值应为 None，因为输入 ID 不再包含特殊图像标记
            # 否则我们需要将像素值传递给模型
            model_inputs["pixel_values"] = pixel_values

        return model_inputs


__all__ = ["LlavaForConditionalGeneration", "LlavaPreTrainedModel", "LlavaModel"]
