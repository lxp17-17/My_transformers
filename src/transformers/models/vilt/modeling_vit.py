# coding=utf-8
# Copyright 2021 Google AI, Ross Wightman, The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch ViT model."""

import collections.abc
import math
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
    MaskedImageModelingOutput,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import auto_docstring, logging, torch_int
from .configuration_vit import ViTConfig


logger = logging.get_logger(__name__)


# ViT模型的主要组件类定义

class ViTEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.
    """

    def __init__(self, config: ViTConfig, use_mask_token: bool = False) -> None:
        """
        参数:
            config: ViT配置对象,包含模型超参数
            use_mask_token: 是否使用mask token,用于掩码图像建模任务
        """
        super().__init__()

        # [1,1,hidden_size] 可学习的分类token
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        # [1,1,hidden_size] 可选的mask token,用于掩码图像建模
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if use_mask_token else None
        # 图像块嵌入层
        self.patch_embeddings = ViTPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        # [1,num_patches+1,hidden_size] 位置编码,+1是为了CLS token
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.patch_size = config.patch_size
        self.config = config

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        对位置编码进行插值,使其适应不同分辨率的输入图像，如384×384或更大，直接使用原始位置编码会导致维度不匹配
        
        参数:
            embeddings: 输入嵌入 [batch_size, seq_len, hidden_size]
            height: 输入图像高度
            width: 输入图像宽度
            
        返回:
            插值后的位置编码 [batch_size, seq_len, hidden_size]
        """
        # 获取patch数量和维度
        num_patches = embeddings.shape[1] - 1  # 减去CLS token
        num_positions = self.position_embeddings.shape[1] - 1
        
        # 如果patch数量相同且高宽相等,直接返回原始位置编码
        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embeddings

        # 分离CLS token的位置编码和patch的位置编码
        class_pos_embed = self.position_embeddings[:, :1]  # [1,1,hidden_size]
        patch_pos_embed = self.position_embeddings[:, 1:]  # [1,num_patches,hidden_size]

        dim = embeddings.shape[-1]
        
        # 计算新的高宽
        new_height = height // self.patch_size
        new_width = width // self.patch_size

        # 重排位置编码为2D形式进行插值
        sqrt_num_positions = torch_int(num_positions**0.5)
        # [1,sqrt_num,sqrt_num,dim] -> [1,dim,sqrt_num,sqrt_num]
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        # 双三次插值到目标大小
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )

        # [1,dim,h,w] -> [1,h*w,dim]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        # 拼接CLS token的位置编码
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(
        self,
        pixel_values: torch.Tensor,  # [batch_size, channels, height, width]
        bool_masked_pos: Optional[torch.BoolTensor] = None,  # [batch_size, num_patches]
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        """
        前向传播过程:
        1. 图像块嵌入
        2. 添加mask token(如果需要)
        3. 添加CLS token
        4. 添加位置编码
        5. Dropout
        """
        batch_size, num_channels, height, width = pixel_values.shape
        # [batch_size, num_patches, hidden_size]
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        # 如果提供了mask位置,替换对应位置的patch embedding为mask token
        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            # [batch_size, seq_len, hidden_size]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # [batch_size, seq_len, 1]
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            # 使用mask token替换被mask的patch
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # 添加CLS token
        # [batch_size, 1, hidden_size]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # [batch_size, 1+num_patches, hidden_size]
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # 添加位置编码
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embeddings

        # Dropout
        embeddings = self.dropout(embeddings)

        return embeddings


class ViTPatchEmbeddings(nn.Module):
    """
    图像块嵌入层:
    将输入图像 [batch_size, channels, height, width] 
    转换为序列形式的patch embeddings [batch_size一次前向传播过程中处理的图像数量, num_patches196（14 * 14）, hidden_size每个块嵌入向量的维度大小]
    """

    def __init__(self, config):
        """
        参数:
            config: 包含image_size(输入图像大小)、patch_size(图像块大小)、
                   num_channels(输入通道数)、hidden_size(嵌入维度)等参数
        """
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        # 处理输入参数,支持非方形图像
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        # 计算patch数量
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        # 使用卷积层实现patch embedding
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        """
        前向传播:
        1. 输入检查
        2. 卷积得到patch embeddings
        3. 重排为序列形式
        """
        batch_size, num_channels, height, width = pixel_values.shape
        # 检查输入通道数是否匹配
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        # 在不插值位置编码时检查输入尺寸是否匹配
        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )
        # [batch_size, hidden_size, h', w'] -> [batch_size, num_patches, hidden_size]
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,  # [batch_size, num_heads, seq_len, head_size]
    key: torch.Tensor,    # [batch_size, num_heads, seq_len, head_size] 
    value: torch.Tensor,  # [batch_size, num_heads, seq_len, head_size]
    attention_mask: Optional[torch.Tensor],  # [batch_size, num_heads, seq_len, seq_len]
    scaling: float,  # 缩放因子,通常为1/sqrt(head_size)
    dropout: float = 0.0,  # dropout概率
    **kwargs,
):
    """
    标准的注意力计算实现:
    1. Q*K^T得到注意力分数
    2. Softmax归一化
    3. Dropout
    4. 与V相乘得到输出
    """
    # [batch_size, num_heads, seq_len, seq_len]
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling

    # Softmax归一化
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    # Dropout
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    # 应用attention mask
    if attention_mask is not None:
        attn_weights = attn_weights * attention_mask

    # [batch_size, num_heads, seq_len, head_size]
    attn_output = torch.matmul(attn_weights, value)
    # [batch_size, seq_len, num_heads, head_size]
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights



class ViTSelfAttention(nn.Module):
    """
    ViT的自注意力模块实现
    包含多头注意力机制,将输入投影到Q、K、V空间,然后计算注意力
    """
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        # 检查hidden_size是否可以被注意力头数整除
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.config = config
        self.num_attention_heads = config.num_attention_heads  # 注意力头数
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)  # 每个头的维度
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # 所有头的总维度
        self.dropout_prob = config.attention_probs_dropout_prob  # 注意力dropout概率
        self.scaling = self.attention_head_size**-0.5  # 缩放因子,防止点积过大
        self.is_causal = False  # 是否使用因果注意力

        # Q、K、V的线性投影层
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        将输入张量重塑为多头格式
        输入: [batch_size, seq_len, hidden_size]
        输出: [batch_size, num_heads, seq_len, head_size]
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        """
        前向传播函数
        1. 线性投影得到Q、K、V
        2. 选择注意力实现方式(eager或sdpa)
        3. 计算注意力得分和输出
        """
        # 生成Q、K、V并转换为多头格式
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        # 选择注意力计算实现方式
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and output_attentions:
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # 计算注意力
        context_layer, attention_probs = attention_interface(
            self,
            query_layer,
            key_layer,
            value_layer,
            head_mask,
            is_causal=self.is_causal,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.dropout_prob,
        )

        # 重塑输出张量 [batch_size, seq_len, hidden_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        # 根据需要返回注意力分数
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class ViTSelfOutput(nn.Module):
    """
    The residual connection is defined in ViTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class ViTAttention(nn.Module):
    """
    ViT的注意力模块,包含自注意力和输出层。
    主要功能:
    1. 执行多头自注意力计算
    2. 处理注意力输出并添加残差连接
    3. 支持注意力头剪枝
    """
    def __init__(self, config: ViTConfig) -> None:
        """
        初始化注意力模块
        Args:
            config: ViT配置对象,包含隐藏层大小、注意力头数等超参数
        """
        super().__init__()
        # 自注意力子层,用于计算query、key、value的点积注意力
        self.attention = ViTSelfAttention(config)
        # 输出投影层,将注意力输出映射回原始维度
        self.output = ViTSelfOutput(config)
        # 记录已被剪枝的注意力头
        self.pruned_heads = set()

    def prune_heads(self, heads: Set[int]) -> None:
        """
        剪枝指定的注意力头
        Args:
            heads: 要剪枝的注意力头索引集合
        功能:
        1. 找到可剪枝的头和对应索引
        2. 剪枝线性层的权重
        3. 更新模型参数
        """
        if len(heads) == 0:
            return
        # 找到要剪枝的头和对应的权重索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # 剪枝query、key、value的线性层
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        # 剪枝输出投影层
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新注意力头数量和输出维度
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        # 记录已剪枝的头
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入张量 [batch_size, seq_len, hidden_size]
        head_mask: Optional[torch.Tensor] = None,  # 注意力头的mask
        output_attentions: bool = False,  # 是否输出注意力权重
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        """
        前向传播
        Args:
            hidden_states: 输入特征
            head_mask: 注意力头的mask矩阵
            output_attentions: 是否返回注意力分数
        Returns:
            attention_output: 注意力输出 [batch_size, seq_len, hidden_size]
            attention_weights: (可选)注意力权重
        """
        # 1. 计算自注意力,得到注意力输出和权重
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        # 2. 通过输出投影层处理注意力输出
        attention_output = self.output(self_outputs[0], hidden_states)

        # 3. 组装输出元组,包含注意力输出和可选的注意力权重
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class ViTIntermediate(nn.Module):
    """
    前馈网络
    ViT中间层,用于特征变换和非线性激活
    主要功能:
    1. 将hidden_size维度投影到更大的intermediate_size
    2. 应用非线性激活函数
    """
    def __init__(self, config: ViTConfig) -> None:
        """
        初始化中间层
        Args:
            config: 配置对象,包含hidden_size、intermediate_size和激活函数类型
        """
        super().__init__()
        # 线性投影层,扩展特征维度
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 获取激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            hidden_states: 输入特征 [batch_size, seq_len, hidden_size]
        Returns:
            transformed_features: 变换后的特征 [batch_size, seq_len, intermediate_size]
        """
        # 1. 线性变换
        hidden_states = self.dense(hidden_states)
        # 2. 非线性激活
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class ViTOutput(nn.Module):
    """
    ViT输出层,用于处理中间层输出并添加残差连接
    主要功能:
    1. 将intermediate_size维度投影回hidden_size
    2. 应用dropout
    3. 添加残差连接
    """
    def __init__(self, config: ViTConfig) -> None:
        """
        初始化输出层
        Args:
            config: 配置对象,包含维度和dropout概率
        """
        super().__init__()
        # 线性投影层,压缩特征维度
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # dropout层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            hidden_states: 中间层输出 [batch_size, seq_len, intermediate_size]
            input_tensor: 残差连接的输入 [batch_size, seq_len, hidden_size]
        Returns:
            output: 最终输出 [batch_size, seq_len, hidden_size]
        """
        # 1. 线性变换
        hidden_states = self.dense(hidden_states)
        # 2. dropout
        hidden_states = self.dropout(hidden_states)
        # 3. 残差连接
        hidden_states = hidden_states + input_tensor
        return hidden_states


class ViTLayer(nn.Module):
    """
    ViT的基本层,对应timm实现中的Block
    包含:
    1. 多头自注意力机制
    2. 前馈神经网络(FFN)
    3. 层标准化
    4. 残差连接
    """
    def __init__(self, config: ViTConfig) -> None:
        """
        初始化ViT层
        Args:
            config: 配置对象,包含各个子模块的参数
        """
        super().__init__()
        # FFN分块大小,用于优化内存使用
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        # 注意力模块
        self.attention = ViTAttention(config)
        # 前馈网络的中间层和输出层
        self.intermediate = ViTIntermediate(config)
        self.output = ViTOutput(config)
        # 两个层标准化,分别用于注意力前和FFN前
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入特征 [batch_size, seq_len, hidden_size]
        head_mask: Optional[torch.Tensor] = None,  # 注意力头mask
        output_attentions: bool = False,  # 是否输出注意力权重
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        """
        前向传播
        处理流程:
        1. 第一个层标准化  层归一化 
        2. 自注意力计算
        3. 第一个残差连接
        4. 第二个层标准化 层归一化
        5. FFN处理 前馈神经网络
        6. 第二个残差连接
        """
        # 1. 注意力模块
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # 注意力前的层标准化
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # 保存注意力权重(如果需要)

        # 2. 第一个残差连接
        hidden_states = attention_output + hidden_states

        # 3. FFN前的层标准化 层归一化
        layer_output = self.layernorm_after(hidden_states)
        # 4. FFN处理 前馈神经网络
        layer_output = self.intermediate(layer_output)
        # 5. FFN输出处理和第二个残差连接
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs
        return outputs


class ViTEncoder(nn.Module):
    """
    ViT编码器,由多个ViTLayer堆叠而成
    主要功能:
    1. 堆叠多层Transformer
    2. 支持梯度检查点以节省内存
    3. 可以输出中间层状态和注意力权重
    """
    def __init__(self, config: ViTConfig) -> None:
        """
        初始化编码器
        Args:
            config: 配置对象,包含层数等参数
        """
        super().__init__()
        self.config = config
        # 创建多层ViTLayer
        self.layer = nn.ModuleList([ViTLayer(config) for _ in range(config.num_hidden_layers)])
        # 梯度检查点标志
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入特征 [batch_size, seq_len, hidden_size]
        head_mask: Optional[torch.Tensor] = None,  # 注意力头mask
        output_attentions: bool = False,  # 是否输出注意力权重
        output_hidden_states: bool = False,  # 是否输出中间层状态
        return_dict: bool = True,  # 是否以字典形式返回
    ) -> Union[tuple, BaseModelOutput]:
        """
        前向传播
        Args:
            hidden_states: 输入特征
            head_mask: 注意力头mask
            output_attentions: 是否返回注意力权重
            output_hidden_states: 是否返回中间层状态
            return_dict: 是否返回字典格式
        Returns:
            输出特征、中间状态、注意力权重
        """
        # 初始化输出收集器
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        # 逐层处理
        for i, layer_module in enumerate(self.layer):
            # 收集中间层状态
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的head_mask
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 使用梯度检查点以节省内存
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

            # 收集注意力权重
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 收集最后一层的状态
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 返回结果
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
