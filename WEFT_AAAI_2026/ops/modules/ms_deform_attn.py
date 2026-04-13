# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import, division, print_function

import math
import warnings

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, xavier_uniform_
from mmseg_custom.models.backbones.test_2 import DepthPointwiseConv1D, DepthPointwiseConv1D_1, DepthPointwiseLinear

from ..functions import MSDeformAttnFunction


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError('invalid input for _is_power_of_2: {} (type: {})'.format(n, type(n)))
    return (n & (n - 1) == 0) and n != 0


def cosine_similarity(query, value):
    """
    计算查询向量与输入特征之间的余弦相似度，并返回与 query 相同形状的输出
    :param query: (N, Len_q, d_model)
    :param value: (N, Len_in, d_model)
    :return: (N, Len_q, d_model) 的加权特征
    """
    # 归一化查询和输入特征
    query_norm = query / (query.norm(dim=-1, keepdim=True) + 1e-8)  # 防止除零错误
    value_norm = value / (value.norm(dim=-1, keepdim=True) + 1e-8)

    # 计算余弦相似度，得到形状为 (N, Len_q, Len_in)
    cos_sim = torch.bmm(query_norm, value_norm.transpose(1, 2))  # (N, Len_q, Len_in)

    # 使用余弦相似度对输入特征进行加权平均
    # 余弦相似度是一个 (N, Len_q, Len_in) 的矩阵，表示查询和输入之间的相似度
    # 对每个查询向量，计算它与所有输入向量的加权和，形状变为 (N, Len_q, d_model)

    # 加权平均：将余弦相似度作为权重，对输入特征进行加权求和
    weighted_value = torch.bmm(cos_sim, value)  # (N, Len_q, d_model)

    return weighted_value  # 结果的形状是 (N, Len_q, d_model)


"""
class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, ratio=1.0):

        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, '
                             'but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2
        # which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn(
                "You'd better set d_model in MSDeformAttn to make "
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.ratio = ratio
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.cosine_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, int(d_model * ratio))
        self.output_proj = nn.Linear(int(d_model * ratio), d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(
            self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
                         self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1

        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes,
                input_level_start_index, input_padding_mask=None):

        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        query_cos = query
        value_cos = input_flatten

        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))

        value = value.view(N, Len_in, self.n_heads, int(self.ratio * self.d_model) // self.n_heads)

        # 计算余弦相似度 (N, Len_q, Len_in)
        #cosine_sim = cosine_similarity(query_cos, value_cos)
        #cosine_sim = self.cosine_weights(cosine_sim)

        # 计算采样偏移量 (N, Len_q, n_heads, n_levels, n_points, 2)
        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 2
        )

        # 计算注意力权重并调整形状 (N, Len_q, n_heads, n_levels * n_points)
        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points
        )

        # 应用 Softmax 对注意力权重归一化，并调整为 (N, Len_q, n_heads, n_levels, n_points)
        attention_weights = F.softmax(attention_weights, -1).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points
        )

        # 扩展 cosine_sim 形状为 (N, Len_q, n_heads, n_levels, n_points, Len_in)
        #cosine_sim = cosine_sim.view(N, Len_q, self.n_heads, self.n_levels, self.n_points)


        # 将 attention_weights 和 cosine_sim 逐元素相乘
        attention_weights = attention_weights

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'
                .format(reference_points.shape[-1]))
        output = MSDeformAttnFunction.apply(value, input_spatial_shapes, input_level_start_index,
                                            sampling_locations, attention_weights, self.im2col_step)
        output = self.output_proj(output)
        return output
"""

class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, ratio=1.0):
        """Multi-Scale Deformable Attention Module.

        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, '
                             'but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2
        # which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn(
                "You'd better set d_model in MSDeformAttn to make "
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.ratio = ratio
        #self.sampling_offsets = DepthPointwiseConv1D(d_model, n_heads * n_levels * n_points * 2)
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        #self.attention_weights = DepthPointwiseConv1D(d_model, n_heads * n_levels * n_points)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        #self.cos_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, int(d_model * ratio))
        #self.value_proj = nn.Linear(d_model, int(d_model * ratio))
        self.output_proj= nn.Linear(int(d_model * ratio), d_model)
        #self.output_proj = nn.Linear(int(d_model * ratio), d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(
            self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1

        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)

        """
        for layer in [self.value_proj, self.output_proj]:
            xavier_uniform_(layer.depthwise.weight)
            xavier_uniform_(layer.pointwise_down.weight)
            xavier_uniform_(layer.pointwise_up.weight)
        """
        #xavier_uniform_(self.value_proj.depthwise.weight)
        #constant_(self.value_proj.depthwise.bias.data, 0.)
        #xavier_uniform_(self.value_proj.pointwise_down.weight)
        #constant_(self.value_proj.pointwise_down.bias.data, 0.)
        #xavier_uniform_(self.value_proj.pointwise_up.weight)
        #constant_(self.value_proj.pointwise_up.bias.data, 0.)



        #xavier_uniform_(self.output_proj.depthwise.weight)
        #constant_(self.output_proj.depthwise.bias.data, 0.)
        #xavier_uniform_(self.output_proj.pointwise_down.weight)
        #constant_(self.output_proj.pointwise_down.bias.data, 0.)
        #xavier_uniform_(self.output_proj.pointwise_up.weight)
        #constant_(self.output_proj.pointwise_up.bias.data, 0.)





        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

        #xavier_uniform_(self.cos_weights.weight.data)  # 使用 Xavier 初始化权重
        #constant_(self.cos_weights.bias.data, 0.)  # 偏置项初始化为 0

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes,
                input_level_start_index, input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """

        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape

        #query_cos = query
        #value_cos = input_flatten

        assert (input_spatial_shapes[:, 0] *
                input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))

        value = value.view(N, Len_in, self.n_heads,
                           int(self.ratio * self.d_model) // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1). \
            view(N, Len_q, self.n_heads, self.n_levels, self.n_points)

        #cosine_sim = cosine_similarity(query_cos, value_cos)

        #cosine_sim = self.attention_weights(cosine_sim).view(
            #N, Len_q, self.n_heads, self.n_levels * self.n_points)

        #cosine_sim = F.softmax(cosine_sim, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)

        attention_weights = attention_weights

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'
                .format(reference_points.shape[-1]))
        output = MSDeformAttnFunction.apply(value, input_spatial_shapes, input_level_start_index,
                                            sampling_locations, attention_weights, self.im2col_step)
        output = self.output_proj(output)
        return output