from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from collections import OrderedDict

import copy
from typing import Optional, List
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class FFNN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, activation='gelu', drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = _get_activation_fn(activation)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    r""" Multi-head self attention module with dynamic position bias.
    Args:
        dim (int): Number of input channels.
        group_size (tuple[int]): The height and width of the group.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim,  num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 position_bias=False):


        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.position_bias = position_bias

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_groups*B, N, C)
            mask: (0/-inf) mask with shape of (num_groups, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        print('qkv.shape' , qkv.shape)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        print('q.shape',q.shape)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 group with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        if self.position_bias:
            flops += self.pos.flops(N)
        return flops




class InceptraEncoderLayer(nn.Module):
    """ Modified from https://github.com/facebookresearch/detr/blob/master/models/transformer.py"""

    def __init__(self, d_model,hidden_size, group_size, nhead, dim_feedforward=2048, 
                qkv_bias=True, qk_scale=None,
                dropout=0., attn_drop=0.1, drop_path=0.,
                 activation="relu", normalize_before=False, return_atten_map=False):
        super().__init__()
        self.dim = d_model
        self.num_heads = nhead
        self.group_size = group_size
        head_dim = dim_feedforward // nhead
        self.scale = qk_scale or head_dim ** -0.5
        self.input_resolution = hidden_size
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.attn = Attention(d_model, 
            nhead, qkv_bias=qkv_bias,qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=dropout)

        self.ffnn = FFNN(in_features=d_model, hidden_features= dim_feedforward, activation=activation, drop=dropout)

    def forward(self, x):
        """
        x = (hw, b, dim)
        """
        H, W = self.input_resolution

        x = x.permute(1,0,2) # change to crossformer format (B,L,C) --> (L,B,C)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size %d, %d, %d" % (L, H, W)

        shortcut = x
        # print('shoftcut', shortcut.shape)
        x = self.norm1(x) 
        x = x.view(B, H, W, C)
        # print('view', x.shape)
        G = self.group_size
        # group embeddings
        x = x.reshape(B, H // G, G, W // G, G, C).permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B * H * W // G**2, G**2, C)
        # print('reshape G', x.shape)
        # multi-head self-attention
        # x = self.attn(x, mask=self.attn_mask)  # nW*B, G*G, C # TODO check attention mask
        x = self.attn(x)  # nW*B, G*G, C
        # print('attention', x.shape)
        # ungroup embeddings
        x = x.reshape(B, H // G, W // G, G, G, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, C)
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        # print('before ffn', x.shape)
        x = x + self.drop_path(self.ffnn(self.norm2(x)))
        x = x.permute(1,0,2)
        return x

     
class Block(nn.Module):
    def __init__(self, num_head,hidden_size,group_size_list,  d_model, d_hidden, qkv_bias=True, qk_scale=None, 
                dropout=0.1, attn_drop=0., drop_path_rate=0. ,
                 activation="relu", normalize_before=False, return_atten_map=False ):
        super().__init__()


        self.layers = nn.ModuleList()
        num_layers = len(group_size_list)

        for gsize in group_size_list:
            self.layers.append (InceptraEncoderLayer(d_model,hidden_size, gsize, num_head, d_hidden,  qkv_bias=qkv_bias, 
                                    qk_scale=qk_scale, attn_drop=attn_drop, drop_path=drop_path_rate,
                                    dropout=dropout, activation=activation, 
                                    normalize_before=normalize_before, 
                                    return_atten_map=return_atten_map))

        self.reduce = nn.Linear(num_layers*d_model, d_model)
    
    
    def forward(self, x):
        # print('block' , x.shape)

        rlist = []
        for layer in self.layers:
            # y= layer(x)
            # print('y' ,y.shape)
            # rlist.append(y)
            rlist.append(layer(x))
        
        # print('rlist[0]', rlist[0].shape)
        x = torch.cat(rlist,2)
        # print('torch cat', x.shape)
        # print(self.reduce)
        x = self.reduce(x)
        # TODO norm ??

        return x

            
class Stage(nn.Module):
    def __init__(self, d_model, d_hidden, n_blocks, n_heads, hidden_size ,group_list,
                 qkv_bias=True, qk_scale=None, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 activation="relu", normalize_before=False, return_atten_map=False ):
        super().__init__()
    
        self.h, self.w = hidden_size
        self._make_position_embedding(d_model)
        self.blocks = nn.ModuleList()
        block = Block(n_heads, hidden_size, group_list, d_model, d_hidden,  qkv_bias=qkv_bias, 
                                    qk_scale=qk_scale, attn_drop=attn_drop_rate, drop_path_rate=drop_path_rate,
                                    dropout=drop_rate, activation=activation, 
                                    normalize_before=normalize_before, 
                                    return_atten_map=return_atten_map)

        self.blocks = _get_clones(block, n_blocks)

    def _make_position_embedding(self, d_model, pe_type='sine'):
        assert pe_type in ['none', 'learnable', 'sine']
        if pe_type == 'none':
            self.pos_embedding = None
            logger.info("==> Without any PositionEmbedding~")
        else:
            with torch.no_grad():
                length = self.h * self.w
            if pe_type == 'learnable':
                self.pos_embedding = nn.Parameter(
                    torch.randn(length, 1, d_model))
                logger.info("==> Add Learnable PositionEmbedding~")
            else:
                self.pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding(d_model, self.h, self.w),
                    requires_grad=False)
                logger.info("==> Add Sine PositionEmbedding~")

    def _make_sine_position_embedding(self, d_model, h , w, temperature=10000,
                                      scale=2*math.pi):
        # logger.info(">> NOTE: this is for testing on unseen input resolutions")
        # # NOTE generalization test with interploation
        # self.pe_h, self.pe_w = 256 // 8 , 192 // 8 #self.pe_h, self.pe_w
        area = torch.ones(1, h, w)  # [b, h, w]
        y_embed = area.cumsum(1, dtype=torch.float32)
        x_embed = area.cumsum(2, dtype=torch.float32)

        one_direction_feats = d_model // 2

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / one_direction_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = pos.flatten(2).permute(2, 0, 1)
        return pos  # [h*w, 1, d_model]

    def forward(self, x):
        x = x + self.pos_embedding
        for block in self.blocks:
            x = block(x)
            # print('block output', x.shape)

        return x



class Inceptra(nn.Module):
    def __init__(self, d_model, d_hidden, n_stage, n_blocks, n_heads, hidden_size_list,group_list_list, qkv_bias=True,
                 qk_scale=None, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 activation="relu", normalize_before=False, return_atten_map=False):
        super().__init__()

        
        self.fc1 = nn.Linear(256, 512)

        self.stages = nn.ModuleList()
        for si in range(n_stage):
            # TODO 'depths' in CrossFormer.. Check here
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(n_blocks))]  # stochastic depth decay rule

            hidden_size= hidden_size_list[si]
            group_list = group_list_list[si]
            num_blocks = n_blocks[si]
            num_head = n_heads[si]
            self.stages.append(Stage( d_model, d_hidden, num_blocks, num_head, hidden_size, group_list, qkv_bias=qkv_bias, 
                                    qk_scale=qk_scale, 
                                    drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, 
                                    drop_path_rate = drop_path_rate, # TODO check here
                                    # drop_path=dpr[sum(n_blocks[:0]):sum(n_blocks[:0 + 1])],
                                    activation=activation, normalize_before=normalize_before, 
                                    return_atten_map=return_atten_map))
     

    def forward(self,x):
        for stage in self.stages:
            x = stage(x)
        return x


    