# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import re
from collections import OrderedDict
import torch
import torch.nn as nn
from functools import partial

import timm
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from timm.models.deit import _create_deit

from timm.models import create_model
import models

import copy
from timm.models.layers import PatchEmbed
from .quantization import *


from urllib.request import urlopen
from PIL import Image


__all__ = [
    "vit_tiny_cifar10", "vit_small_cifar10", "vit_base_cifar10",
    "deit_tiny_cifar10", "deit_small_cifar10", "deit_base_cifar10",

    "vit_tiny_cifar100", "vit_small_cifar100", "vit_base_cifar100",
    "deit_tiny_cifar100", "deit_small_cifar100", "deit_base_cifar100",

    "deit_tiny_imagenet100", "deit_small_imagenet100", "deit_base_imagenet100",

    "deit_base_10blk_cifar100", "deit_base_8blk_cifar100", "deit_base_6blk_cifar100", "deit_base_4blk_cifar100", "deit_base_2blk_cifar100",
    "deit_base_11blk_cifar10", "deit_base_10blk_cifar10", "deit_base_9blk_cifar10", "deit_base_8blk_cifar10", "deit_base_7blk_cifar10", "deit_base_6blk_cifar10", "deit_base_4blk_cifar10", "deit_base_2blk_cifar10",

    "deit_tiny_cifar100_mlp", "deit_small_cifar100_mlp", "deit_base_cifar100_mlp",
    "deit_tiny_cifar100_attn", "deit_small_cifar100_attn", "deit_base_cifar100_attn",
    "deit_tiny_cifar100_patchembed", "deit_small_cifar100_patchembed", "deit_base_cifar100_patchembed",
    "deit_tiny_cifar100_head", "deit_small_cifar100_head", "deit_base_cifar100_head",

    # num heads
    "vit_tiny_cifar10_2heads",
    "deit_base_10heads_cifar10",
    ]



class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        else:
            return (x + x_dist) / 2
    



class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2


@register_model
def vit_tiny_cifar10(pretrained=True, **kwargs):

    model = timm.create_model(
                'vit_tiny_patch16_224.augreg_in21k_ft_in1k', 
                pretrained=True,
                num_classes=10
                )

    
    
    conv_layer = model.patch_embed.proj
    in_c = conv_layer.in_channels
    out_c = conv_layer.out_channels
    ker = conv_layer.kernel_size
    stri = conv_layer.stride
    pad = conv_layer.padding
    temp_quanConv2d = quan_Conv2d(  in_c,
                                    out_c,
                                    kernel_size=ker,
                                    stride = stri,
                                    padding=pad,
                                    bias=False)
    model.patch_embed.proj = temp_quanConv2d


    for module_idx, block in enumerate(model.blocks):
        
        in_f = model.blocks[module_idx].attn.qkv.in_features
        out_f = model.blocks[module_idx].attn.qkv.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.qkv = temp_quanLinear

        in_f = model.blocks[module_idx].attn.proj.in_features
        out_f = model.blocks[module_idx].attn.proj.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.proj = temp_quanLinear
    
        in_f = model.blocks[module_idx].mlp.fc1.in_features
        out_f = model.blocks[module_idx].mlp.fc1.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc1 = temp_quanLinear

        in_f = model.blocks[module_idx].mlp.fc2.in_features
        out_f = model.blocks[module_idx].mlp.fc2.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc2 = temp_quanLinear

    in_f = model.head.in_features
    out_f = model.head.out_features
    temp_quanLinear = quan_Linear(in_f, out_f)
    model.head = temp_quanLinear
   
    
    # this training is terminated in epoch 141 
    checkpoint=torch.load("/data1/Xuan_vit_ckp/2024-01-13/cifar10_vit_tiny_100_AdamW/checkpoint.pth.tar", map_location=torch.device('cpu'))
    ckp = checkpoint['state_dict']
    updated_ckp = OrderedDict((key.replace("module.", "", 1), value) for key, value in ckp.items())
    model.load_state_dict(updated_ckp)


    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    return model


@register_model
def vit_small_cifar10(pretrained=True, **kwargs):

    model = timm.create_model(
                'vit_small_patch16_224.augreg_in21k_ft_in1k', 
                pretrained=True,
                num_classes=10
                )

    
    
    conv_layer = model.patch_embed.proj
    in_c = conv_layer.in_channels
    out_c = conv_layer.out_channels
    ker = conv_layer.kernel_size
    stri = conv_layer.stride
    pad = conv_layer.padding
    temp_quanConv2d = quan_Conv2d(  in_c,
                                    out_c,
                                    kernel_size=ker,
                                    stride = stri,
                                    padding=pad,
                                    bias=False)
    model.patch_embed.proj = temp_quanConv2d


    for module_idx, block in enumerate(model.blocks):
        
        in_f = model.blocks[module_idx].attn.qkv.in_features
        out_f = model.blocks[module_idx].attn.qkv.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.qkv = temp_quanLinear

        in_f = model.blocks[module_idx].attn.proj.in_features
        out_f = model.blocks[module_idx].attn.proj.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.proj = temp_quanLinear
    
        in_f = model.blocks[module_idx].mlp.fc1.in_features
        out_f = model.blocks[module_idx].mlp.fc1.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc1 = temp_quanLinear

        in_f = model.blocks[module_idx].mlp.fc2.in_features
        out_f = model.blocks[module_idx].mlp.fc2.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc2 = temp_quanLinear

    in_f = model.head.in_features
    out_f = model.head.out_features
    temp_quanLinear = quan_Linear(in_f, out_f)
    model.head = temp_quanLinear
   
    
    # this training is terminated in epoch 141 
    checkpoint=torch.load("/data1/Xuan_vit_ckp/2024-01-14/cifar10_vit_small_300_AdamW/checkpoint.pth.tar", map_location=torch.device('cpu'))
    ckp = checkpoint['state_dict']
    updated_ckp = OrderedDict((key.replace("module.", "", 1), value) for key, value in ckp.items())
    model.load_state_dict(updated_ckp)


    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    return model


@register_model
def vit_base_cifar10(pretrained=True, **kwargs):

    model = timm.create_model(
                'vit_base_patch16_224.augreg_in21k_ft_in1k', 
                pretrained=True,
                num_classes=10
                )

    
    
    conv_layer = model.patch_embed.proj
    in_c = conv_layer.in_channels
    out_c = conv_layer.out_channels
    ker = conv_layer.kernel_size
    stri = conv_layer.stride
    pad = conv_layer.padding
    temp_quanConv2d = quan_Conv2d(  in_c,
                                    out_c,
                                    kernel_size=ker,
                                    stride = stri,
                                    padding=pad,
                                    bias=False)
    model.patch_embed.proj = temp_quanConv2d


    for module_idx, block in enumerate(model.blocks):
        
        in_f = model.blocks[module_idx].attn.qkv.in_features
        out_f = model.blocks[module_idx].attn.qkv.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.qkv = temp_quanLinear

        in_f = model.blocks[module_idx].attn.proj.in_features
        out_f = model.blocks[module_idx].attn.proj.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.proj = temp_quanLinear
    
        in_f = model.blocks[module_idx].mlp.fc1.in_features
        out_f = model.blocks[module_idx].mlp.fc1.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc1 = temp_quanLinear

        in_f = model.blocks[module_idx].mlp.fc2.in_features
        out_f = model.blocks[module_idx].mlp.fc2.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc2 = temp_quanLinear

    in_f = model.head.in_features
    out_f = model.head.out_features
    temp_quanLinear = quan_Linear(in_f, out_f)
    model.head = temp_quanLinear
   
    
    # this training is terminated in epoch 141 
    checkpoint=torch.load("/data1/Xuan_vit_ckp/2024-01-12/cifar10_vit_base_500_AdamW/checkpoint.pth.tar", map_location=torch.device('cpu'))
    ckp = checkpoint['state_dict']
    updated_ckp = OrderedDict((key.replace("module.", "", 1), value) for key, value in ckp.items())
    model.load_state_dict(updated_ckp)


    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    return model


@register_model
def deit_base_cifar10(pretrained=True, **kwargs):

    model = timm.create_model(
                'deit_base_patch16_224', 
                pretrained=True,
                num_classes=10
                )

    
    
    conv_layer = model.patch_embed.proj
    in_c = conv_layer.in_channels
    out_c = conv_layer.out_channels
    ker = conv_layer.kernel_size
    stri = conv_layer.stride
    pad = conv_layer.padding
    temp_quanConv2d = quan_Conv2d(  in_c,
                                    out_c,
                                    kernel_size=ker,
                                    stride = stri,
                                    padding=pad,
                                    bias=False)
    model.patch_embed.proj = temp_quanConv2d


    for module_idx, block in enumerate(model.blocks):
        
        in_f = model.blocks[module_idx].attn.qkv.in_features
        out_f = model.blocks[module_idx].attn.qkv.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.qkv = temp_quanLinear

        in_f = model.blocks[module_idx].attn.proj.in_features
        out_f = model.blocks[module_idx].attn.proj.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.proj = temp_quanLinear
    
        in_f = model.blocks[module_idx].mlp.fc1.in_features
        out_f = model.blocks[module_idx].mlp.fc1.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc1 = temp_quanLinear

        in_f = model.blocks[module_idx].mlp.fc2.in_features
        out_f = model.blocks[module_idx].mlp.fc2.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc2 = temp_quanLinear

    in_f = model.head.in_features
    out_f = model.head.out_features
    temp_quanLinear = quan_Linear(in_f, out_f)
    model.head = temp_quanLinear
   
    
    # this training is terminated in epoch 141 
    checkpoint=torch.load("/data1/Xuan_vit_ckp/2024-01-12/cifar10_deit_base_400_AdamW/checkpoint.pth.tar", map_location=torch.device('cpu'))
    ckp = checkpoint['state_dict']
    updated_ckp = OrderedDict((key.replace("module.", "", 1), value) for key, value in ckp.items())
    model.load_state_dict(updated_ckp)


    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    # Print the number of weights for each layer
    weights_dict = {}
    for name, parameter in model.named_parameters():
        if ('attn' in name or 'mlp' in name):
            parts = name.split('.')
            layer = '.'.join(parts[:-3]) + '.' + parts[-3]
        elif ('blocks' in name):
            parts = name.split('.')
            layer = '.'.join(parts[:-2]) + '.' + parts[-2]
        else:
            pos = name.rfind('.')
            layer = name[:pos]

        
        if (name.split('.')[-1] == 'weight' or name.split('.')[-1] == 'bias'):
            if (layer not in weights_dict.keys()):
                weights_dict[layer] = parameter.numel()
            else:
                weights_dict[layer] += parameter.numel()
    print (weights_dict)

    return model


@register_model
def deit_tiny_cifar10(pretrained=True, **kwargs):

    model = timm.create_model(
                'deit_tiny_patch16_224', 
                pretrained=True,
                num_classes=10
                )

    
    
    conv_layer = model.patch_embed.proj
    in_c = conv_layer.in_channels
    out_c = conv_layer.out_channels
    ker = conv_layer.kernel_size
    stri = conv_layer.stride
    pad = conv_layer.padding
    temp_quanConv2d = quan_Conv2d(  in_c,
                                    out_c,
                                    kernel_size=ker,
                                    stride = stri,
                                    padding=pad,
                                    bias=False)
    model.patch_embed.proj = temp_quanConv2d


    for module_idx, block in enumerate(model.blocks):
        
        in_f = model.blocks[module_idx].attn.qkv.in_features
        out_f = model.blocks[module_idx].attn.qkv.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.qkv = temp_quanLinear

        in_f = model.blocks[module_idx].attn.proj.in_features
        out_f = model.blocks[module_idx].attn.proj.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.proj = temp_quanLinear
    
        in_f = model.blocks[module_idx].mlp.fc1.in_features
        out_f = model.blocks[module_idx].mlp.fc1.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc1 = temp_quanLinear

        in_f = model.blocks[module_idx].mlp.fc2.in_features
        out_f = model.blocks[module_idx].mlp.fc2.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc2 = temp_quanLinear

    in_f = model.head.in_features
    out_f = model.head.out_features
    temp_quanLinear = quan_Linear(in_f, out_f)
    model.head = temp_quanLinear
   
    
    # this training is terminated in epoch 141 
    checkpoint=torch.load("/data1/Xuan_vit_ckp/2024-01-15/cifar10_deit_tiny_200_AdamW/checkpoint.pth.tar", map_location=torch.device('cpu'))
    ckp = checkpoint['state_dict']
    updated_ckp = OrderedDict((key.replace("module.", "", 1), value) for key, value in ckp.items())
    model.load_state_dict(updated_ckp)


    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    
    # Print the number of weights for each layer
    weights_dict = {}
    for name, parameter in model.named_parameters():
        if ('attn' in name or 'mlp' in name):
            parts = name.split('.')
            layer = '.'.join(parts[:-3]) + '.' + parts[-3]
        elif ('blocks' in name):
            parts = name.split('.')
            layer = '.'.join(parts[:-2]) + '.' + parts[-2]
        else:
            pos = name.rfind('.')
            layer = name[:pos]

        
        if (name.split('.')[-1] == 'weight' or name.split('.')[-1] == 'bias'):
            if (layer not in weights_dict.keys()):
                weights_dict[layer] = parameter.numel()
            else:
                weights_dict[layer] += parameter.numel()
    print (weights_dict)

    return model



@register_model
def deit_small_cifar10(pretrained=True, **kwargs):

    model = timm.create_model(
                'deit_small_patch16_224', 
                pretrained=True,
                num_classes=10
                )

    
    
    conv_layer = model.patch_embed.proj
    in_c = conv_layer.in_channels
    out_c = conv_layer.out_channels
    ker = conv_layer.kernel_size
    stri = conv_layer.stride
    pad = conv_layer.padding
    temp_quanConv2d = quan_Conv2d(  in_c,
                                    out_c,
                                    kernel_size=ker,
                                    stride = stri,
                                    padding=pad,
                                    bias=False)
    model.patch_embed.proj = temp_quanConv2d


    for module_idx, block in enumerate(model.blocks):
        
        in_f = model.blocks[module_idx].attn.qkv.in_features
        out_f = model.blocks[module_idx].attn.qkv.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.qkv = temp_quanLinear

        in_f = model.blocks[module_idx].attn.proj.in_features
        out_f = model.blocks[module_idx].attn.proj.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.proj = temp_quanLinear
    
        in_f = model.blocks[module_idx].mlp.fc1.in_features
        out_f = model.blocks[module_idx].mlp.fc1.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc1 = temp_quanLinear

        in_f = model.blocks[module_idx].mlp.fc2.in_features
        out_f = model.blocks[module_idx].mlp.fc2.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc2 = temp_quanLinear

    in_f = model.head.in_features
    out_f = model.head.out_features
    temp_quanLinear = quan_Linear(in_f, out_f)
    model.head = temp_quanLinear
   
    
    # this training is terminated in epoch 141 
    checkpoint=torch.load("/data1/Xuan_vit_ckp/2024-01-15/cifar10_deit_small_400_AdamW/checkpoint.pth.tar", map_location=torch.device('cpu'))
    ckp = checkpoint['state_dict']
    updated_ckp = OrderedDict((key.replace("module.", "", 1), value) for key, value in ckp.items())
    model.load_state_dict(updated_ckp)


    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    
    # Print the number of weights for each layer
    weights_dict = {}
    for name, parameter in model.named_parameters():
        if ('attn' in name or 'mlp' in name):
            parts = name.split('.')
            layer = '.'.join(parts[:-3]) + '.' + parts[-3]
        elif ('blocks' in name):
            parts = name.split('.')
            layer = '.'.join(parts[:-2]) + '.' + parts[-2]
        else:
            pos = name.rfind('.')
            layer = name[:pos]

        
        if (name.split('.')[-1] == 'weight' or name.split('.')[-1] == 'bias'):
            if (layer not in weights_dict.keys()):
                weights_dict[layer] = parameter.numel()
            else:
                weights_dict[layer] += parameter.numel()
    print (weights_dict)


    return model




@register_model
def vit_tiny_cifar100(pretrained=True, **kwargs):

    model = timm.create_model(
                'vit_tiny_patch16_224.augreg_in21k', 
                pretrained=True,
                num_classes=100
                )

    
    
    conv_layer = model.patch_embed.proj
    in_c = conv_layer.in_channels
    out_c = conv_layer.out_channels
    ker = conv_layer.kernel_size
    stri = conv_layer.stride
    pad = conv_layer.padding
    temp_quanConv2d = quan_Conv2d(  in_c,
                                    out_c,
                                    kernel_size=ker,
                                    stride = stri,
                                    padding=pad,
                                    bias=False)
    model.patch_embed.proj = temp_quanConv2d


    for module_idx, block in enumerate(model.blocks):
        
        in_f = model.blocks[module_idx].attn.qkv.in_features
        out_f = model.blocks[module_idx].attn.qkv.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.qkv = temp_quanLinear

        in_f = model.blocks[module_idx].attn.proj.in_features
        out_f = model.blocks[module_idx].attn.proj.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.proj = temp_quanLinear
    
        in_f = model.blocks[module_idx].mlp.fc1.in_features
        out_f = model.blocks[module_idx].mlp.fc1.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc1 = temp_quanLinear

        in_f = model.blocks[module_idx].mlp.fc2.in_features
        out_f = model.blocks[module_idx].mlp.fc2.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc2 = temp_quanLinear

    in_f = model.head.in_features
    out_f = model.head.out_features
    temp_quanLinear = quan_Linear(in_f, out_f)
    model.head = temp_quanLinear
   
    
    # this training is terminated in epoch 141 
    checkpoint=torch.load("/data1/Xuan_vit_ckp/2024-01-23/cifar100_vit_tiny_100_AdamW/checkpoint.pth.tar", map_location=torch.device('cpu'))
    ckp = checkpoint['state_dict']
    updated_ckp = OrderedDict((key.replace("module.", "", 1), value) for key, value in ckp.items())
    model.load_state_dict(updated_ckp)


    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    return model


@register_model
def vit_small_cifar100(pretrained=True, **kwargs):

    model = timm.create_model(
                'vit_small_patch16_224.augreg_in21k', 
                pretrained=True,
                num_classes=100
                )

    
    conv_layer = model.patch_embed.proj
    in_c = conv_layer.in_channels
    out_c = conv_layer.out_channels
    ker = conv_layer.kernel_size
    stri = conv_layer.stride
    pad = conv_layer.padding
    temp_quanConv2d = quan_Conv2d(  in_c,
                                    out_c,
                                    kernel_size=ker,
                                    stride = stri,
                                    padding=pad,
                                    bias=False)
    model.patch_embed.proj = temp_quanConv2d


    for module_idx, block in enumerate(model.blocks):
        
        in_f = model.blocks[module_idx].attn.qkv.in_features
        out_f = model.blocks[module_idx].attn.qkv.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.qkv = temp_quanLinear

        in_f = model.blocks[module_idx].attn.proj.in_features
        out_f = model.blocks[module_idx].attn.proj.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.proj = temp_quanLinear
    
        in_f = model.blocks[module_idx].mlp.fc1.in_features
        out_f = model.blocks[module_idx].mlp.fc1.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc1 = temp_quanLinear

        in_f = model.blocks[module_idx].mlp.fc2.in_features
        out_f = model.blocks[module_idx].mlp.fc2.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc2 = temp_quanLinear

    in_f = model.head.in_features
    out_f = model.head.out_features
    temp_quanLinear = quan_Linear(in_f, out_f)
    model.head = temp_quanLinear
   
    
    # this training is terminated in epoch 141 
    checkpoint=torch.load("/data1/Xuan_vit_ckp/2024-01-22/cifar100_vit_small_150_AdamW/checkpoint.pth.tar", map_location=torch.device('cpu'))
    ckp = checkpoint['state_dict']
    updated_ckp = OrderedDict((key.replace("module.", "", 1), value) for key, value in ckp.items())
    model.load_state_dict(updated_ckp)


    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    return model


@register_model
def vit_base_cifar100(pretrained=True, **kwargs):

    model = timm.create_model(
                'vit_base_patch16_224.augreg_in21k', 
                pretrained=True,
                num_classes=100
                )

    
    
    conv_layer = model.patch_embed.proj
    in_c = conv_layer.in_channels
    out_c = conv_layer.out_channels
    ker = conv_layer.kernel_size
    stri = conv_layer.stride
    pad = conv_layer.padding
    temp_quanConv2d = quan_Conv2d(  in_c,
                                    out_c,
                                    kernel_size=ker,
                                    stride = stri,
                                    padding=pad,
                                    bias=False)
    model.patch_embed.proj = temp_quanConv2d


    for module_idx, block in enumerate(model.blocks):
        
        in_f = model.blocks[module_idx].attn.qkv.in_features
        out_f = model.blocks[module_idx].attn.qkv.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.qkv = temp_quanLinear

        in_f = model.blocks[module_idx].attn.proj.in_features
        out_f = model.blocks[module_idx].attn.proj.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.proj = temp_quanLinear
    
        in_f = model.blocks[module_idx].mlp.fc1.in_features
        out_f = model.blocks[module_idx].mlp.fc1.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc1 = temp_quanLinear

        in_f = model.blocks[module_idx].mlp.fc2.in_features
        out_f = model.blocks[module_idx].mlp.fc2.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc2 = temp_quanLinear

    in_f = model.head.in_features
    out_f = model.head.out_features
    temp_quanLinear = quan_Linear(in_f, out_f)
    model.head = temp_quanLinear
   
    
    # this training is terminated in epoch 141 
    checkpoint=torch.load("/data1/Xuan_vit_ckp/2024-01-22/cifar100_vit_base_500_AdamW/checkpoint.pth.tar", map_location=torch.device('cpu'))
    ckp = checkpoint['state_dict']
    updated_ckp = OrderedDict((key.replace("module.", "", 1), value) for key, value in ckp.items())
    model.load_state_dict(updated_ckp)


    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    return model



@register_model
def deit_base_cifar100(pretrained=True, **kwargs):

    model = timm.create_model(
                'deit_base_patch16_224', 
                pretrained=True,
                num_classes=100
                )

    
    
    conv_layer = model.patch_embed.proj
    in_c = conv_layer.in_channels
    out_c = conv_layer.out_channels
    ker = conv_layer.kernel_size
    stri = conv_layer.stride
    pad = conv_layer.padding
    temp_quanConv2d = quan_Conv2d(  in_c,
                                    out_c,
                                    kernel_size=ker,
                                    stride = stri,
                                    padding=pad,
                                    bias=False)
    model.patch_embed.proj = temp_quanConv2d


    for module_idx, block in enumerate(model.blocks):
        
        in_f = model.blocks[module_idx].attn.qkv.in_features
        out_f = model.blocks[module_idx].attn.qkv.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.qkv = temp_quanLinear

        in_f = model.blocks[module_idx].attn.proj.in_features
        out_f = model.blocks[module_idx].attn.proj.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.proj = temp_quanLinear
    
        in_f = model.blocks[module_idx].mlp.fc1.in_features
        out_f = model.blocks[module_idx].mlp.fc1.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc1 = temp_quanLinear

        in_f = model.blocks[module_idx].mlp.fc2.in_features
        out_f = model.blocks[module_idx].mlp.fc2.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc2 = temp_quanLinear

    in_f = model.head.in_features
    out_f = model.head.out_features
    temp_quanLinear = quan_Linear(in_f, out_f)
    model.head = temp_quanLinear
   
    
    # this training is terminated in epoch 141 
    checkpoint=torch.load("/data1/Xuan_vit_ckp/2024-01-22/cifar100_deit_base_100_AdamW/checkpoint.pth.tar", map_location=torch.device('cpu'))
    ckp = checkpoint['state_dict']
    updated_ckp = OrderedDict((key.replace("module.", "", 1), value) for key, value in ckp.items())
    model.load_state_dict(updated_ckp)


    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    return model


@register_model
def deit_tiny_cifar100(pretrained=True, **kwargs):

    model = timm.create_model(
                'deit_tiny_patch16_224', 
                pretrained=True,
                num_classes=100
                )

    
    
    conv_layer = model.patch_embed.proj
    in_c = conv_layer.in_channels
    out_c = conv_layer.out_channels
    ker = conv_layer.kernel_size
    stri = conv_layer.stride
    pad = conv_layer.padding
    temp_quanConv2d = quan_Conv2d(  in_c,
                                    out_c,
                                    kernel_size=ker,
                                    stride = stri,
                                    padding=pad,
                                    bias=False)
    model.patch_embed.proj = temp_quanConv2d


    for module_idx, block in enumerate(model.blocks):
        
        in_f = model.blocks[module_idx].attn.qkv.in_features
        out_f = model.blocks[module_idx].attn.qkv.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.qkv = temp_quanLinear

        in_f = model.blocks[module_idx].attn.proj.in_features
        out_f = model.blocks[module_idx].attn.proj.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.proj = temp_quanLinear
    
        in_f = model.blocks[module_idx].mlp.fc1.in_features
        out_f = model.blocks[module_idx].mlp.fc1.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc1 = temp_quanLinear

        in_f = model.blocks[module_idx].mlp.fc2.in_features
        out_f = model.blocks[module_idx].mlp.fc2.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc2 = temp_quanLinear

    in_f = model.head.in_features
    out_f = model.head.out_features
    temp_quanLinear = quan_Linear(in_f, out_f)
    model.head = temp_quanLinear
   
    
    # this training is terminated in epoch 141 
    checkpoint=torch.load("/data1/Xuan_vit_ckp/2024-01-23/cifar100_deit_tiny_150_AdamW/checkpoint.pth.tar", map_location=torch.device('cpu'))
    ckp = checkpoint['state_dict']
    updated_ckp = OrderedDict((key.replace("module.", "", 1), value) for key, value in ckp.items())
    model.load_state_dict(updated_ckp)


    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    return model



@register_model
def deit_small_cifar100(pretrained=True, **kwargs):

    model = timm.create_model(
                'deit_small_patch16_224', 
                pretrained=True,
                num_classes=100
                )

    
    
    conv_layer = model.patch_embed.proj
    in_c = conv_layer.in_channels
    out_c = conv_layer.out_channels
    ker = conv_layer.kernel_size
    stri = conv_layer.stride
    pad = conv_layer.padding
    temp_quanConv2d = quan_Conv2d(  in_c,
                                    out_c,
                                    kernel_size=ker,
                                    stride = stri,
                                    padding=pad,
                                    bias=False)
    model.patch_embed.proj = temp_quanConv2d


    for module_idx, block in enumerate(model.blocks):
        
        in_f = model.blocks[module_idx].attn.qkv.in_features
        out_f = model.blocks[module_idx].attn.qkv.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.qkv = temp_quanLinear

        in_f = model.blocks[module_idx].attn.proj.in_features
        out_f = model.blocks[module_idx].attn.proj.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.proj = temp_quanLinear
    
        in_f = model.blocks[module_idx].mlp.fc1.in_features
        out_f = model.blocks[module_idx].mlp.fc1.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc1 = temp_quanLinear

        in_f = model.blocks[module_idx].mlp.fc2.in_features
        out_f = model.blocks[module_idx].mlp.fc2.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc2 = temp_quanLinear

    in_f = model.head.in_features
    out_f = model.head.out_features
    temp_quanLinear = quan_Linear(in_f, out_f)
    model.head = temp_quanLinear

   
    
    # this training is terminated in epoch 141 
    checkpoint=torch.load("/data1/Xuan_vit_ckp/2024-01-22/cifar100_deit_small_200_AdamW/checkpoint.pth.tar", map_location=torch.device('cpu'))
    ckp = checkpoint['state_dict']
    updated_ckp = OrderedDict((key.replace("module.", "", 1), value) for key, value in ckp.items())
    model.load_state_dict(updated_ckp)


    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    return model



#################################### early exit model for cifar100 ######################################

@register_model
def deit_base_10blk_cifar100(pretrained=True, **kwargs):

    model = timm.create_model(
                'deit_base_patch16_224', 
                pretrained=True,
                num_classes=100
                )

    
    
    conv_layer = model.patch_embed.proj
    in_c = conv_layer.in_channels
    out_c = conv_layer.out_channels
    ker = conv_layer.kernel_size
    stri = conv_layer.stride
    pad = conv_layer.padding
    temp_quanConv2d = quan_Conv2d(  in_c,
                                    out_c,
                                    kernel_size=ker,
                                    stride = stri,
                                    padding=pad,
                                    bias=False)
    model.patch_embed.proj = temp_quanConv2d


    for module_idx, block in enumerate(model.blocks):
        
        in_f = model.blocks[module_idx].attn.qkv.in_features
        out_f = model.blocks[module_idx].attn.qkv.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.qkv = temp_quanLinear

        in_f = model.blocks[module_idx].attn.proj.in_features
        out_f = model.blocks[module_idx].attn.proj.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.proj = temp_quanLinear
    
        in_f = model.blocks[module_idx].mlp.fc1.in_features
        out_f = model.blocks[module_idx].mlp.fc1.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc1 = temp_quanLinear

        in_f = model.blocks[module_idx].mlp.fc2.in_features
        out_f = model.blocks[module_idx].mlp.fc2.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc2 = temp_quanLinear

    in_f = model.head.in_features
    out_f = model.head.out_features
    temp_quanLinear = quan_Linear(in_f, out_f)
    model.head = temp_quanLinear
   
    # Remove blocks 8 to 11 from the 'blocks' Sequential container
    new_blocks = nn.Sequential(*list(model.blocks.children())[:10])

    # Assign the modified blocks back to the model
    model.blocks = new_blocks

    
    # this training is terminated in epoch 141 
    checkpoint=torch.load("/data1/Xuan_vit_ckp/2024-02-06/cifar100_deit_base_10blocks_cifar100_100_AdamW/checkpoint.pth.tar", map_location=torch.device('cpu'))
    ckp = checkpoint['state_dict']
    updated_ckp = OrderedDict((key.replace("module.", "", 1), value) for key, value in ckp.items())
    model.load_state_dict(updated_ckp)


    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    return model


@register_model
def deit_base_8blk_cifar100(pretrained=True, **kwargs):

    model = timm.create_model(
                'deit_base_patch16_224', 
                pretrained=True,
                num_classes=100
                )

    
    
    conv_layer = model.patch_embed.proj
    in_c = conv_layer.in_channels
    out_c = conv_layer.out_channels
    ker = conv_layer.kernel_size
    stri = conv_layer.stride
    pad = conv_layer.padding
    temp_quanConv2d = quan_Conv2d(  in_c,
                                    out_c,
                                    kernel_size=ker,
                                    stride = stri,
                                    padding=pad,
                                    bias=False)
    model.patch_embed.proj = temp_quanConv2d


    for module_idx, block in enumerate(model.blocks):
        
        in_f = model.blocks[module_idx].attn.qkv.in_features
        out_f = model.blocks[module_idx].attn.qkv.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.qkv = temp_quanLinear

        in_f = model.blocks[module_idx].attn.proj.in_features
        out_f = model.blocks[module_idx].attn.proj.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.proj = temp_quanLinear
    
        in_f = model.blocks[module_idx].mlp.fc1.in_features
        out_f = model.blocks[module_idx].mlp.fc1.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc1 = temp_quanLinear

        in_f = model.blocks[module_idx].mlp.fc2.in_features
        out_f = model.blocks[module_idx].mlp.fc2.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc2 = temp_quanLinear

    in_f = model.head.in_features
    out_f = model.head.out_features
    temp_quanLinear = quan_Linear(in_f, out_f)
    model.head = temp_quanLinear
   
    # Remove blocks 8 to 11 from the 'blocks' Sequential container
    new_blocks = nn.Sequential(*list(model.blocks.children())[:8])

    # Assign the modified blocks back to the model
    model.blocks = new_blocks

    
    # this training is terminated in epoch 141 
    checkpoint=torch.load("/data1/Xuan_vit_ckp/2024-01-27/cifar100_deit_base_8blocks_cifar100_100_AdamW/checkpoint.pth.tar", map_location=torch.device('cpu'))
    ckp = checkpoint['state_dict']
    updated_ckp = OrderedDict((key.replace("module.", "", 1), value) for key, value in ckp.items())
    model.load_state_dict(updated_ckp)


    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    return model


@register_model
def deit_base_6blk_cifar100(pretrained=True, **kwargs):

    model = timm.create_model(
                'deit_base_patch16_224', 
                pretrained=True,
                num_classes=100
                )

    
    
    conv_layer = model.patch_embed.proj
    in_c = conv_layer.in_channels
    out_c = conv_layer.out_channels
    ker = conv_layer.kernel_size
    stri = conv_layer.stride
    pad = conv_layer.padding
    temp_quanConv2d = quan_Conv2d(  in_c,
                                    out_c,
                                    kernel_size=ker,
                                    stride = stri,
                                    padding=pad,
                                    bias=False)
    model.patch_embed.proj = temp_quanConv2d


    for module_idx, block in enumerate(model.blocks):
        
        in_f = model.blocks[module_idx].attn.qkv.in_features
        out_f = model.blocks[module_idx].attn.qkv.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.qkv = temp_quanLinear

        in_f = model.blocks[module_idx].attn.proj.in_features
        out_f = model.blocks[module_idx].attn.proj.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.proj = temp_quanLinear
    
        in_f = model.blocks[module_idx].mlp.fc1.in_features
        out_f = model.blocks[module_idx].mlp.fc1.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc1 = temp_quanLinear

        in_f = model.blocks[module_idx].mlp.fc2.in_features
        out_f = model.blocks[module_idx].mlp.fc2.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc2 = temp_quanLinear

    in_f = model.head.in_features
    out_f = model.head.out_features
    temp_quanLinear = quan_Linear(in_f, out_f)
    model.head = temp_quanLinear
   
    # Remove blocks 8 to 11 from the 'blocks' Sequential container
    new_blocks = nn.Sequential(*list(model.blocks.children())[:6])

    # Assign the modified blocks back to the model
    model.blocks = new_blocks

    
    # this training is terminated in epoch 141 
    checkpoint=torch.load("/data1/Xuan_vit_ckp/2024-02-06/cifar100_deit_base_6blocks_cifar100_100_AdamW/checkpoint.pth.tar", map_location=torch.device('cpu'))
    ckp = checkpoint['state_dict']
    updated_ckp = OrderedDict((key.replace("module.", "", 1), value) for key, value in ckp.items())
    model.load_state_dict(updated_ckp)


    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    return model


@register_model
def deit_base_4blk_cifar100(pretrained=True, **kwargs):

    model = timm.create_model(
                'deit_base_patch16_224', 
                pretrained=True,
                num_classes=100
                )

    
    
    conv_layer = model.patch_embed.proj
    in_c = conv_layer.in_channels
    out_c = conv_layer.out_channels
    ker = conv_layer.kernel_size
    stri = conv_layer.stride
    pad = conv_layer.padding
    temp_quanConv2d = quan_Conv2d(  in_c,
                                    out_c,
                                    kernel_size=ker,
                                    stride = stri,
                                    padding=pad,
                                    bias=False)
    model.patch_embed.proj = temp_quanConv2d


    for module_idx, block in enumerate(model.blocks):
        
        in_f = model.blocks[module_idx].attn.qkv.in_features
        out_f = model.blocks[module_idx].attn.qkv.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.qkv = temp_quanLinear

        in_f = model.blocks[module_idx].attn.proj.in_features
        out_f = model.blocks[module_idx].attn.proj.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.proj = temp_quanLinear
    
        in_f = model.blocks[module_idx].mlp.fc1.in_features
        out_f = model.blocks[module_idx].mlp.fc1.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc1 = temp_quanLinear

        in_f = model.blocks[module_idx].mlp.fc2.in_features
        out_f = model.blocks[module_idx].mlp.fc2.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc2 = temp_quanLinear

    in_f = model.head.in_features
    out_f = model.head.out_features
    temp_quanLinear = quan_Linear(in_f, out_f)
    model.head = temp_quanLinear
   
    # Remove blocks 8 to 11 from the 'blocks' Sequential container
    new_blocks = nn.Sequential(*list(model.blocks.children())[:4])

    # Assign the modified blocks back to the model
    model.blocks = new_blocks

    
    # this training is terminated in epoch 141 
    checkpoint=torch.load("/data1/Xuan_vit_ckp/2024-01-27/cifar100_deit_base_4blocks_cifar100_100_AdamW/checkpoint.pth.tar", map_location=torch.device('cpu'))
    ckp = checkpoint['state_dict']
    updated_ckp = OrderedDict((key.replace("module.", "", 1), value) for key, value in ckp.items())
    model.load_state_dict(updated_ckp)


    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    return model


@register_model
def deit_base_2blk_cifar100(pretrained=True, **kwargs):

    model = timm.create_model(
                'deit_base_patch16_224', 
                pretrained=True,
                num_classes=100
                )

    
    
    conv_layer = model.patch_embed.proj
    in_c = conv_layer.in_channels
    out_c = conv_layer.out_channels
    ker = conv_layer.kernel_size
    stri = conv_layer.stride
    pad = conv_layer.padding
    temp_quanConv2d = quan_Conv2d(  in_c,
                                    out_c,
                                    kernel_size=ker,
                                    stride = stri,
                                    padding=pad,
                                    bias=False)
    model.patch_embed.proj = temp_quanConv2d


    for module_idx, block in enumerate(model.blocks):
        
        in_f = model.blocks[module_idx].attn.qkv.in_features
        out_f = model.blocks[module_idx].attn.qkv.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.qkv = temp_quanLinear

        in_f = model.blocks[module_idx].attn.proj.in_features
        out_f = model.blocks[module_idx].attn.proj.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.proj = temp_quanLinear
    
        in_f = model.blocks[module_idx].mlp.fc1.in_features
        out_f = model.blocks[module_idx].mlp.fc1.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc1 = temp_quanLinear

        in_f = model.blocks[module_idx].mlp.fc2.in_features
        out_f = model.blocks[module_idx].mlp.fc2.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc2 = temp_quanLinear

    in_f = model.head.in_features
    out_f = model.head.out_features
    temp_quanLinear = quan_Linear(in_f, out_f)
    model.head = temp_quanLinear
   
    # Remove blocks 8 to 11 from the 'blocks' Sequential container
    new_blocks = nn.Sequential(*list(model.blocks.children())[:2])

    # Assign the modified blocks back to the model
    model.blocks = new_blocks

    
    # this training is terminated in epoch 141 
    checkpoint=torch.load("/data1/Xuan_vit_ckp/2024-02-06/cifar100_deit_base_2blocks_cifar100_100_AdamW/checkpoint.pth.tar", map_location=torch.device('cpu'))
    ckp = checkpoint['state_dict']
    updated_ckp = OrderedDict((key.replace("module.", "", 1), value) for key, value in ckp.items())
    model.load_state_dict(updated_ckp)


    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    return model




######################### early exit models for cifar 10 #############################

@register_model
def deit_base_11blk_cifar10(pretrained=True, **kwargs):

    model = timm.create_model(
                'deit_base_patch16_224', 
                pretrained=True,
                num_classes=10
                )

    
    
    conv_layer = model.patch_embed.proj
    in_c = conv_layer.in_channels
    out_c = conv_layer.out_channels
    ker = conv_layer.kernel_size
    stri = conv_layer.stride
    pad = conv_layer.padding
    temp_quanConv2d = quan_Conv2d(  in_c,
                                    out_c,
                                    kernel_size=ker,
                                    stride = stri,
                                    padding=pad,
                                    bias=False)
    model.patch_embed.proj = temp_quanConv2d


    for module_idx, block in enumerate(model.blocks):
        
        in_f = model.blocks[module_idx].attn.qkv.in_features
        out_f = model.blocks[module_idx].attn.qkv.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.qkv = temp_quanLinear

        in_f = model.blocks[module_idx].attn.proj.in_features
        out_f = model.blocks[module_idx].attn.proj.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.proj = temp_quanLinear
    
        in_f = model.blocks[module_idx].mlp.fc1.in_features
        out_f = model.blocks[module_idx].mlp.fc1.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc1 = temp_quanLinear

        in_f = model.blocks[module_idx].mlp.fc2.in_features
        out_f = model.blocks[module_idx].mlp.fc2.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc2 = temp_quanLinear

    in_f = model.head.in_features
    out_f = model.head.out_features
    temp_quanLinear = quan_Linear(in_f, out_f)
    model.head = temp_quanLinear
   
    # Remove blocks 8 to 11 from the 'blocks' Sequential container
    new_blocks = nn.Sequential(*list(model.blocks.children())[:11])

    # Assign the modified blocks back to the model
    model.blocks = new_blocks

    
    # this training is terminated in epoch 141 
    checkpoint=torch.load("/data1/Xuan_vit_ckp/2024-02-08/cifar10_deit_base_11blocks_cifar10_150_AdamW/checkpoint.pth.tar", map_location=torch.device('cpu'))
    ckp = checkpoint['state_dict']
    updated_ckp = OrderedDict((key.replace("module.", "", 1), value) for key, value in ckp.items())
    model.load_state_dict(updated_ckp)


    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    return model



@register_model
def deit_base_10blk_cifar10(pretrained=True, **kwargs):

    model = timm.create_model(
                'deit_base_patch16_224', 
                pretrained=True,
                num_classes=10
                )

    
    conv_layer = model.patch_embed.proj
    in_c = conv_layer.in_channels
    out_c = conv_layer.out_channels
    ker = conv_layer.kernel_size
    stri = conv_layer.stride
    pad = conv_layer.padding
    temp_quanConv2d = quan_Conv2d(  in_c,
                                    out_c,
                                    kernel_size=ker,
                                    stride = stri,
                                    padding=pad,
                                    bias=False)
    model.patch_embed.proj = temp_quanConv2d


    for module_idx, block in enumerate(model.blocks):
        
        in_f = model.blocks[module_idx].attn.qkv.in_features
        out_f = model.blocks[module_idx].attn.qkv.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.qkv = temp_quanLinear

        in_f = model.blocks[module_idx].attn.proj.in_features
        out_f = model.blocks[module_idx].attn.proj.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.proj = temp_quanLinear
    
        in_f = model.blocks[module_idx].mlp.fc1.in_features
        out_f = model.blocks[module_idx].mlp.fc1.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc1 = temp_quanLinear

        in_f = model.blocks[module_idx].mlp.fc2.in_features
        out_f = model.blocks[module_idx].mlp.fc2.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc2 = temp_quanLinear

    in_f = model.head.in_features
    out_f = model.head.out_features
    temp_quanLinear = quan_Linear(in_f, out_f)
    model.head = temp_quanLinear
   
    # Remove blocks 8 to 11 from the 'blocks' Sequential container
    new_blocks = nn.Sequential(*list(model.blocks.children())[:10])

    # Assign the modified blocks back to the model
    model.blocks = new_blocks

    
    # this training is terminated in epoch 141 
    checkpoint=torch.load("/data1/Xuan_vit_ckp/2024-02-06/cifar10_deit_base_10blocks_cifar10_150_AdamW/checkpoint.pth.tar", map_location=torch.device('cpu'))
    ckp = checkpoint['state_dict']
    updated_ckp = OrderedDict((key.replace("module.", "", 1), value) for key, value in ckp.items())
    model.load_state_dict(updated_ckp)


    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    return model



@register_model
def deit_base_9blk_cifar10(pretrained=True, **kwargs):

    model = timm.create_model(
                'deit_base_patch16_224', 
                pretrained=True,
                num_classes=10
                )

    
    
    conv_layer = model.patch_embed.proj
    in_c = conv_layer.in_channels
    out_c = conv_layer.out_channels
    ker = conv_layer.kernel_size
    stri = conv_layer.stride
    pad = conv_layer.padding
    temp_quanConv2d = quan_Conv2d(  in_c,
                                    out_c,
                                    kernel_size=ker,
                                    stride = stri,
                                    padding=pad,
                                    bias=False)
    model.patch_embed.proj = temp_quanConv2d


    for module_idx, block in enumerate(model.blocks):
        
        in_f = model.blocks[module_idx].attn.qkv.in_features
        out_f = model.blocks[module_idx].attn.qkv.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.qkv = temp_quanLinear

        in_f = model.blocks[module_idx].attn.proj.in_features
        out_f = model.blocks[module_idx].attn.proj.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.proj = temp_quanLinear
    
        in_f = model.blocks[module_idx].mlp.fc1.in_features
        out_f = model.blocks[module_idx].mlp.fc1.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc1 = temp_quanLinear

        in_f = model.blocks[module_idx].mlp.fc2.in_features
        out_f = model.blocks[module_idx].mlp.fc2.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc2 = temp_quanLinear

    in_f = model.head.in_features
    out_f = model.head.out_features
    temp_quanLinear = quan_Linear(in_f, out_f)
    model.head = temp_quanLinear
   
    # Remove blocks 8 to 11 from the 'blocks' Sequential container
    new_blocks = nn.Sequential(*list(model.blocks.children())[:9])

    # Assign the modified blocks back to the model
    model.blocks = new_blocks

    
    # this training is terminated in epoch 141 
    checkpoint=torch.load("/data1/Xuan_vit_ckp/2024-02-08/cifar10_deit_base_9blocks_cifar10_150_AdamW/checkpoint.pth.tar", map_location=torch.device('cpu'))
    ckp = checkpoint['state_dict']
    updated_ckp = OrderedDict((key.replace("module.", "", 1), value) for key, value in ckp.items())
    model.load_state_dict(updated_ckp)


    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    return model



@register_model
def deit_base_8blk_cifar10(pretrained=True, **kwargs):

    model = timm.create_model(
                'deit_base_patch16_224', 
                pretrained=True,
                num_classes=10
                )

    
    
    conv_layer = model.patch_embed.proj
    in_c = conv_layer.in_channels
    out_c = conv_layer.out_channels
    ker = conv_layer.kernel_size
    stri = conv_layer.stride
    pad = conv_layer.padding
    temp_quanConv2d = quan_Conv2d(  in_c,
                                    out_c,
                                    kernel_size=ker,
                                    stride = stri,
                                    padding=pad,
                                    bias=False)
    model.patch_embed.proj = temp_quanConv2d


    for module_idx, block in enumerate(model.blocks):
        
        in_f = model.blocks[module_idx].attn.qkv.in_features
        out_f = model.blocks[module_idx].attn.qkv.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.qkv = temp_quanLinear

        in_f = model.blocks[module_idx].attn.proj.in_features
        out_f = model.blocks[module_idx].attn.proj.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.proj = temp_quanLinear
    
        in_f = model.blocks[module_idx].mlp.fc1.in_features
        out_f = model.blocks[module_idx].mlp.fc1.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc1 = temp_quanLinear

        in_f = model.blocks[module_idx].mlp.fc2.in_features
        out_f = model.blocks[module_idx].mlp.fc2.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc2 = temp_quanLinear

    in_f = model.head.in_features
    out_f = model.head.out_features
    temp_quanLinear = quan_Linear(in_f, out_f)
    model.head = temp_quanLinear
   
    # Remove blocks 8 to 11 from the 'blocks' Sequential container
    new_blocks = nn.Sequential(*list(model.blocks.children())[:8])

    # Assign the modified blocks back to the model
    model.blocks = new_blocks

    
    # this training is terminated in epoch 141 
    checkpoint=torch.load("/data1/Xuan_vit_ckp/2024-01-27/cifar10_deit_base_8blocks_cifar10_150_AdamW/checkpoint.pth.tar", map_location=torch.device('cpu'))
    ckp = checkpoint['state_dict']
    updated_ckp = OrderedDict((key.replace("module.", "", 1), value) for key, value in ckp.items())
    model.load_state_dict(updated_ckp)


    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    return model


@register_model
def deit_base_7blk_cifar10(pretrained=True, **kwargs):

    model = timm.create_model(
                'deit_base_patch16_224', 
                pretrained=True,
                num_classes=10
                )

    
    
    conv_layer = model.patch_embed.proj
    in_c = conv_layer.in_channels
    out_c = conv_layer.out_channels
    ker = conv_layer.kernel_size
    stri = conv_layer.stride
    pad = conv_layer.padding
    temp_quanConv2d = quan_Conv2d(  in_c,
                                    out_c,
                                    kernel_size=ker,
                                    stride = stri,
                                    padding=pad,
                                    bias=False)
    model.patch_embed.proj = temp_quanConv2d


    for module_idx, block in enumerate(model.blocks):
        
        in_f = model.blocks[module_idx].attn.qkv.in_features
        out_f = model.blocks[module_idx].attn.qkv.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.qkv = temp_quanLinear

        in_f = model.blocks[module_idx].attn.proj.in_features
        out_f = model.blocks[module_idx].attn.proj.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.proj = temp_quanLinear
    
        in_f = model.blocks[module_idx].mlp.fc1.in_features
        out_f = model.blocks[module_idx].mlp.fc1.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc1 = temp_quanLinear

        in_f = model.blocks[module_idx].mlp.fc2.in_features
        out_f = model.blocks[module_idx].mlp.fc2.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc2 = temp_quanLinear

    in_f = model.head.in_features
    out_f = model.head.out_features
    temp_quanLinear = quan_Linear(in_f, out_f)
    model.head = temp_quanLinear
   
    # Remove blocks 8 to 11 from the 'blocks' Sequential container
    new_blocks = nn.Sequential(*list(model.blocks.children())[:7])

    # Assign the modified blocks back to the model
    model.blocks = new_blocks

    
    # this training is terminated in epoch 141 
    checkpoint=torch.load("/data1/Xuan_vit_ckp/2024-02-08/cifar10_deit_base_7blocks_cifar10_150_AdamW/checkpoint.pth.tar", map_location=torch.device('cpu'))
    ckp = checkpoint['state_dict']
    updated_ckp = OrderedDict((key.replace("module.", "", 1), value) for key, value in ckp.items())
    model.load_state_dict(updated_ckp)


    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    return model


@register_model
def deit_base_6blk_cifar10(pretrained=True, **kwargs):

    model = timm.create_model(
                'deit_base_patch16_224', 
                pretrained=True,
                num_classes=10
                )

    
    
    conv_layer = model.patch_embed.proj
    in_c = conv_layer.in_channels
    out_c = conv_layer.out_channels
    ker = conv_layer.kernel_size
    stri = conv_layer.stride
    pad = conv_layer.padding
    temp_quanConv2d = quan_Conv2d(  in_c,
                                    out_c,
                                    kernel_size=ker,
                                    stride = stri,
                                    padding=pad,
                                    bias=False)
    model.patch_embed.proj = temp_quanConv2d


    for module_idx, block in enumerate(model.blocks):
        
        in_f = model.blocks[module_idx].attn.qkv.in_features
        out_f = model.blocks[module_idx].attn.qkv.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.qkv = temp_quanLinear

        in_f = model.blocks[module_idx].attn.proj.in_features
        out_f = model.blocks[module_idx].attn.proj.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.proj = temp_quanLinear
    
        in_f = model.blocks[module_idx].mlp.fc1.in_features
        out_f = model.blocks[module_idx].mlp.fc1.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc1 = temp_quanLinear

        in_f = model.blocks[module_idx].mlp.fc2.in_features
        out_f = model.blocks[module_idx].mlp.fc2.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc2 = temp_quanLinear

    in_f = model.head.in_features
    out_f = model.head.out_features
    temp_quanLinear = quan_Linear(in_f, out_f)
    model.head = temp_quanLinear

    # Remove blocks 8 to 11 from the 'blocks' Sequential container
    new_blocks = nn.Sequential(*list(model.blocks.children())[:6])

    # Assign the modified blocks back to the model
    model.blocks = new_blocks

   
    
    # this training is terminated in epoch 141 
    checkpoint=torch.load("/data1/Xuan_vit_ckp/2024-02-06/cifar10_deit_base_6blocks_cifar10_150_AdamW/checkpoint.pth.tar", map_location=torch.device('cpu'))
    ckp = checkpoint['state_dict']
    updated_ckp = OrderedDict((key.replace("module.", "", 1), value) for key, value in ckp.items())
    model.load_state_dict(updated_ckp)


    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    return model


@register_model
def deit_base_4blk_cifar10(pretrained=True, **kwargs):

    model = timm.create_model(
                'deit_base_patch16_224', 
                pretrained=True,
                num_classes=10
                )

    
    
    conv_layer = model.patch_embed.proj
    in_c = conv_layer.in_channels
    out_c = conv_layer.out_channels
    ker = conv_layer.kernel_size
    stri = conv_layer.stride
    pad = conv_layer.padding
    temp_quanConv2d = quan_Conv2d(  in_c,
                                    out_c,
                                    kernel_size=ker,
                                    stride = stri,
                                    padding=pad,
                                    bias=False)
    model.patch_embed.proj = temp_quanConv2d


    for module_idx, block in enumerate(model.blocks):
        
        in_f = model.blocks[module_idx].attn.qkv.in_features
        out_f = model.blocks[module_idx].attn.qkv.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.qkv = temp_quanLinear

        in_f = model.blocks[module_idx].attn.proj.in_features
        out_f = model.blocks[module_idx].attn.proj.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.proj = temp_quanLinear
    
        in_f = model.blocks[module_idx].mlp.fc1.in_features
        out_f = model.blocks[module_idx].mlp.fc1.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc1 = temp_quanLinear

        in_f = model.blocks[module_idx].mlp.fc2.in_features
        out_f = model.blocks[module_idx].mlp.fc2.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc2 = temp_quanLinear

    in_f = model.head.in_features
    out_f = model.head.out_features
    temp_quanLinear = quan_Linear(in_f, out_f)
    model.head = temp_quanLinear

    # Remove blocks 8 to 11 from the 'blocks' Sequential container
    new_blocks = nn.Sequential(*list(model.blocks.children())[:4])

    # Assign the modified blocks back to the model
    model.blocks = new_blocks

   
    
    # this training is terminated in epoch 141 
    checkpoint=torch.load("/data1/Xuan_vit_ckp/2024-01-27/cifar10_deit_base_4blocks_cifar10_150_AdamW/checkpoint.pth.tar", map_location=torch.device('cpu'))
    ckp = checkpoint['state_dict']
    updated_ckp = OrderedDict((key.replace("module.", "", 1), value) for key, value in ckp.items())
    model.load_state_dict(updated_ckp)


    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    return model


@register_model
def deit_base_2blk_cifar10(pretrained=True, **kwargs):

    model = timm.create_model(
                'deit_base_patch16_224', 
                pretrained=True,
                num_classes=10
                )

    
    
    conv_layer = model.patch_embed.proj
    in_c = conv_layer.in_channels
    out_c = conv_layer.out_channels
    ker = conv_layer.kernel_size
    stri = conv_layer.stride
    pad = conv_layer.padding
    temp_quanConv2d = quan_Conv2d(  in_c,
                                    out_c,
                                    kernel_size=ker,
                                    stride = stri,
                                    padding=pad,
                                    bias=False)
    model.patch_embed.proj = temp_quanConv2d


    for module_idx, block in enumerate(model.blocks):
        
        in_f = model.blocks[module_idx].attn.qkv.in_features
        out_f = model.blocks[module_idx].attn.qkv.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.qkv = temp_quanLinear

        in_f = model.blocks[module_idx].attn.proj.in_features
        out_f = model.blocks[module_idx].attn.proj.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.proj = temp_quanLinear
    
        in_f = model.blocks[module_idx].mlp.fc1.in_features
        out_f = model.blocks[module_idx].mlp.fc1.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc1 = temp_quanLinear

        in_f = model.blocks[module_idx].mlp.fc2.in_features
        out_f = model.blocks[module_idx].mlp.fc2.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc2 = temp_quanLinear

    in_f = model.head.in_features
    out_f = model.head.out_features
    temp_quanLinear = quan_Linear(in_f, out_f)
    model.head = temp_quanLinear

    # Remove blocks 8 to 11 from the 'blocks' Sequential container
    new_blocks = nn.Sequential(*list(model.blocks.children())[:2])

    # Assign the modified blocks back to the model
    model.blocks = new_blocks

   
    
    # this training is terminated in epoch 141 
    checkpoint=torch.load("/data1/Xuan_vit_ckp/2024-02-06/cifar10_deit_base_2blocks_cifar10_150_AdamW/checkpoint.pth.tar", map_location=torch.device('cpu'))
    ckp = checkpoint['state_dict']
    updated_ckp = OrderedDict((key.replace("module.", "", 1), value) for key, value in ckp.items())
    model.load_state_dict(updated_ckp)


    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    return model




################################## imagenet-100 #################################
@register_model
def deit_base_imagenet100(pretrained=True, **kwargs):

    model = timm.create_model(
                'deit_base_patch16_224', 
                pretrained=True,
                num_classes=100
                )

    
    
    conv_layer = model.patch_embed.proj
    in_c = conv_layer.in_channels
    out_c = conv_layer.out_channels
    ker = conv_layer.kernel_size
    stri = conv_layer.stride
    pad = conv_layer.padding
    temp_quanConv2d = quan_Conv2d(  in_c,
                                    out_c,
                                    kernel_size=ker,
                                    stride = stri,
                                    padding=pad,
                                    bias=False)
    model.patch_embed.proj = temp_quanConv2d


    for module_idx, block in enumerate(model.blocks):
        
        in_f = model.blocks[module_idx].attn.qkv.in_features
        out_f = model.blocks[module_idx].attn.qkv.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.qkv = temp_quanLinear

        in_f = model.blocks[module_idx].attn.proj.in_features
        out_f = model.blocks[module_idx].attn.proj.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.proj = temp_quanLinear
    
        in_f = model.blocks[module_idx].mlp.fc1.in_features
        out_f = model.blocks[module_idx].mlp.fc1.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc1 = temp_quanLinear

        in_f = model.blocks[module_idx].mlp.fc2.in_features
        out_f = model.blocks[module_idx].mlp.fc2.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc2 = temp_quanLinear

    in_f = model.head.in_features
    out_f = model.head.out_features
    temp_quanLinear = quan_Linear(in_f, out_f)
    model.head = temp_quanLinear
   
    
    # this training is terminated in epoch 141 
    checkpoint=torch.load("/data1/Xuan_vit_ckp/2024-02-06/imagenet-100_deit_base_cifar100_250_AdamW/checkpoint.pth.tar", map_location=torch.device('cpu'))
    ckp = checkpoint['state_dict']
    updated_ckp = OrderedDict((key.replace("module.", "", 1), value) for key, value in ckp.items())
    model.load_state_dict(updated_ckp)


    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    return model


@register_model
def deit_tiny_imagenet100(pretrained=True, **kwargs):

    model = timm.create_model(
                'deit_tiny_patch16_224', 
                pretrained=True,
                num_classes=100
                )

    
    
    conv_layer = model.patch_embed.proj
    in_c = conv_layer.in_channels
    out_c = conv_layer.out_channels
    ker = conv_layer.kernel_size
    stri = conv_layer.stride
    pad = conv_layer.padding
    temp_quanConv2d = quan_Conv2d(  in_c,
                                    out_c,
                                    kernel_size=ker,
                                    stride = stri,
                                    padding=pad,
                                    bias=False)
    model.patch_embed.proj = temp_quanConv2d


    for module_idx, block in enumerate(model.blocks):
        
        in_f = model.blocks[module_idx].attn.qkv.in_features
        out_f = model.blocks[module_idx].attn.qkv.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.qkv = temp_quanLinear

        in_f = model.blocks[module_idx].attn.proj.in_features
        out_f = model.blocks[module_idx].attn.proj.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.proj = temp_quanLinear
    
        in_f = model.blocks[module_idx].mlp.fc1.in_features
        out_f = model.blocks[module_idx].mlp.fc1.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc1 = temp_quanLinear

        in_f = model.blocks[module_idx].mlp.fc2.in_features
        out_f = model.blocks[module_idx].mlp.fc2.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc2 = temp_quanLinear

    in_f = model.head.in_features
    out_f = model.head.out_features
    temp_quanLinear = quan_Linear(in_f, out_f)
    model.head = temp_quanLinear
   
    
    # this training is terminated in epoch 141 
    checkpoint=torch.load("/data1/Xuan_vit_ckp/2024-02-06/imagenet-100_deit_tiny_cifar100_150_AdamW/checkpoint.pth.tar", map_location=torch.device('cpu'))
    ckp = checkpoint['state_dict']
    updated_ckp = OrderedDict((key.replace("module.", "", 1), value) for key, value in ckp.items())
    model.load_state_dict(updated_ckp)


    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    return model



@register_model
def deit_small_imagenet100(pretrained=True, **kwargs):

    model = timm.create_model(
                'deit_small_patch16_224', 
                pretrained=True,
                num_classes=100
                )

    
    
    conv_layer = model.patch_embed.proj
    in_c = conv_layer.in_channels
    out_c = conv_layer.out_channels
    ker = conv_layer.kernel_size
    stri = conv_layer.stride
    pad = conv_layer.padding
    temp_quanConv2d = quan_Conv2d(  in_c,
                                    out_c,
                                    kernel_size=ker,
                                    stride = stri,
                                    padding=pad,
                                    bias=False)
    model.patch_embed.proj = temp_quanConv2d


    for module_idx, block in enumerate(model.blocks):
        
        in_f = model.blocks[module_idx].attn.qkv.in_features
        out_f = model.blocks[module_idx].attn.qkv.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.qkv = temp_quanLinear

        in_f = model.blocks[module_idx].attn.proj.in_features
        out_f = model.blocks[module_idx].attn.proj.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.proj = temp_quanLinear
    
        in_f = model.blocks[module_idx].mlp.fc1.in_features
        out_f = model.blocks[module_idx].mlp.fc1.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc1 = temp_quanLinear

        in_f = model.blocks[module_idx].mlp.fc2.in_features
        out_f = model.blocks[module_idx].mlp.fc2.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc2 = temp_quanLinear

    in_f = model.head.in_features
    out_f = model.head.out_features
    temp_quanLinear = quan_Linear(in_f, out_f)
    model.head = temp_quanLinear

   
    
    # this training is terminated in epoch 141 
    checkpoint=torch.load("/data1/Xuan_vit_ckp/2024-02-06/imagenet-100_deit_small_cifar100_250_AdamW/checkpoint.pth.tar", map_location=torch.device('cpu'))
    ckp = checkpoint['state_dict']
    updated_ckp = OrderedDict((key.replace("module.", "", 1), value) for key, value in ckp.items())
    model.load_state_dict(updated_ckp)


    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    return model


@register_model
def vit_tiny_imagenet100(pretrained=True, **kwargs):

    model = timm.create_model(
                'vit_tiny_patch16_224.augreg_in21k_ft_in1k', 
                pretrained=True,
                num_classes=100
                )

    
    
    conv_layer = model.patch_embed.proj
    in_c = conv_layer.in_channels
    out_c = conv_layer.out_channels
    ker = conv_layer.kernel_size
    stri = conv_layer.stride
    pad = conv_layer.padding
    temp_quanConv2d = quan_Conv2d(  in_c,
                                    out_c,
                                    kernel_size=ker,
                                    stride = stri,
                                    padding=pad,
                                    bias=False)
    model.patch_embed.proj = temp_quanConv2d


    for module_idx, block in enumerate(model.blocks):
        
        in_f = model.blocks[module_idx].attn.qkv.in_features
        out_f = model.blocks[module_idx].attn.qkv.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.qkv = temp_quanLinear

        in_f = model.blocks[module_idx].attn.proj.in_features
        out_f = model.blocks[module_idx].attn.proj.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.proj = temp_quanLinear
    
        in_f = model.blocks[module_idx].mlp.fc1.in_features
        out_f = model.blocks[module_idx].mlp.fc1.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc1 = temp_quanLinear

        in_f = model.blocks[module_idx].mlp.fc2.in_features
        out_f = model.blocks[module_idx].mlp.fc2.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc2 = temp_quanLinear

    in_f = model.head.in_features
    out_f = model.head.out_features
    temp_quanLinear = quan_Linear(in_f, out_f)
    model.head = temp_quanLinear
   
    
    # this training is terminated in epoch 141 
    checkpoint=torch.load("/data1/Xuan_vit_ckp/2024-02-08/imagenet-100_vit_tiny_cifar100_100_AdamW/checkpoint.pth.tar", map_location=torch.device('cpu'))
    ckp = checkpoint['state_dict']
    updated_ckp = OrderedDict((key.replace("module.", "", 1), value) for key, value in ckp.items())
    model.load_state_dict(updated_ckp)


    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    return model


@register_model
def vit_small_imagenet100(pretrained=True, **kwargs):

    model = timm.create_model(
                'vit_small_patch16_224.augreg_in21k_ft_in1k', 
                pretrained=True,
                num_classes=100
                )

    
    
    conv_layer = model.patch_embed.proj
    in_c = conv_layer.in_channels
    out_c = conv_layer.out_channels
    ker = conv_layer.kernel_size
    stri = conv_layer.stride
    pad = conv_layer.padding
    temp_quanConv2d = quan_Conv2d(  in_c,
                                    out_c,
                                    kernel_size=ker,
                                    stride = stri,
                                    padding=pad,
                                    bias=False)
    model.patch_embed.proj = temp_quanConv2d


    for module_idx, block in enumerate(model.blocks):
        
        in_f = model.blocks[module_idx].attn.qkv.in_features
        out_f = model.blocks[module_idx].attn.qkv.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.qkv = temp_quanLinear

        in_f = model.blocks[module_idx].attn.proj.in_features
        out_f = model.blocks[module_idx].attn.proj.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.proj = temp_quanLinear
    
        in_f = model.blocks[module_idx].mlp.fc1.in_features
        out_f = model.blocks[module_idx].mlp.fc1.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc1 = temp_quanLinear

        in_f = model.blocks[module_idx].mlp.fc2.in_features
        out_f = model.blocks[module_idx].mlp.fc2.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc2 = temp_quanLinear

    in_f = model.head.in_features
    out_f = model.head.out_features
    temp_quanLinear = quan_Linear(in_f, out_f)
    model.head = temp_quanLinear
   
    
    # this training is terminated in epoch 141 
    checkpoint=torch.load("/data1/Xuan_vit_ckp/2024-02-07/imagenet-100_vit_small_cifar100_150_AdamW/checkpoint.pth.tar", map_location=torch.device('cpu'))
    ckp = checkpoint['state_dict']
    updated_ckp = OrderedDict((key.replace("module.", "", 1), value) for key, value in ckp.items())
    model.load_state_dict(updated_ckp)


    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    return model


@register_model
def vit_base_imagenet100(pretrained=True, **kwargs):

    model = timm.create_model(
                'vit_base_patch16_224.augreg_in21k_ft_in1k', 
                pretrained=True,
                num_classes=100
                )

    
    
    conv_layer = model.patch_embed.proj
    in_c = conv_layer.in_channels
    out_c = conv_layer.out_channels
    ker = conv_layer.kernel_size
    stri = conv_layer.stride
    pad = conv_layer.padding
    temp_quanConv2d = quan_Conv2d(  in_c,
                                    out_c,
                                    kernel_size=ker,
                                    stride = stri,
                                    padding=pad,
                                    bias=False)
    model.patch_embed.proj = temp_quanConv2d


    for module_idx, block in enumerate(model.blocks):
        
        in_f = model.blocks[module_idx].attn.qkv.in_features
        out_f = model.blocks[module_idx].attn.qkv.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.qkv = temp_quanLinear

        in_f = model.blocks[module_idx].attn.proj.in_features
        out_f = model.blocks[module_idx].attn.proj.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.proj = temp_quanLinear
    
        in_f = model.blocks[module_idx].mlp.fc1.in_features
        out_f = model.blocks[module_idx].mlp.fc1.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc1 = temp_quanLinear

        in_f = model.blocks[module_idx].mlp.fc2.in_features
        out_f = model.blocks[module_idx].mlp.fc2.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc2 = temp_quanLinear

    in_f = model.head.in_features
    out_f = model.head.out_features
    temp_quanLinear = quan_Linear(in_f, out_f)
    model.head = temp_quanLinear
   
    
    # this training is terminated in epoch 141 
    checkpoint=torch.load("/data1/Xuan_vit_ckp/2024-02-07/imagenet-100_vit_base_cifar100_200_AdamW/checkpoint.pth.tar", map_location=torch.device('cpu'))
    ckp = checkpoint['state_dict']
    updated_ckp = OrderedDict((key.replace("module.", "", 1), value) for key, value in ckp.items())
    model.load_state_dict(updated_ckp)


    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    return model


######################### num heads ############################
@register_model
def vit_tiny_cifar10_2heads(pretrained=True, **kwargs):

    model = timm.create_model(
                'vit_tiny_patch16_224.augreg_in21k_ft_in1k', 
                pretrained=True,
                num_classes=10,
                num_attention_heads=2,
                )

    
    
    conv_layer = model.patch_embed.proj
    in_c = conv_layer.in_channels
    out_c = conv_layer.out_channels
    ker = conv_layer.kernel_size
    stri = conv_layer.stride
    pad = conv_layer.padding
    temp_quanConv2d = quan_Conv2d(  in_c,
                                    out_c,
                                    kernel_size=ker,
                                    stride = stri,
                                    padding=pad,
                                    bias=False)
    model.patch_embed.proj = temp_quanConv2d


    for module_idx, block in enumerate(model.blocks):
        
        in_f = model.blocks[module_idx].attn.qkv.in_features
        out_f = model.blocks[module_idx].attn.qkv.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.qkv = temp_quanLinear

        in_f = model.blocks[module_idx].attn.proj.in_features
        out_f = model.blocks[module_idx].attn.proj.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.proj = temp_quanLinear
    
        in_f = model.blocks[module_idx].mlp.fc1.in_features
        out_f = model.blocks[module_idx].mlp.fc1.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc1 = temp_quanLinear

        in_f = model.blocks[module_idx].mlp.fc2.in_features
        out_f = model.blocks[module_idx].mlp.fc2.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc2 = temp_quanLinear

    in_f = model.head.in_features
    out_f = model.head.out_features
    temp_quanLinear = quan_Linear(in_f, out_f)
    model.head = temp_quanLinear
   
    
    # this training is terminated in epoch 141 
    checkpoint=torch.load("/data1/Xuan_vit_ckp/2024-01-13/cifar10_vit_tiny_100_AdamW/checkpoint.pth.tar", map_location=torch.device('cpu'))
    ckp = checkpoint['state_dict']
    updated_ckp = OrderedDict((key.replace("module.", "", 1), value) for key, value in ckp.items())
    model.load_state_dict(updated_ckp)


    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    return model

################# number of heads experiments #####################
@register_model
def deit_base_10heads_cifar10(pretrained=True, **kwargs):
    num_heads = 10

    model = timm.create_model(
                'deit_base_patch16_224', 
                pretrained=True,
                num_classes=10,
                )

    
    
    conv_layer = model.patch_embed.proj
    in_c = conv_layer.in_channels
    out_c = conv_layer.out_channels
    ker = conv_layer.kernel_size
    stri = conv_layer.stride
    pad = conv_layer.padding
    temp_quanConv2d = quan_Conv2d(  in_c,
                                    out_c,
                                    kernel_size=ker,
                                    stride = stri,
                                    padding=pad,
                                    bias=False)
    model.patch_embed.proj = temp_quanConv2d


    for module_idx, block in enumerate(model.blocks):
        
        in_f = model.blocks[module_idx].attn.qkv.in_features
        out_f = model.blocks[module_idx].attn.qkv.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.qkv = temp_quanLinear

        in_f = model.blocks[module_idx].attn.proj.in_features
        out_f = model.blocks[module_idx].attn.proj.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.proj = temp_quanLinear
    
        in_f = model.blocks[module_idx].mlp.fc1.in_features
        out_f = model.blocks[module_idx].mlp.fc1.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc1 = temp_quanLinear

        in_f = model.blocks[module_idx].mlp.fc2.in_features
        out_f = model.blocks[module_idx].mlp.fc2.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc2 = temp_quanLinear

    in_f = model.head.in_features
    out_f = model.head.out_features
    temp_quanLinear = quan_Linear(in_f, out_f)
    model.head = temp_quanLinear
   
    
    # this training is terminated in epoch 141 
    checkpoint=torch.load("/data1/Xuan_vit_ckp/2024-01-12/cifar10_deit_base_400_AdamW/checkpoint.pth.tar", map_location=torch.device('cpu'))
    ckp = checkpoint['state_dict']
    updated_ckp = OrderedDict((key.replace("module.", "", 1), value) for key, value in ckp.items())
    model.load_state_dict(updated_ckp)


    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    
    for block in model.blocks:
        # Assuming each block has an `attn` attribute with a `qkv` linear layer
        # and that the original in_features is 768 (as per your model definition)
        qkv_out = 192*num_heads  # New out_features calculated for 10 heads
        proj_in = 64*num_heads
    
        # Update the `qkv` layer of each block
        #block.attn.qkv = nn.Linear(in_features=block.attn.qkv.in_features, out_features=qkv_out, bias=True)
        block.attn.qkv = nn.Linear(in_features=block.attn.qkv.in_features, out_features=qkv_out, bias=True)
        block.attn.proj = nn.Linear(in_features=proj_in, out_features=block.attn.proj.out_features, bias=True)


    return model









# =========================== only attack mlp ===============================
@register_model
def deit_base_cifar100_mlp(pretrained=True, **kwargs):

    model = timm.create_model(
                'deit_base_patch16_224', 
                pretrained=True,
                num_classes=100
                )

    
    
    conv_layer = model.patch_embed.proj
    in_c = conv_layer.in_channels
    out_c = conv_layer.out_channels
    ker = conv_layer.kernel_size
    stri = conv_layer.stride
    pad = conv_layer.padding
    temp_quanConv2d = quan_Conv2d(  in_c,
                                    out_c,
                                    kernel_size=ker,
                                    stride = stri,
                                    padding=pad,
                                    bias=False)
    model.patch_embed.proj = temp_quanConv2d


    # print (f"enu = {enumerate(model.blocks)}")
    for module_idx, block in enumerate(model.blocks):
        # print (f"module_idx={module_idx}")
        
        in_f = model.blocks[module_idx].mlp.fc1.in_features
        out_f = model.blocks[module_idx].mlp.fc1.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc1 = temp_quanLinear

        in_f = model.blocks[module_idx].mlp.fc2.in_features
        out_f = model.blocks[module_idx].mlp.fc2.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc2 = temp_quanLinear

    in_f = model.head.in_features
    out_f = model.head.out_features
    temp_quanLinear = quan_Linear(in_f, out_f)
    model.head = temp_quanLinear
   
    print (model)
    
    # this training is terminated in epoch 141 
    checkpoint=torch.load("/data1/Xuan_vit_ckp/2024-01-22/cifar100_deit_base_100_AdamW/checkpoint.pth.tar", map_location=torch.device('cpu'))
    print (f"ckp dict key = {checkpoint.keys()}")
    #print (f"ckp dict = {checkpoint}")
    ckp = checkpoint['state_dict']
    updated_ckp = OrderedDict((key.replace("module.", "", 1), value) for key, value in ckp.items())


    # Compile the regular expression patterns
    pattern1 = re.compile(r"blocks\.\d+\.attn\.\w+\.step_size")
    pattern2 = re.compile(r"blocks\.\d+\.attn\.\w+\.b_w")

    # Find keys to delete
    keys_to_delete = [key for key in updated_ckp.keys() if pattern1.match(key) or pattern2.match(key)]

    # Delete the keys
    for key in keys_to_delete:
        del updated_ckp[key]


    model.load_state_dict(updated_ckp)


    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    return model


@register_model
def deit_tiny_cifar100_mlp(pretrained=True, **kwargs):

    model = timm.create_model(
                'deit_tiny_patch16_224', 
                pretrained=True,
                num_classes=100
                )

    
    
    conv_layer = model.patch_embed.proj
    in_c = conv_layer.in_channels
    out_c = conv_layer.out_channels
    ker = conv_layer.kernel_size
    stri = conv_layer.stride
    pad = conv_layer.padding
    temp_quanConv2d = quan_Conv2d(  in_c,
                                    out_c,
                                    kernel_size=ker,
                                    stride = stri,
                                    padding=pad,
                                    bias=False)
    model.patch_embed.proj = temp_quanConv2d


    # print (f"enu = {enumerate(model.blocks)}")
    for module_idx, block in enumerate(model.blocks):
        # print (f"module_idx={module_idx}")
        
        in_f = model.blocks[module_idx].mlp.fc1.in_features
        out_f = model.blocks[module_idx].mlp.fc1.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc1 = temp_quanLinear

        in_f = model.blocks[module_idx].mlp.fc2.in_features
        out_f = model.blocks[module_idx].mlp.fc2.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc2 = temp_quanLinear

    in_f = model.head.in_features
    out_f = model.head.out_features
    temp_quanLinear = quan_Linear(in_f, out_f)
    model.head = temp_quanLinear
   
    
    # this training is terminated in epoch 141 
    checkpoint=torch.load("/data1/Xuan_vit_ckp/2024-01-23/cifar100_deit_tiny_150_AdamW/checkpoint.pth.tar", map_location=torch.device('cpu'))
    ckp = checkpoint['state_dict']
    updated_ckp = OrderedDict((key.replace("module.", "", 1), value) for key, value in ckp.items())



    # Compile the regular expression patterns
    pattern1 = re.compile(r"blocks\.\d+\.attn\.\w+\.step_size")
    pattern2 = re.compile(r"blocks\.\d+\.attn\.\w+\.b_w")

    # Find keys to delete
    keys_to_delete = [key for key in updated_ckp.keys() if pattern1.match(key) or pattern2.match(key)]

    # Delete the keys
    for key in keys_to_delete:
        del updated_ckp[key]

    model.load_state_dict(updated_ckp)


    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    return model



@register_model
def deit_small_cifar100_mlp(pretrained=True, **kwargs):

    model = timm.create_model(
                'deit_small_patch16_224', 
                pretrained=True,
                num_classes=100
                )

    
    
    conv_layer = model.patch_embed.proj
    in_c = conv_layer.in_channels
    out_c = conv_layer.out_channels
    ker = conv_layer.kernel_size
    stri = conv_layer.stride
    pad = conv_layer.padding
    temp_quanConv2d = quan_Conv2d(  in_c,
                                    out_c,
                                    kernel_size=ker,
                                    stride = stri,
                                    padding=pad,
                                    bias=False)
    model.patch_embed.proj = temp_quanConv2d


    # print (f"enu = {enumerate(model.blocks)}")
    for module_idx, block in enumerate(model.blocks):
        # print (f"module_idx={module_idx}")
        
        in_f = model.blocks[module_idx].mlp.fc1.in_features
        out_f = model.blocks[module_idx].mlp.fc1.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc1 = temp_quanLinear

        in_f = model.blocks[module_idx].mlp.fc2.in_features
        out_f = model.blocks[module_idx].mlp.fc2.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].mlp.fc2 = temp_quanLinear

    in_f = model.head.in_features
    out_f = model.head.out_features
    temp_quanLinear = quan_Linear(in_f, out_f)
    model.head = temp_quanLinear

   
    
    # this training is terminated in epoch 141 
    checkpoint=torch.load("/data1/Xuan_vit_ckp/2024-01-22/cifar100_deit_small_200_AdamW/checkpoint.pth.tar", map_location=torch.device('cpu'))
    ckp = checkpoint['state_dict']
    updated_ckp = OrderedDict((key.replace("module.", "", 1), value) for key, value in ckp.items())


    # Compile the regular expression patterns
    pattern1 = re.compile(r"blocks\.\d+\.attn\.\w+\.step_size")
    pattern2 = re.compile(r"blocks\.\d+\.attn\.\w+\.b_w")

    # Find keys to delete
    keys_to_delete = [key for key in updated_ckp.keys() if pattern1.match(key) or pattern2.match(key)]

    # Delete the keys
    for key in keys_to_delete:
        del updated_ckp[key]


    model.load_state_dict(updated_ckp)


    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    return model


# =========================== only attack attention ===============================
@register_model
def deit_base_cifar100_attn(pretrained=True, **kwargs):

    model = timm.create_model(
                'deit_base_patch16_224', 
                pretrained=True,
                num_classes=100
                )

    
    
    conv_layer = model.patch_embed.proj
    in_c = conv_layer.in_channels
    out_c = conv_layer.out_channels
    ker = conv_layer.kernel_size
    stri = conv_layer.stride
    pad = conv_layer.padding
    temp_quanConv2d = quan_Conv2d(  in_c,
                                    out_c,
                                    kernel_size=ker,
                                    stride = stri,
                                    padding=pad,
                                    bias=False)
    model.patch_embed.proj = temp_quanConv2d


    # print (f"enu = {enumerate(model.blocks)}")
    for module_idx, block in enumerate(model.blocks):
        # print (f"module_idx={module_idx}")
        
        in_f = model.blocks[module_idx].attn.qkv.in_features
        out_f = model.blocks[module_idx].attn.qkv.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.qkv = temp_quanLinear

        in_f = model.blocks[module_idx].attn.proj.in_features
        out_f = model.blocks[module_idx].attn.proj.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.proj = temp_quanLinear
    
    in_f = model.head.in_features
    out_f = model.head.out_features
    temp_quanLinear = quan_Linear(in_f, out_f)
    model.head = temp_quanLinear
   
    
    # this training is terminated in epoch 141 
    checkpoint=torch.load("/data1/Xuan_vit_ckp/2024-01-22/cifar100_deit_base_100_AdamW/checkpoint.pth.tar", map_location=torch.device('cpu'))
    ckp = checkpoint['state_dict']
    updated_ckp = OrderedDict((key.replace("module.", "", 1), value) for key, value in ckp.items())


    # Compile the regular expression patterns
    pattern1 = re.compile(r"blocks\.\d+\.mlp\.\w+\.step_size")
    pattern2 = re.compile(r"blocks\.\d+\.mlp\.\w+\.b_w")

    # Find keys to delete
    keys_to_delete = [key for key in updated_ckp.keys() if pattern1.match(key) or pattern2.match(key)]

    # Delete the keys
    for key in keys_to_delete:
        del updated_ckp[key]


    model.load_state_dict(updated_ckp)


    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    return model


@register_model
def deit_tiny_cifar100_attn(pretrained=True, **kwargs):

    model = timm.create_model(
                'deit_tiny_patch16_224', 
                pretrained=True,
                num_classes=100
                )

    
    
    conv_layer = model.patch_embed.proj
    in_c = conv_layer.in_channels
    out_c = conv_layer.out_channels
    ker = conv_layer.kernel_size
    stri = conv_layer.stride
    pad = conv_layer.padding
    temp_quanConv2d = quan_Conv2d(  in_c,
                                    out_c,
                                    kernel_size=ker,
                                    stride = stri,
                                    padding=pad,
                                    bias=False)
    model.patch_embed.proj = temp_quanConv2d


    # print (f"enu = {enumerate(model.blocks)}")
    for module_idx, block in enumerate(model.blocks):
        # print (f"module_idx={module_idx}")
        
        in_f = model.blocks[module_idx].attn.qkv.in_features
        out_f = model.blocks[module_idx].attn.qkv.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.qkv = temp_quanLinear

        in_f = model.blocks[module_idx].attn.proj.in_features
        out_f = model.blocks[module_idx].attn.proj.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.proj = temp_quanLinear
    
    in_f = model.head.in_features
    out_f = model.head.out_features
    temp_quanLinear = quan_Linear(in_f, out_f)
    model.head = temp_quanLinear
   
    
    # this training is terminated in epoch 141 
    checkpoint=torch.load("/data1/Xuan_vit_ckp/2024-01-23/cifar100_deit_tiny_150_AdamW/checkpoint.pth.tar", map_location=torch.device('cpu'))
    ckp = checkpoint['state_dict']
    updated_ckp = OrderedDict((key.replace("module.", "", 1), value) for key, value in ckp.items())


    # Compile the regular expression patterns
    pattern1 = re.compile(r"blocks\.\d+\.mlp\.\w+\.step_size")
    pattern2 = re.compile(r"blocks\.\d+\.mlp\.\w+\.b_w")

    # Find keys to delete
    keys_to_delete = [key for key in updated_ckp.keys() if pattern1.match(key) or pattern2.match(key)]

    # Delete the keys
    for key in keys_to_delete:
        del updated_ckp[key]

    model.load_state_dict(updated_ckp)


    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    return model



@register_model
def deit_small_cifar100_attn(pretrained=True, **kwargs):

    model = timm.create_model(
                'deit_small_patch16_224', 
                pretrained=True,
                num_classes=100
                )

    
    
    conv_layer = model.patch_embed.proj
    in_c = conv_layer.in_channels
    out_c = conv_layer.out_channels
    ker = conv_layer.kernel_size
    stri = conv_layer.stride
    pad = conv_layer.padding
    temp_quanConv2d = quan_Conv2d(  in_c,
                                    out_c,
                                    kernel_size=ker,
                                    stride = stri,
                                    padding=pad,
                                    bias=False)
    model.patch_embed.proj = temp_quanConv2d


    # print (f"enu = {enumerate(model.blocks)}")
    for module_idx, block in enumerate(model.blocks):
        # print (f"module_idx={module_idx}")
        
        in_f = model.blocks[module_idx].attn.qkv.in_features
        out_f = model.blocks[module_idx].attn.qkv.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.qkv = temp_quanLinear

        in_f = model.blocks[module_idx].attn.proj.in_features
        out_f = model.blocks[module_idx].attn.proj.out_features
        temp_quanLinear = quan_Linear(in_f, out_f)
        model.blocks[module_idx].attn.proj = temp_quanLinear
    
    in_f = model.head.in_features
    out_f = model.head.out_features
    temp_quanLinear = quan_Linear(in_f, out_f)
    model.head = temp_quanLinear

   
    
    # this training is terminated in epoch 141 
    checkpoint=torch.load("/data1/Xuan_vit_ckp/2024-01-22/cifar100_deit_small_200_AdamW/checkpoint.pth.tar", map_location=torch.device('cpu'))
    ckp = checkpoint['state_dict']
    updated_ckp = OrderedDict((key.replace("module.", "", 1), value) for key, value in ckp.items())


    # Compile the regular expression patterns
    pattern1 = re.compile(r"blocks\.\d+\.mlp\.\w+\.step_size")
    pattern2 = re.compile(r"blocks\.\d+\.mlp\.\w+\.b_w")

    # Find keys to delete
    keys_to_delete = [key for key in updated_ckp.keys() if pattern1.match(key) or pattern2.match(key)]

    # Delete the keys
    for key in keys_to_delete:
        del updated_ckp[key]


    model.load_state_dict(updated_ckp)


    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    return model


# =========================== only attack patch embed ===============================
@register_model
def deit_base_cifar100_patchembed(pretrained=True, **kwargs):

    model = timm.create_model(
                'deit_base_patch16_224', 
                pretrained=True,
                num_classes=100
                )

    
    
    conv_layer = model.patch_embed.proj
    in_c = conv_layer.in_channels
    out_c = conv_layer.out_channels
    ker = conv_layer.kernel_size
    stri = conv_layer.stride
    pad = conv_layer.padding
    temp_quanConv2d = quan_Conv2d(  in_c,
                                    out_c,
                                    kernel_size=ker,
                                    stride = stri,
                                    padding=pad,
                                    bias=False)
    model.patch_embed.proj = temp_quanConv2d


   
    
    # this training is terminated in epoch 141 
    checkpoint=torch.load("/data1/Xuan_vit_ckp/2024-01-22/cifar100_deit_base_100_AdamW/checkpoint.pth.tar", map_location=torch.device('cpu'))
    ckp = checkpoint['state_dict']
    updated_ckp = OrderedDict((key.replace("module.", "", 1), value) for key, value in ckp.items())


    # Compile the regular expression patterns
    pattern1 = re.compile(r"blocks\.\d+\.mlp\.\w+\.step_size")
    pattern2 = re.compile(r"blocks\.\d+\.mlp\.\w+\.b_w")
    pattern3 = re.compile(r"blocks\.\d+\.attn\.\w+\.step_size")
    pattern4 = re.compile(r"blocks\.\d+\.attn\.\w+\.b_w")
    pattern5 = re.compile(r"head.step_size")
    pattern6 = re.compile(r"head.b_w")
    # Find keys to delete
    keys_to_delete = [key for key in updated_ckp.keys() if pattern1.match(key) or pattern2.match(key) or pattern3.match(key) or pattern4.match(key) or pattern5.match(key) or pattern6.match(key)]

    # Delete the keys
    for key in keys_to_delete:
        del updated_ckp[key]


    model.load_state_dict(updated_ckp)


    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    

    return model


@register_model
def deit_tiny_cifar100_patchembed(pretrained=True, **kwargs):

    model = timm.create_model(
                'deit_tiny_patch16_224', 
                pretrained=True,
                num_classes=100
                )

    
    
    conv_layer = model.patch_embed.proj
    in_c = conv_layer.in_channels
    out_c = conv_layer.out_channels
    ker = conv_layer.kernel_size
    stri = conv_layer.stride
    pad = conv_layer.padding
    temp_quanConv2d = quan_Conv2d(  in_c,
                                    out_c,
                                    kernel_size=ker,
                                    stride = stri,
                                    padding=pad,
                                    bias=False)
    model.patch_embed.proj = temp_quanConv2d


    
    # this training is terminated in epoch 141 
    checkpoint=torch.load("/data1/Xuan_vit_ckp/2024-01-23/cifar100_deit_tiny_150_AdamW/checkpoint.pth.tar", map_location=torch.device('cpu'))
    ckp = checkpoint['state_dict']
    updated_ckp = OrderedDict((key.replace("module.", "", 1), value) for key, value in ckp.items())


    # Compile the regular expression patterns
    pattern1 = re.compile(r"blocks\.\d+\.mlp\.\w+\.step_size")
    pattern2 = re.compile(r"blocks\.\d+\.mlp\.\w+\.b_w")
    pattern3 = re.compile(r"blocks\.\d+\.attn\.\w+\.step_size")
    pattern4 = re.compile(r"blocks\.\d+\.attn\.\w+\.b_w")
    pattern5 = re.compile(r"head.step_size")
    pattern6 = re.compile(r"head.b_w")
    # Find keys to delete
    keys_to_delete = [key for key in updated_ckp.keys() if pattern1.match(key) or pattern2.match(key) or pattern3.match(key) or pattern4.match(key) or pattern5.match(key) or pattern6.match(key)]


    # Delete the keys
    for key in keys_to_delete:
        del updated_ckp[key]

    model.load_state_dict(updated_ckp)


    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()

    return model



@register_model
def deit_small_cifar100_patchembed(pretrained=True, **kwargs):

    model = timm.create_model(
                'deit_small_patch16_224', 
                pretrained=True,
                num_classes=100
                )

    
    
    conv_layer = model.patch_embed.proj
    in_c = conv_layer.in_channels
    out_c = conv_layer.out_channels
    ker = conv_layer.kernel_size
    stri = conv_layer.stride
    pad = conv_layer.padding
    temp_quanConv2d = quan_Conv2d(  in_c,
                                    out_c,
                                    kernel_size=ker,
                                    stride = stri,
                                    padding=pad,
                                    bias=False)
    model.patch_embed.proj = temp_quanConv2d


    
    # this training is terminated in epoch 141 
    checkpoint=torch.load("/data1/Xuan_vit_ckp/2024-01-22/cifar100_deit_small_200_AdamW/checkpoint.pth.tar", map_location=torch.device('cpu'))
    ckp = checkpoint['state_dict']
    updated_ckp = OrderedDict((key.replace("module.", "", 1), value) for key, value in ckp.items())


    # Compile the regular expression patterns
    pattern1 = re.compile(r"blocks\.\d+\.mlp\.\w+\.step_size")
    pattern2 = re.compile(r"blocks\.\d+\.mlp\.\w+\.b_w")
    pattern3 = re.compile(r"blocks\.\d+\.attn\.\w+\.step_size")
    pattern4 = re.compile(r"blocks\.\d+\.attn\.\w+\.b_w")
    pattern5 = re.compile(r"head.step_size")
    pattern6 = re.compile(r"head.b_w")
    # Find keys to delete
    keys_to_delete = [key for key in updated_ckp.keys() if pattern1.match(key) or pattern2.match(key) or pattern3.match(key) or pattern4.match(key) or pattern5.match(key) or pattern6.match(key)]

    # Delete the keys
    for key in keys_to_delete:
        del updated_ckp[key]


    model.load_state_dict(updated_ckp)


    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()

    return model



#################################### only in head layer ========================================
@register_model
def deit_tiny_cifar100_head(pretrained=True, **kwargs):

    model = timm.create_model(
                'deit_tiny_patch16_224', 
                pretrained=True,
                num_classes=100
                )

    
    conv_layer = model.patch_embed.proj
    in_c = conv_layer.in_channels
    out_c = conv_layer.out_channels
    ker = conv_layer.kernel_size
    stri = conv_layer.stride
    pad = conv_layer.padding
    temp_quanConv2d = quan_Conv2d(  in_c,
                                    out_c,
                                    kernel_size=ker,
                                    stride = stri,
                                    padding=pad,
                                    bias=False)
    model.patch_embed.proj = temp_quanConv2d
    
    in_f = model.head.in_features
    out_f = model.head.out_features
    temp_quanLinear = quan_Linear(in_f, out_f)
    model.head = temp_quanLinear
   
    
    # this training is terminated in epoch 141 
    checkpoint=torch.load("/data1/Xuan_vit_ckp/2024-01-23/cifar100_deit_tiny_150_AdamW/checkpoint.pth.tar", map_location=torch.device('cpu'))
    ckp = checkpoint['state_dict']
    updated_ckp = OrderedDict((key.replace("module.", "", 1), value) for key, value in ckp.items())

    # Compile the regular expression patterns
    pattern1 = re.compile(r"blocks\.\d+\.mlp\.\w+\.step_size")
    pattern2 = re.compile(r"blocks\.\d+\.mlp\.\w+\.b_w")
    pattern3 = re.compile(r"blocks\.\d+\.attn\.\w+\.step_size")
    pattern4 = re.compile(r"blocks\.\d+\.attn\.\w+\.b_w")
    # Find keys to delete
    keys_to_delete = [key for key in updated_ckp.keys() if pattern1.match(key) or pattern2.match(key) or pattern3.match(key) or pattern4.match(key)]

    # Delete the keys
    for key in keys_to_delete:
        del updated_ckp[key]


    model.load_state_dict(updated_ckp)


    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    return model



@register_model
def deit_small_cifar100_head(pretrained=True, **kwargs):

    model = timm.create_model(
                'deit_small_patch16_224', 
                pretrained=True,
                num_classes=100
                )

    
    conv_layer = model.patch_embed.proj
    in_c = conv_layer.in_channels
    out_c = conv_layer.out_channels
    ker = conv_layer.kernel_size
    stri = conv_layer.stride
    pad = conv_layer.padding
    temp_quanConv2d = quan_Conv2d(  in_c,
                                    out_c,
                                    kernel_size=ker,
                                    stride = stri,
                                    padding=pad,
                                    bias=False)
    model.patch_embed.proj = temp_quanConv2d
    
    in_f = model.head.in_features
    out_f = model.head.out_features
    temp_quanLinear = quan_Linear(in_f, out_f)
    model.head = temp_quanLinear

   
    
    # this training is terminated in epoch 141 
    checkpoint=torch.load("/data1/Xuan_vit_ckp/2024-01-22/cifar100_deit_small_200_AdamW/checkpoint.pth.tar", map_location=torch.device('cpu'))
    ckp = checkpoint['state_dict']
    updated_ckp = OrderedDict((key.replace("module.", "", 1), value) for key, value in ckp.items())

    # Compile the regular expression patterns
    pattern1 = re.compile(r"blocks\.\d+\.mlp\.\w+\.step_size")
    pattern2 = re.compile(r"blocks\.\d+\.mlp\.\w+\.b_w")
    pattern3 = re.compile(r"blocks\.\d+\.attn\.\w+\.step_size")
    pattern4 = re.compile(r"blocks\.\d+\.attn\.\w+\.b_w")
    # Find keys to delete
    keys_to_delete = [key for key in updated_ckp.keys() if pattern1.match(key) or pattern2.match(key) or pattern3.match(key) or pattern4.match(key)]

    # Delete the keys
    for key in keys_to_delete:
        del updated_ckp[key]


    model.load_state_dict(updated_ckp)


    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    return model


@register_model
def deit_base_cifar100_head(pretrained=True, **kwargs):

    model = timm.create_model(
                'deit_base_patch16_224', 
                pretrained=True,
                num_classes=100,
                #pretrained=True
                )


    for name, parameter in model.named_parameters():
        print(f"{name}: shape = {parameter.shape}, number of weights = {parameter.numel()}")
    
    
    conv_layer = model.patch_embed.proj
    in_c = conv_layer.in_channels
    out_c = conv_layer.out_channels
    ker = conv_layer.kernel_size
    stri = conv_layer.stride
    pad = conv_layer.padding
    temp_quanConv2d = quan_Conv2d(  in_c,
                                    out_c,
                                    kernel_size=ker,
                                    stride = stri,
                                    padding=pad,
                                    bias=False)
    model.patch_embed.proj = temp_quanConv2d
    in_f = model.head.in_features
    out_f = model.head.out_features
    temp_quanLinear = quan_Linear(in_f, out_f)
    model.head = temp_quanLinear
   
    
    # this training is terminated in epoch 141 
    checkpoint=torch.load("/data1/Xuan_vit_ckp/2024-01-22/cifar100_deit_base_100_AdamW/checkpoint.pth.tar", map_location=torch.device('cpu'))
    ckp = checkpoint['state_dict']
    updated_ckp = OrderedDict((key.replace("module.", "", 1), value) for key, value in ckp.items())


    # Compile the regular expression patterns
    pattern1 = re.compile(r"blocks\.\d+\.mlp\.\w+\.step_size")
    pattern2 = re.compile(r"blocks\.\d+\.mlp\.\w+\.b_w")
    pattern3 = re.compile(r"blocks\.\d+\.attn\.\w+\.step_size")
    pattern4 = re.compile(r"blocks\.\d+\.attn\.\w+\.b_w")
    # Find keys to delete
    keys_to_delete = [key for key in updated_ckp.keys() if pattern1.match(key) or pattern2.match(key) or pattern3.match(key) or pattern4.match(key)]

    #print (keys_to_delete)
    # Delete the keys
    for key in keys_to_delete:
        del updated_ckp[key]

    model.load_state_dict(updated_ckp)


    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    return model


