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
    "vit_tiny", "vit_tiny", "vit_tiny",
    "deit_tiny", "deit_tiny", "deit_tiny",
    "swin_tiny", "swin_small", "swin_base",

    "deit_base_11blocks_cifar10", "deit_base_10blocks_cifar10", "deit_base_9blocks_cifar10", "deit_base_8blocks_cifar10", "deit_base_7blocks_cifar10", "deit_base_6blocks_cifar10", "deit_base_4blocks_cifar10", "deit_base_2blocks_cifar10",
    "deit_base_10blocks_cifar100", "deit_base_8blocks_cifar100", "deit_base_6blocks_cifar100", "deit_base_4blocks_cifar100", "deit_base_2blocks_cifar100",

    # added to look into the num_heads
    "deit_large", "vit_huge", "deit_huge",

    # number of heads experiments
    "deit_base_10heads_cifar10_train",

    # attack mlp only
    "deit_tiny_head", "deit_tiny_head", "deit_tiny_head",
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
def vit_tiny(pretrained=True, **kwargs):

    model = timm.create_model(
                'vit_tiny_patch16_224.augreg_in21k', 
                pretrained=True,
                num_classes=100
                )

    ckp_save = model.state_dict()
    
    
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
   

    model_dict = model.state_dict() # after introduce quan layers, b_w and step_size will appear in model dictionary, which do not exist in the model dictionary before

    new_ckp = {k: v for k, v in ckp_save.items() if k in model_dict} # update the value in ckp into the state dict of model with quan layers
    
    # load new dictionary
    model_dict.update(new_ckp)
    model.load_state_dict(model_dict)

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
def vit_small(pretrained=True, **kwargs):

    
    model = timm.create_model(
                'vit_small_patch16_224.augreg_in21k', 
                pretrained=True,
                num_classes=100
                )

    ckp_save = model.state_dict()
    
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
   

    model_dict = model.state_dict() # after introduce quan layers, b_w and step_size will appear in model dictionary, which do not exist in the model dictionary before

    new_ckp = {k: v for k, v in ckp_save.items() if k in model_dict} # update the value in ckp into the state dict of model with quan layers
    
    # load new dictionary
    model_dict.update(new_ckp)
    model.load_state_dict(model_dict)

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
def vit_base(pretrained=True, **kwargs):

    
    model = timm.create_model(
                'vit_base_patch16_224.augreg_in21k', 
                pretrained=True,
                num_classes=100
                )

    ckp_save = model.state_dict()
    
    
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
   

    model_dict = model.state_dict() # after introduce quan layers, b_w and step_size will appear in model dictionary, which do not exist in the model dictionary before

    new_ckp = {k: v for k, v in ckp_save.items() if k in model_dict} # update the value in ckp into the state dict of model with quan layers
    
    # load new dictionary
    model_dict.update(new_ckp)
    model.load_state_dict(model_dict)

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
def deit_tiny(pretrained=True, **kwargs):

    
    model = timm.create_model(
                'deit_tiny_patch16_224', 
                pretrained=True,
                num_classes=100
                )

    ckp_save = model.state_dict()
    
    
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
   

    model_dict = model.state_dict() # after introduce quan layers, b_w and step_size will appear in model dictionary, which do not exist in the model dictionary before

    new_ckp = {k: v for k, v in ckp_save.items() if k in model_dict} # update the value in ckp into the state dict of model with quan layers
    
    # load new dictionary
    model_dict.update(new_ckp)
    model.load_state_dict(model_dict)

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
def deit_small(pretrained=True, **kwargs):

    
    model = timm.create_model(
                'deit_small_patch16_224', 
                pretrained=True,
                num_classes=100
                )

    ckp_save = model.state_dict()
    
    
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
   

    model_dict = model.state_dict() # after introduce quan layers, b_w and step_size will appear in model dictionary, which do not exist in the model dictionary before

    new_ckp = {k: v for k, v in ckp_save.items() if k in model_dict} # update the value in ckp into the state dict of model with quan layers
    
    # load new dictionary
    model_dict.update(new_ckp)
    model.load_state_dict(model_dict)

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
def deit_base(pretrained=True, **kwargs):

    
    model = timm.create_model(
                'deit_base_patch16_224', 
                num_classes=100
                )

    ckp_save = model.state_dict()
    
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
   

    model_dict = model.state_dict() # after introduce quan layers, b_w and step_size will appear in model dictionary, which do not exist in the model dictionary before

    new_ckp = {k: v for k, v in ckp_save.items() if k in model_dict} # update the value in ckp into the state dict of model with quan layers
    
    # load new dictionary
    model_dict.update(new_ckp)
    model.load_state_dict(model_dict)

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
def deit_base_4blocks_cifar100(pretrained=True, **kwargs):

    
    model = timm.create_model(
                'deit_base_patch16_224', 
                pretrained=True,
                num_classes=100
                )

    ckp_save = model.state_dict()
    
    
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
   

    model_dict = model.state_dict() # after introduce quan layers, b_w and step_size will appear in model dictionary, which do not exist in the model dictionary before

    new_ckp = {k: v for k, v in ckp_save.items() if k in model_dict} # update the value in ckp into the state dict of model with quan layers
    
    # load new dictionary
    model_dict.update(new_ckp)
    model.load_state_dict(model_dict)

    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    # Remove blocks 8 to 11 from the 'blocks' Sequential container
    new_blocks = nn.Sequential(*list(model.blocks.children())[:4])

    # Assign the modified blocks back to the model
    model.blocks = new_blocks


    return model


@register_model
def deit_base_8blocks_cifar10(pretrained=True, **kwargs):

    
    model = timm.create_model(
                'deit_base_patch16_224', 
                pretrained=True,
                num_classes=10
                )

    ckp_save = model.state_dict()
    
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
   

    model_dict = model.state_dict() # after introduce quan layers, b_w and step_size will appear in model dictionary, which do not exist in the model dictionary before

    new_ckp = {k: v for k, v in ckp_save.items() if k in model_dict} # update the value in ckp into the state dict of model with quan layers
    
    # load new dictionary
    model_dict.update(new_ckp)
    model.load_state_dict(model_dict)

    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    # Remove blocks 8 to 11 from the 'blocks' Sequential container
    new_blocks = nn.Sequential(*list(model.blocks.children())[:8])

    # Assign the modified blocks back to the model
    model.blocks = new_blocks


    return model



@register_model
def deit_base_4blocks_cifar10(pretrained=True, **kwargs):

    
    model = timm.create_model(
                'deit_base_patch16_224', 
                pretrained=True,
                num_classes=10
                )

    ckp_save = model.state_dict()
    
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
   

    model_dict = model.state_dict() # after introduce quan layers, b_w and step_size will appear in model dictionary, which do not exist in the model dictionary before

    new_ckp = {k: v for k, v in ckp_save.items() if k in model_dict} # update the value in ckp into the state dict of model with quan layers
    
    # load new dictionary
    model_dict.update(new_ckp)
    model.load_state_dict(model_dict)

    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    # Remove blocks 8 to 11 from the 'blocks' Sequential container
    new_blocks = nn.Sequential(*list(model.blocks.children())[:4])

    # Assign the modified blocks back to the model
    model.blocks = new_blocks


    return model


@register_model
def deit_base_10blocks_cifar100(pretrained=True, **kwargs):

    
    model = timm.create_model(
                'deit_base_patch16_224', 
                pretrained=True,
                num_classes=100
                )

    ckp_save = model.state_dict()
    
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
   

    model_dict = model.state_dict() # after introduce quan layers, b_w and step_size will appear in model dictionary, which do not exist in the model dictionary before

    new_ckp = {k: v for k, v in ckp_save.items() if k in model_dict} # update the value in ckp into the state dict of model with quan layers
    
    # load new dictionary
    model_dict.update(new_ckp)
    model.load_state_dict(model_dict)

    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    # Remove blocks 8 to 11 from the 'blocks' Sequential container
    new_blocks = nn.Sequential(*list(model.blocks.children())[:10])

    # Assign the modified blocks back to the model
    model.blocks = new_blocks


    return model


@register_model
def deit_base_10blocks_cifar10(pretrained=True, **kwargs):

    
    model = timm.create_model(
                'deit_base_patch16_224', 
                pretrained=True,
                num_classes=10
                )

    ckp_save = model.state_dict()
    
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
   

    model_dict = model.state_dict() # after introduce quan layers, b_w and step_size will appear in model dictionary, which do not exist in the model dictionary before

    new_ckp = {k: v for k, v in ckp_save.items() if k in model_dict} # update the value in ckp into the state dict of model with quan layers
    
    # load new dictionary
    model_dict.update(new_ckp)
    model.load_state_dict(model_dict)

    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    # Remove blocks 8 to 11 from the 'blocks' Sequential container
    new_blocks = nn.Sequential(*list(model.blocks.children())[:10])

    # Assign the modified blocks back to the model
    model.blocks = new_blocks


    return model


@register_model
def deit_base_6blocks_cifar100(pretrained=True, **kwargs):

    
    model = timm.create_model(
                'deit_base_patch16_224', 
                pretrained=True,
                num_classes=100
                )

    ckp_save = model.state_dict()
    
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
   

    model_dict = model.state_dict() # after introduce quan layers, b_w and step_size will appear in model dictionary, which do not exist in the model dictionary before

    new_ckp = {k: v for k, v in ckp_save.items() if k in model_dict} # update the value in ckp into the state dict of model with quan layers
    
    # load new dictionary
    model_dict.update(new_ckp)
    model.load_state_dict(model_dict)

    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    # Remove blocks 8 to 11 from the 'blocks' Sequential container
    new_blocks = nn.Sequential(*list(model.blocks.children())[:6])

    # Assign the modified blocks back to the model
    model.blocks = new_blocks


    return model


@register_model
def deit_base_6blocks_cifar10(pretrained=True, **kwargs):

    
    model = timm.create_model(
                'deit_base_patch16_224', 
                pretrained=True,
                num_classes=10
                )

    ckp_save = model.state_dict()
    
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
   

    model_dict = model.state_dict() # after introduce quan layers, b_w and step_size will appear in model dictionary, which do not exist in the model dictionary before

    new_ckp = {k: v for k, v in ckp_save.items() if k in model_dict} # update the value in ckp into the state dict of model with quan layers
    
    # load new dictionary
    model_dict.update(new_ckp)
    model.load_state_dict(model_dict)

    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    # Remove blocks 8 to 11 from the 'blocks' Sequential container
    new_blocks = nn.Sequential(*list(model.blocks.children())[:6])

    # Assign the modified blocks back to the model
    model.blocks = new_blocks


    return model


@register_model
def deit_base_2blocks_cifar100(pretrained=True, **kwargs):

    
    model = timm.create_model(
                'deit_base_patch16_224', 
                pretrained=True,
                num_classes=100
                )

    ckp_save = model.state_dict()
    
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
   

    model_dict = model.state_dict() # after introduce quan layers, b_w and step_size will appear in model dictionary, which do not exist in the model dictionary before

    new_ckp = {k: v for k, v in ckp_save.items() if k in model_dict} # update the value in ckp into the state dict of model with quan layers
    
    # load new dictionary
    model_dict.update(new_ckp)
    model.load_state_dict(model_dict)

    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    # Remove blocks 8 to 11 from the 'blocks' Sequential container
    new_blocks = nn.Sequential(*list(model.blocks.children())[:2])

    # Assign the modified blocks back to the model
    model.blocks = new_blocks


    return model


@register_model
def deit_base_2blocks_cifar10(pretrained=True, **kwargs):

    
    model = timm.create_model(
                'deit_base_patch16_224', 
                pretrained=True,
                num_classes=10
                )

    ckp_save = model.state_dict()
    
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
   

    model_dict = model.state_dict() # after introduce quan layers, b_w and step_size will appear in model dictionary, which do not exist in the model dictionary before

    new_ckp = {k: v for k, v in ckp_save.items() if k in model_dict} # update the value in ckp into the state dict of model with quan layers
    
    # load new dictionary
    model_dict.update(new_ckp)
    model.load_state_dict(model_dict)

    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    # Remove blocks 8 to 11 from the 'blocks' Sequential container
    new_blocks = nn.Sequential(*list(model.blocks.children())[:2])

    # Assign the modified blocks back to the model
    model.blocks = new_blocks


    return model


@register_model
def deit_base_11blocks_cifar10(pretrained=True, **kwargs):

    
    model = timm.create_model(
                'deit_base_patch16_224', 
                pretrained=True,
                num_classes=10
                )

    ckp_save = model.state_dict()
    
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
   

    model_dict = model.state_dict() # after introduce quan layers, b_w and step_size will appear in model dictionary, which do not exist in the model dictionary before

    new_ckp = {k: v for k, v in ckp_save.items() if k in model_dict} # update the value in ckp into the state dict of model with quan layers
    
    # load new dictionary
    model_dict.update(new_ckp)
    model.load_state_dict(model_dict)

    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    # Remove blocks 8 to 11 from the 'blocks' Sequential container
    new_blocks = nn.Sequential(*list(model.blocks.children())[:11])

    # Assign the modified blocks back to the model
    model.blocks = new_blocks


    return model


@register_model
def deit_base_9blocks_cifar10(pretrained=True, **kwargs):

    
    model = timm.create_model(
                'deit_base_patch16_224', 
                pretrained=True,
                num_classes=10
                )

    ckp_save = model.state_dict()
    
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
   

    model_dict = model.state_dict() # after introduce quan layers, b_w and step_size will appear in model dictionary, which do not exist in the model dictionary before

    new_ckp = {k: v for k, v in ckp_save.items() if k in model_dict} # update the value in ckp into the state dict of model with quan layers
    
    # load new dictionary
    model_dict.update(new_ckp)
    model.load_state_dict(model_dict)

    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    # Remove blocks 8 to 11 from the 'blocks' Sequential container
    new_blocks = nn.Sequential(*list(model.blocks.children())[:9])

    # Assign the modified blocks back to the model
    model.blocks = new_blocks


    return model


@register_model
def deit_base_7blocks_cifar10(pretrained=True, **kwargs):

    
    model = timm.create_model(
                'deit_base_patch16_224', 
                pretrained=True,
                num_classes=10
                )

    ckp_save = model.state_dict()
    
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
   

    model_dict = model.state_dict() # after introduce quan layers, b_w and step_size will appear in model dictionary, which do not exist in the model dictionary before

    new_ckp = {k: v for k, v in ckp_save.items() if k in model_dict} # update the value in ckp into the state dict of model with quan layers
    
    # load new dictionary
    model_dict.update(new_ckp)
    model.load_state_dict(model_dict)

    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    # Remove blocks 8 to 11 from the 'blocks' Sequential container
    new_blocks = nn.Sequential(*list(model.blocks.children())[:7])

    # Assign the modified blocks back to the model
    model.blocks = new_blocks


    return model


def vit_large(pretrained=True, **kwargs):

    
    model = timm.create_model(
                'vit_large_patch16_224.augreg_in21k', 
                pretrained=True,
                num_classes=10
                )

    ckp_save = model.state_dict()
    
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
   

    model_dict = model.state_dict() # after introduce quan layers, b_w and step_size will appear in model dictionary, which do not exist in the model dictionary before

    new_ckp = {k: v for k, v in ckp_save.items() if k in model_dict} # update the value in ckp into the state dict of model with quan layers
    
    # load new dictionary
    model_dict.update(new_ckp)
    model.load_state_dict(model_dict)

    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    return model



def vit_huge(pretrained=True, **kwargs):

    
    model = timm.create_model(
                'vit_huge_patch14_224', 
                pretrained=True,
                num_classes=10
                )

    ckp_save = model.state_dict()
    
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
   

    model_dict = model.state_dict() # after introduce quan layers, b_w and step_size will appear in model dictionary, which do not exist in the model dictionary before

    new_ckp = {k: v for k, v in ckp_save.items() if k in model_dict} # update the value in ckp into the state dict of model with quan layers
    
    # load new dictionary
    model_dict.update(new_ckp)
    model.load_state_dict(model_dict)

    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    return model



def deit_huge(pretrained=True, **kwargs):

    
    model = timm.create_model(
                'deit3_huge_patch14_224', 
                pretrained=True,
                num_classes=10
                )

    ckp_save = model.state_dict()
    
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
   

    model_dict = model.state_dict() # after introduce quan layers, b_w and step_size will appear in model dictionary, which do not exist in the model dictionary before

    new_ckp = {k: v for k, v in ckp_save.items() if k in model_dict} # update the value in ckp into the state dict of model with quan layers
    
    # load new dictionary
    model_dict.update(new_ckp)
    model.load_state_dict(model_dict)

    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    return model


#################################### number of heads ##########################################
@register_model
def deit_base_10heads_cifar10_train(pretrained=True, **kwargs):

    model = timm.create_model(
                'deit_base_patch16_224', 
                pretrained=True,
                num_classes=10
                )

    ckp_save = model.state_dict()
    
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
   

    model_dict = model.state_dict() # after introduce quan layers, b_w and step_size will appear in model dictionary, which do not exist in the model dictionary before

    new_ckp = {k: v for k, v in ckp_save.items() if k in model_dict} # update the value in ckp into the state dict of model with quan layers
    
    # load new dictionary
    model_dict.update(new_ckp)
    model.load_state_dict(model_dict)

    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    
    
    ########################## modify the number of heads in qkv ######################
    for block in model.blocks:
        # Assuming each block has an `attn` attribute with a `qkv` linear layer
        # and that the original in_features is 768 (as per your model definition)
        in_features = 768  # This remains unchanged
        out_features = 1920  # New out_features calculated for 10 heads
    
        # Update the `qkv` layer of each block
        block.attn.qkv = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
    print (model)

    print (kwargs)
    print (type(kwargs))
    return model



########################## only in head layer =====================
@register_model
def deit_tiny_head(pretrained=True, **kwargs):

    
    model = timm.create_model(
                'deit_tiny_patch16_224', 
                pretrained=True,
                num_classes=100
                )

    ckp_save = model.state_dict()
    
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
   

    model_dict = model.state_dict() # after introduce quan layers, b_w and step_size will appear in model dictionary, which do not exist in the model dictionary before

    new_ckp = {k: v for k, v in ckp_save.items() if k in model_dict} # update the value in ckp into the state dict of model with quan layers
    
    # load new dictionary
    model_dict.update(new_ckp)
    model.load_state_dict(model_dict)

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
def deit_small(pretrained=True, **kwargs):

    
    model = timm.create_model(
                'deit_small_patch16_224', 
                pretrained=True,
                num_classes=100
                )

    ckp_save = model.state_dict()
    
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
   

    model_dict = model.state_dict() # after introduce quan layers, b_w and step_size will appear in model dictionary, which do not exist in the model dictionary before

    new_ckp = {k: v for k, v in ckp_save.items() if k in model_dict} # update the value in ckp into the state dict of model with quan layers
    
    # load new dictionary
    model_dict.update(new_ckp)
    model.load_state_dict(model_dict)

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
def deit_base(pretrained=True, **kwargs):

    
    model = timm.create_model(
                'deit_base_patch16_224', 
                pretrained=True,
                num_classes=100
                )

    ckp_save = model.state_dict()
    
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
   

    model_dict = model.state_dict() # after introduce quan layers, b_w and step_size will appear in model dictionary, which do not exist in the model dictionary before

    new_ckp = {k: v for k, v in ckp_save.items() if k in model_dict} # update the value in ckp into the state dict of model with quan layers
    
    # load new dictionary
    model_dict.update(new_ckp)
    model.load_state_dict(model_dict)

    # reset stepsize for quan_Conv2d and quan_Linear to get quantized weight
    model.patch_embed.proj.__reset_stepsize__()
    for module_idx, block in enumerate(model.blocks):
        model.blocks[module_idx].attn.qkv.__reset_stepsize__()
        model.blocks[module_idx].attn.proj.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc1.__reset_stepsize__()
        model.blocks[module_idx].mlp.fc2.__reset_stepsize__()
    model.head.__reset_stepsize__()
    

    return model


