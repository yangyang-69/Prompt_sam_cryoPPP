#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Sam 
@File    ：image_encoder_prompting.py
@Author  ：yang
@Date    ：2023/8/2 11:50
'''
# PromptVit继承了transformer(原版Vit的类)
# 做一个ImageEncoder的继承类，重点是实现追加的token，具体追加的地方在每一个Block内部，相当于每一个transformer的内部追加
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn import Dropout
import math

from collections import Counter
from typing import Optional, Tuple, Type, Any
from functools import reduce
from operator import mul

from notebooks.SAM_adapter_conf.SAM_adapter_utils import get_network
from notebooks.dataset import CryopppDataset
from .common import LayerNorm2d, MLPBlock

from segment_anything.modeling.image_encoder_adapter import ImageEncoderViT, Block, Attention
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

def save_image_embedding(args):

    transform_train = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])
    transform_train_seg = transforms.Compose([
        transforms.Resize((args.out_size, args.out_size)),
        transforms.ToTensor(),
    ])

    GPUdevice = torch.device('cuda', args.gpu_device)

    train_dataset = CryopppDataset(args, args.data_path, transform=transform_train,
                                   transform_msk=transform_train_seg, mode='train', prompt=args.prompt_approach)
    nice_train_loader = DataLoader(train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)

    from segment_anything.build_sam_adapter import sam_model_registry
    if args.sam_vit_model == "b":
        ori_sam_net = sam_model_registry['vit_b'](args, checkpoint=args.sam_ckpt).to(GPUdevice)
    elif args.sam_vit_model == "h":
        ori_sam_net = sam_model_registry['vit_h'](args, checkpoint=args.sam_ckpt).to(GPUdevice)

    tensor_save_path = os.path.join(args.data_path,"train_image_embedding")
    if not os.path.exists(tensor_save_path):
        os.makedirs(tensor_save_path)

    for img_path in nice_train_loader.dataset.name_list:
        img = Image.open(img_path).convert('RGB')
        img = transform_train(img)
        img = img[None,:].to(dtype=torch.float32, device=GPUdevice)

        # img_embedding = ori_sam_net.image_encoder(img)

        img_embedding = ori_sam_net.image_encoder.patch_embed(img)
        if ori_sam_net.image_encoder.pos_embed is not None:
            img_embedding = img_embedding + ori_sam_net.image_encoder.pos_embed

        for blk in ori_sam_net.image_encoder.blocks:
            img_embedding = blk(img_embedding)

        torch.save(img_embedding, os.path.join(tensor_save_path,img_path.split("/")[-1].split(".")[0] + ".pt"))

    print()
    # return embedding_mean

def load_image_embedding(args):

    GPUdevice = torch.device('cuda', args.gpu_device)

    transform_train = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])
    transform_train_seg = transforms.Compose([
        transforms.Resize((args.out_size, args.out_size)),
        transforms.ToTensor(),
    ])

    train_dataset = CryopppDataset(args, args.data_path, transform=transform_train,
                                   transform_msk=transform_train_seg, mode='train', prompt=args.prompt_approach)
    nice_train_loader = DataLoader(train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)


    embedding_sum = torch.zeros(1,64,64,768).to(GPUdevice)
    tensor_save_path = os.path.join(args.data_path,"train_image_embedding")
    for image_embedding_path in os.listdir(tensor_save_path):
        img_embedding = torch.load(os.path.join(tensor_save_path, image_embedding_path),map_location=GPUdevice)
        embedding_sum += img_embedding

    embedding_mean = embedding_sum / len(nice_train_loader.dataset.name_list)
    return embedding_mean

class PromptedImageEncoderViT(ImageEncoderViT):
    def __init__(self,
                 args,
                 img_size: int = 1024,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 out_chans: int = 256,
                 qkv_bias: bool = True,
                 norm_layer: Type[nn.Module] = nn.LayerNorm,
                 act_layer: Type[nn.Module] = nn.GELU,
                 use_abs_pos: bool = True,
                 use_rel_pos: bool = False,
                 rel_pos_zero_init: bool = True,
                 window_size: int = 0,
                 global_attn_indexes: Tuple[int, ...] = ()) -> None:

        super().__init__(args, img_size, patch_size, in_chans, embed_dim, depth, num_heads, mlp_ratio, out_chans,
                         qkv_bias, norm_layer, act_layer, use_abs_pos, use_rel_pos, rel_pos_zero_init, window_size,
                         global_attn_indexes)
        # super().__init__()

        self.args.hidden_size = embed_dim  # h: 1280  ,  b: 768
        self.patch_size = patch_size
        patch_sizes = (patch_size, patch_size)

        num_tokens = self.args.NUM_TOKENS
        self.num_tokens = num_tokens  # number of prompted tokens

        self.prompt_dropout = Dropout(self.args.DROPOUT)

        # if project the prompt embeddings
        if self.args.PROJECT > -1:
            # only for prepend / add
            prompt_dim = self.args.PROJECT
            self.prompt_proj = nn.Linear(
                prompt_dim, self.args.hidden_size)
            nn.init.kaiming_normal_(
                self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            prompt_dim = self.args.hidden_size
            self.prompt_proj = nn.Identity()

        # initiate prompt:
        if self.args.INITIATION == "random":
            val = math.sqrt(6. / float(3 * reduce(mul, patch_sizes, 1) + prompt_dim))  # noqa

            self.prompt_embeddings = nn.Parameter(torch.zeros(  # nn.Parameter: 可以被修正的类型
                self.args.b, int(num_tokens/int(img_size / patch_size)), int(img_size / patch_size), prompt_dim))  # 0为最初始
            # self.prompt_embeddings = nn.Parameter(torch.zeros(  # nn.Parameter: 可以被修正的类型
            #     self.args.b, int(img_size / patch_size), int(num_tokens / int(img_size / patch_size)), prompt_dim))  # 0为最初始

            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)  # 对token做一个初始化

            if self.args.PROMPT_DEEP:

                # save_image_embedding(args)
                # embedding_mean = torch.mean(load_image_embedding(args), dim=1)

                self.deep_num_layers = Counter(self.args.deep_token_block_configuration)[1]

                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                    self.deep_num_layers, self.args.b, int(num_tokens/int(img_size / patch_size)), int(img_size / patch_size), prompt_dim))
                # self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                #     self.deep_num_layers, self.args.b,int(img_size / patch_size), int(num_tokens / int(img_size / patch_size)),prompt_dim))
                # 用image_embedding的均值进行初始化
                # self.deep_prompt_embeddings.data = embedding_mean.repeat(self.deep_num_layers, self.args.b, int(num_tokens/int(img_size / patch_size)), 1, 1)
                # xavier_uniform initialization
                nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

        else:
            raise ValueError("Other initiation scheme is not supported")

        self.token_output = self.args.token_output_type

        self.linear = nn.Linear((int(num_tokens/int(img_size / patch_size)) + int(img_size / patch_size)), int(img_size / patch_size))

    def incorporate_prompt(self, x):  # token列表追加了prompt
        # combine prompt embeddings with image-patch embeddings
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        B = x.shape[0]

        x = torch.cat((
            self.prompt_dropout(self.prompt_proj(self.prompt_embeddings)[:B, :, :, :]), x), dim=1)

        # x = torch.cat((
        #     self.prompt_dropout(self.prompt_proj(self.prompt_embeddings)[:B,:,:,:]), x), dim=2)

        return x

    def forward_deep_prompt(self, x):
        B = x.shape[0]
        num_layers = len(self.blocks)
        deep_index_count = 0

        for i in range(num_layers):

            if i != 0 and self.args.deep_token_block_configuration[i-1] == 1:  # 31
                deep_prompt_emb = self.prompt_dropout(self.prompt_proj(self.deep_prompt_embeddings[deep_index_count][:B,:,:,:]))

                x = torch.cat((
                    deep_prompt_emb, x[:, int(self.args.NUM_TOKENS/int(self.img_size / self.patch_size)):, :, :]
                ), dim=1)

                # x = torch.cat((
                #     deep_prompt_emb, x[:, :, int(self.args.NUM_TOKENS / int(self.img_size / self.patch_size)):, :]
                # ), dim=2)

                deep_index_count += 1

            x = self.blocks[i](x)

        return x

    def forward(self, x):
        embedding_output = self.incorporate_prompt(x)

        if self.args.PROMPT_DEEP:
            x = self.forward_deep_prompt(embedding_output)

        else:
            for blk in self.blocks:
                x = blk(embedding_output)

        x = self.neck(x.permute(0, 3, 1, 2))

        if self.token_output == 'slice':
            image_embedding = x[:, :, int(self.args.NUM_TOKENS/int(self.img_size / self.patch_size)):, :]  #  (1,256,64,64)   或者做一个projector
            # image_embedding = x[:, :, :, int(self.args.NUM_TOKENS / int(self.img_size / self.patch_size)):]
            # token = x[:, :, :int(self.args.NUM_TOKENS/int(self.img_size / self.patch_size)), :]

        elif self.token_output == 'linear':
            x = x.permute(0, 1, 3, 2)
            x = self.linear(x)
            image_embedding = x.permute(0, 1, 3, 2)

        return image_embedding
