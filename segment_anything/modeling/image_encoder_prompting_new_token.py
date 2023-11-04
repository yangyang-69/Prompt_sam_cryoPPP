
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

        self.blocks = nn.ModuleList()
        self.img_size = img_size
        self.patch_size = patch_size
        for i in range(depth):
            block = Block_new(
                args=self.args,
                block_number=i,
                dim=embed_dim,
                patch_size=patch_size,
                img_size=img_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)
        # super().__init__()

    def forward(self, x):

        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.neck(x.permute(0, 3, 1, 2))

        if self.args.token_output_type == 'slice':
            if x.shape[2] > x.shape[3]:
                image_embedding = x[:, :, int(self.args.NUM_TOKENS / int(self.img_size / self.patch_size)):, :]
            else:
                image_embedding = x

        elif self.args.token_output_type == 'linear':
            x = x.permute(0, 1, 3, 2)
            x = self.linear(x)
            image_embedding = x.permute(0, 1, 3, 2)

        return image_embedding


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x

class Block_new(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        args,
        block_number: int,
        dim: int,
        num_heads: int,
        patch_size: int,
        img_size:int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.prompt_embeddings = None
        self.args = args
        self.block_number = block_number
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

        # decide whether add token
        if self.args.deep_token_block_configuration[self.block_number] == 1:

            self.args.hidden_size = dim  # h: 1280  ,  b: 768
            self.patch_size = patch_size
            self.img_size = img_size
            self.num_tokens = self.args.NUM_TOKENS   # number of prompted tokens
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
                self.prompt_dim = self.args.hidden_size
                self.prompt_proj = nn.Identity()

            if self.args.deep_token_block_configuration[self.block_number] == 1:

                # initiate prompt:
                if self.args.INITIATION == "random":
                    val = math.sqrt(
                        6. / float(3 * reduce(mul, (self.patch_size, self.patch_size), 1) + self.prompt_dim))

                    self.prompt_embeddings = nn.Parameter(torch.zeros(
                        self.args.b, int(self.num_tokens / int(self.img_size / self.patch_size)),
                        int(self.img_size / self.patch_size),
                        self.prompt_dim))

                    # xavier_uniform initialization
                    nn.init.uniform_(self.prompt_embeddings.data, -val, val)

            self.linear = nn.Linear((int(self.num_tokens/int(img_size / patch_size)) + int(img_size / patch_size)), int(img_size / patch_size))


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.args.deep_token_block_configuration[self.block_number] == 1:

            B = x.shape[0]
            deep_prompt_emb = self.prompt_dropout(self.prompt_proj(self.prompt_embeddings[:B,:,:,:])).to(self.args.gpu_device)

            if x.shape[1] > x.shape[2]:
                x = torch.cat(
                    (deep_prompt_emb, x[:, int(self.num_tokens / int(self.img_size / self.patch_size)):, :, :]), dim=1)
            else:
                x = torch.cat((deep_prompt_emb, x), dim=1)

        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x