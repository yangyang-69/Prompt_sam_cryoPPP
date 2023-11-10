""" helper function

author junde
"""
import copy
import sys
from copy import deepcopy

import numpy
import pandas as pd
from matplotlib import pyplot as plt, patches
from pycocotools.cocoeval import Params
from torchvision import transforms
import torch.nn as nn
from torch.autograd import Function
from torch.optim.lr_scheduler import _LRScheduler
import cv2
import torchvision
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable
from torch import autograd
import random
from detectron2 import _C

import collections
import logging
import math
import os
import time
from datetime import datetime

import dateutil.tz

from typing import Union, Optional, List, Tuple, Text, BinaryIO, Iterable, Callable
import pathlib
# from lucent.optvis.param.spatial import pixel_image, fft_image, init_image
# from lucent.optvis.param.color import to_valid_rgb
# from lucent.optvis import objectives, transform, param
# from lucent.misc.io import show

import warnings
from collections import OrderedDict
import numpy as np
from PIL import Image
import torch

from notebooks.SAM_conf import SAM_cfg
# from precpt import run_precpt
from segment_anything.modeling.discriminator import Discriminator
# from siren_pytorch import SirenNet, SirenWrapper

from tqdm import tqdm

from monai.transforms import (
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    EnsureTyped,
)

from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    load_decathlon_datalist,
    set_track_meta,
)
args = SAM_cfg.parse_args()
device = torch.device('cuda', args.gpu_device)

'''preparation of domain loss'''
# cnn = vgg19(pretrained=True).features.to(device).eval()
# cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
# cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# netD = Discriminator(1).to(device)
# netD.apply(init_D)
# beta1 = 0.5
# dis_lr = 0.0002
# optimizerD = optim.Adam(netD.parameters(), lr=dis_lr, betas=(beta1, 0.999))
'''end'''

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print()
    print('total:{}'.format(total_num))
    print('trainable:{}'.format(trainable_num))

def get_network(args, net, use_gpu=True, gpu_device = 0, distribution = True):
    """ return given network
    """

    if net == 'sam':
        from segment_anything.build_sam import sam_model_registry
        net = sam_model_registry['vit_h'](checkpoint=args.sam_ckpt).to(device)
    elif net == 'sam_adapter':
        from segment_anything.build_sam_adapter import sam_model_registry
        net = sam_model_registry['vit_h'](args,checkpoint=args.sam_ckpt).to(device)
    elif net == 'sam_fineTuning':
        from segment_anything.build_sam_adapter import sam_model_registry
        net = sam_model_registry['vit_h'](args,checkpoint=args.sam_ckpt).to(device)
    elif net == 'PromptVit':
        if args.sam_vit_model == "h":
            if args.token_method == "new":
                from segment_anything.build_sam_promptvit_new_token import sam_model_registry
                net = sam_model_registry['vit_h'](args,checkpoint=args.sam_ckpt).to(device)

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if use_gpu:
        #net = net.cuda(device = gpu_device)
        if distribution != 'none':
            net = torch.nn.DataParallel(net,device_ids=[int(id) for id in args.distributed.split(',')])
            net = net.to(device=gpu_device)
        else:
            net = net.to(device=gpu_device)

    return net


def get_decath_loader(args):

    train_transforms = Compose(
        [   
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_size, args.roi_size, args.chunk),
                pos=1,
                neg=1,
                num_samples=args.num_sample,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[2],
                prob=0.10,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
            ),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
        ]
    )



    data_dir = args.data_path
    split_JSON = "dataset_0.json"

    datasets = os.path.join(data_dir, split_JSON)
    datalist = load_decathlon_datalist(datasets, True, "training")
    val_files = load_decathlon_datalist(datasets, True, "validation")
    train_ds = CacheDataset(
        data=datalist,
        transform=train_transforms,
        cache_num=24,
        cache_rate=1.0,
        num_workers=8,
    )
    train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=args.b, shuffle=True)
    val_ds = CacheDataset(
        data=val_files, transform=val_transforms, cache_num=2, cache_rate=1.0, num_workers=0
    )
    val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)

    set_track_meta(False)

    return train_loader, val_loader, train_transforms, val_transforms, datalist, val_files


def cka_loss(gram_featureA, gram_featureB):

    scaled_hsic = torch.dot(torch.flatten(gram_featureA),torch.flatten(gram_featureB))
    normalization_x = gram_featureA.norm()
    normalization_y = gram_featureB.norm()
    return scaled_hsic / (normalization_x * normalization_y)


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)



@torch.no_grad()
def make_grid(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: int = 0,
    **kwargs
) -> torch.Tensor:
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if "range" in kwargs.keys():
        warning = "range will be deprecated, please use value_range instead."
        warnings.warn(warning)
        value_range = kwargs["range"]

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if value_range is not None:
            assert isinstance(value_range, tuple), \
                "value_range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, low, high):
            img.clamp(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            # Tensor.copy_() is a valid method but seems to be missing from the stubs
            # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.copy_
            grid.narrow(1, y * height + padding, height - padding).narrow(  # type: ignore[attr-defined]
                2, x * width + padding, width - padding
            ).copy_(tensor[k])
            k = k + 1
    return grid


@torch.no_grad()
def save_image(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    fp: Union[Text, pathlib.Path, BinaryIO],
    format: Optional[str] = None,
    **kwargs
) -> None:
    """
    Save a given Tensor into an image file.
    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """

    grid = make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)
    

def create_logger(log_dir, phase='train'):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(time_str, phase)
    final_log_file = os.path.join(log_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger


def set_log_dir(root_dir, exp_name):
    path_dict = {}
    os.makedirs(root_dir, exist_ok=True)

    # set log path
    exp_path = os.path.join(root_dir, exp_name)
    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    prefix = exp_path + '_' + timestamp
    os.makedirs(prefix)
    path_dict['prefix'] = prefix

    # set checkpoint path
    ckpt_path = os.path.join(prefix, 'Model')
    os.makedirs(ckpt_path)
    path_dict['ckpt_path'] = ckpt_path

    log_path = os.path.join(prefix, 'Log')
    os.makedirs(log_path)
    path_dict['log_path'] = log_path

    # set sample image path for fid calculation
    sample_path = os.path.join(prefix, 'Samples', 'train')
    os.makedirs(sample_path)
    path_dict['train_sample_path'] = sample_path

    # set sample image path for fid calculation
    sample_path = os.path.join(prefix, 'Samples', 'valid')
    os.makedirs(sample_path)
    path_dict['valid_sample_path'] = sample_path

    # set sample image path for fid calculation
    sample_path = os.path.join(prefix, 'Samples', 'test')
    os.makedirs(sample_path)
    path_dict['test_sample_path'] = sample_path
    return path_dict


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth'):
    # torch.save(states, os.path.join(output_dir, filename))
    if is_best:
        torch.save(states, os.path.join(output_dir, 'checkpoint_best.pth'))

from torch.optim import Optimizer
class AdamW(Optimizer):
    """ Implements Adam algorithm with weight decay fix.
    Parameters:
        lr (float): learning rate. Default 1e-3.
        betas (tuple of 2 floats): Adams beta parameters (b1, b2). Default: (0.9, 0.999)
        eps (float): Adams epsilon. Default: 1e-6
        weight_decay (float): Weight decay. Default: 0.0
        correct_bias (bool): can be set to False to avoid correcting bias in Adam (e.g. like in Bert TF repository). Default True.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True
    ) -> None:
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = {
            "lr": lr, "betas": betas, "eps": eps,
            "weight_decay": weight_decay, "correct_bias": correct_bias
        }
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None) -> Optional[Callable]:
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, "
                        "please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(1.0 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1.0 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                step_size = group['lr']
                if group['correct_bias']:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state['step']
                    bias_correction2 = 1.0 - beta2 ** state['step']
                    step_size = step_size * math.sqrt(
                        bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group['weight_decay'] > 0.0:
                    p.data.add_(-group['lr'] * group['weight_decay'], p.data)

        return loss


class RunningStats:
    def __init__(self, WIN_SIZE):
        self.mean = 0
        self.run_var = 0
        self.WIN_SIZE = WIN_SIZE

        self.window = collections.deque(maxlen=WIN_SIZE)

    def clear(self):
        self.window.clear()
        self.mean = 0
        self.run_var = 0

    def is_full(self):
        return len(self.window) == self.WIN_SIZE

    def push(self, x):

        if len(self.window) == self.WIN_SIZE:
            # Adjusting variance
            x_removed = self.window.popleft()
            self.window.append(x)
            old_m = self.mean
            self.mean += (x - x_removed) / self.WIN_SIZE
            self.run_var += (x + x_removed - old_m - self.mean) * (x - x_removed)
        else:
            # Calculating first variance
            self.window.append(x)
            delta = x - self.mean
            self.mean += delta / len(self.window)
            self.run_var += delta * (x - self.mean)

    def get_mean(self):
        return self.mean if len(self.window) else 0.0

    def get_var(self):
        return self.run_var / len(self.window) if len(self.window) > 1 else 0.0

    def get_std(self):
        return math.sqrt(self.get_var())

    def get_all(self):
        return list(self.window)

    def __str__(self):
        return "Current window values: {}".format(list(self.window))

def iou(outputs: np.array, labels: np.array):
    
    SMOOTH = 1e-6
    intersection = (outputs & labels).sum((1, 2))  # & 按位与运算符
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    return iou.mean()

def iou_N_P(outputs_p: np.array, labels_p: np.array, outputs_n: np.array, labels_n: np.array):
    SMOOTH = 1e-6
    intersection = (((outputs_p + outputs_n)/2) & ((labels_p + labels_n)/2)).sum((1, 2))  # & 按位与运算符
    union = (outputs_p | labels_p).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    return iou.mean()

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target

class DiceCoeff_N_P(Function):
    """Dice coeff for individual examples"""

    def forward(self, input_p, target_p, input_n, target_n):
        self.save_for_backward(input_p, target_p)
        eps = 0.0001
        self.inter = torch.dot((input_p * 0.5 + input_n * 0.5).view(-1), ((target_p * 0.5 + target_n * 0.5)).view(-1))
        self.union = torch.sum(input_p) + torch.sum(target_p) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

class DiceCoeff_N_P_2and3(Function):
    """Dice coeff for individual examples"""

    def forward(self, input_p, target_p, input_n, target_n):
        self.save_for_backward(input_p, target_p)
        eps = 0.0001
        self.inter = torch.dot((input_p * 0.5 + input_n * 0.5).view(-1), ((target_p * 0.5 + target_n * 0.5)).view(-1))
        self.union = torch.sum(input_p) + torch.sum(target_p) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).to(device = input.device).zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

def dice_coeff_N_P(input_p, target_p, input_n, target_n):
    """Dice coeff for batches"""
    if input_p.is_cuda:
        s = torch.FloatTensor(1).to(device = input_p.device).zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input_p, target_p, input_n, target_n)):
        s = s + DiceCoeff_N_P().forward(c[0], c[1], c[2], c[3])

    return s / (i + 1)

def dice_coeff_N_P_2and3(input_p, target_p, input_n, target_n):
    """Dice coeff for batches"""
    if input_p.is_cuda:
        s = torch.FloatTensor(1).to(device = input_p.device).zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input_p, target_p, input_n, target_n)):
        s = s + DiceCoeff_N_P().forward(c[0], c[1], c[2], c[3])

    return s / (i + 1)

'''parameter'''
def para_image(w, h=None, img = None, mode = 'multi', seg = None, sd=None, batch=None,
          fft = False, channels=None, init = None):
    h = h or w
    batch = batch or 1
    ch = channels or 3
    shape = [batch, ch, h, w]
    param_f = fft_image if fft else pixel_image
    if init is not None:
        param_f = init_image
        params, maps_f = param_f(init)
    else:
        params, maps_f = param_f(shape, sd=sd)
    if mode == 'multi':
        output = to_valid_out(maps_f,img,seg)
    elif mode == 'seg':
        output = gene_out(maps_f,img)
    elif mode == 'raw':
        output = raw_out(maps_f,img)
    return params, output

def to_valid_out(maps_f,img,seg): #multi-rater
    def inner():
        maps = maps_f()
        maps = maps.to(device = img.device)
        maps = torch.nn.Softmax(dim = 1)(maps)
        final_seg = torch.multiply(seg,maps).sum(dim = 1, keepdim = True)
        return torch.cat((img,final_seg),1)
        # return torch.cat((img,maps),1)
    return inner

def gene_out(maps_f,img): #pure seg
    def inner():
        maps = maps_f()
        maps = maps.to(device = img.device)
        # maps = torch.nn.Sigmoid()(maps)
        return torch.cat((img,maps),1)
        # return torch.cat((img,maps),1)
    return inner

def raw_out(maps_f,img): #raw
    def inner():
        maps = maps_f()
        maps = maps.to(device = img.device)
        # maps = torch.nn.Sigmoid()(maps)
        return maps
        # return torch.cat((img,maps),1)
    return inner    


class CompositeActivation(torch.nn.Module):

    def forward(self, x):
        x = torch.atan(x)
        return torch.cat([x/0.67, (x*x)/0.6], 1)
        # return x


def cppn(args, size, img = None, seg = None, batch=None, num_output_channels=1, num_hidden_channels=128, num_layers=8,
         activation_fn=CompositeActivation, normalize=False, device = "cuda:0"):

    r = 3 ** 0.5

    coord_range = torch.linspace(-r, r, size)
    x = coord_range.view(-1, 1).repeat(1, coord_range.size(0))
    y = coord_range.view(1, -1).repeat(coord_range.size(0), 1)

    input_tensor = torch.stack([x, y], dim=0).unsqueeze(0).repeat(batch,1,1,1).to(device)

    layers = []
    kernel_size = 1
    for i in range(num_layers):
        out_c = num_hidden_channels
        in_c = out_c * 2 # * 2 for composite activation
        if i == 0:
            in_c = 2
        if i == num_layers - 1:
            out_c = num_output_channels
        layers.append(('conv{}'.format(i), torch.nn.Conv2d(in_c, out_c, kernel_size)))
        if normalize:
            layers.append(('norm{}'.format(i), torch.nn.InstanceNorm2d(out_c)))
        if i < num_layers - 1:
            layers.append(('actv{}'.format(i), activation_fn()))
        else:
            layers.append(('output', torch.nn.Sigmoid()))

    # Initialize model
    net = torch.nn.Sequential(OrderedDict(layers)).to(device)
    # Initialize weights
    def weights_init(module):
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.normal_(module.weight, 0, np.sqrt(1/module.in_channels))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    net.apply(weights_init)
    # Set last conv2d layer's weights to 0
    torch.nn.init.zeros_(dict(net.named_children())['conv{}'.format(num_layers - 1)].weight)
    outimg = raw_out(lambda: net(input_tensor),img) if args.netype == 'raw' else to_valid_out(lambda: net(input_tensor),img,seg)
    return net.parameters(), outimg

def get_siren(args):
    wrapper = get_network(args, 'siren', use_gpu=args.gpu, gpu_device=torch.device('cuda', args.gpu_device), distribution = args.distributed)
    '''load init weights'''
    checkpoint = torch.load('./logs/siren_train_init_2022_08_19_21_00_16/Model/checkpoint_best.pth')
    wrapper.load_state_dict(checkpoint['state_dict'],strict=False)
    '''end'''

    '''load prompt'''
    checkpoint = torch.load('./logs/vae_standard_refuge1_2022_08_21_17_56_49/Model/checkpoint500')
    vae = get_network(args, 'vae', use_gpu=args.gpu, gpu_device=torch.device('cuda', args.gpu_device), distribution = args.distributed)
    vae.load_state_dict(checkpoint['state_dict'],strict=False)
    '''end'''

    return wrapper, vae


def siren(args, wrapper, vae, img = None, seg = None, batch=None, num_output_channels=1, num_hidden_channels=128, num_layers=8,
         activation_fn=CompositeActivation, normalize=False, device = "cuda:0"):
    vae_img = torchvision.transforms.Resize(64)(img)
    latent = vae.encoder(vae_img).view(-1).detach()
    outimg = raw_out(lambda: wrapper(latent = latent),img) if args.netype == 'raw' else to_valid_out(lambda: wrapper(latent = latent),img,seg)
    # img = torch.randn(1, 3, 256, 256)
    # loss = wrapper(img)
    # loss.backward()

    # # after much training ...
    # # simply invoke the wrapper without passing in anything

    # pred_img = wrapper() # (1, 3, 256, 256)
    return wrapper.parameters(), outimg
        

'''adversary'''
def render_vis(
    args,
    model,
    objective_f,
    real_img,
    param_f=None,
    optimizer=None,
    transforms=None,
    thresholds=(256,),
    verbose=True,
    preprocess=True,
    progress=True,
    show_image=True,
    save_image=False,
    image_name=None,
    show_inline=False,
    fixed_image_size=None,
    label = 1,
    raw_img = None,
    prompt = None
):
    if label == 1:
        sign = 1
    elif label == 0:
        sign = -1
    else:
        print('label is wrong, label is',label)
    if args.reverse:
        sign = -sign
    if args.multilayer:
        sign = 1

    '''prepare'''
    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y, %H:%M:%S")

    netD, optD = pre_d()
    '''end'''

    if param_f is None:
        param_f = lambda: param.image(128)
    # param_f is a function that should return two things
    # params - parameters to update, which we pass to the optimizer
    # image_f - a function that returns an image as a tensor
    params, image_f = param_f()
    
    if optimizer is None:
        optimizer = lambda params: torch.optim.Adam(params, lr=5e-1)
    optimizer = optimizer(params)

    if transforms is None:
        transforms = []
    transforms = transforms.copy()

    # Upsample images smaller than 224
    image_shape = image_f().shape

    if fixed_image_size is not None:
        new_size = fixed_image_size
    elif image_shape[2] < 224 or image_shape[3] < 224:
        new_size = 224
    else:
        new_size = None
    if new_size:
        transforms.append(
            torch.nn.Upsample(size=new_size, mode="bilinear", align_corners=True)
        )

    transform_f = transform.compose(transforms)

    hook = hook_model(model, image_f)
    objective_f = objectives.as_objective(objective_f)

    if verbose:
        model(transform_f(image_f()))
        print("Initial loss of ad: {:.3f}".format(objective_f(hook)))

    images = []
    try:
        for i in tqdm(range(1, max(thresholds) + 1), disable=(not progress)):
            optimizer.zero_grad()
            try:
                model(transform_f(image_f()))
            except RuntimeError as ex:
                if i == 1:
                    # Only display the warning message
                    # on the first iteration, no need to do that
                    # every iteration
                    warnings.warn(
                        "Some layers could not be computed because the size of the "
                        "image is not big enough. It is fine, as long as the non"
                        "computed layers are not used in the objective function"
                        f"(exception details: '{ex}')"
                    )
            if args.disc:
                '''dom loss part'''
                # content_img = raw_img
                # style_img = raw_img
                # precpt_loss = run_precpt(cnn, cnn_normalization_mean, cnn_normalization_std, content_img, style_img, transform_f(image_f()))
                for p in netD.parameters():
                    p.requires_grad = True
                for _ in range(args.drec):
                    netD.zero_grad()
                    real = real_img
                    fake = image_f()
                    # for _ in range(6):
                    #     errD, D_x, D_G_z1 = update_d(args, netD, optD, real, fake)

                    # label = torch.full((args.b,), 1., dtype=torch.float, device=device)
                    # label.fill_(1.)
                    # output = netD(fake).view(-1)
                    # errG = nn.BCELoss()(output, label)
                    # D_G_z2 = output.mean().item()
                    # dom_loss = err
                    one = torch.tensor(1, dtype=torch.float)
                    mone = one * -1
                    one = one.cuda(args.gpu_device)
                    mone = mone.cuda(args.gpu_device)

                    d_loss_real = netD(real)
                    d_loss_real = d_loss_real.mean()
                    d_loss_real.backward(mone)

                    d_loss_fake = netD(fake)
                    d_loss_fake = d_loss_fake.mean()
                    d_loss_fake.backward(one)

                    # Train with gradient penalty
                    gradient_penalty = calculate_gradient_penalty(netD, real.data, fake.data)
                    gradient_penalty.backward()


                    d_loss = d_loss_fake - d_loss_real + gradient_penalty
                    Wasserstein_D = d_loss_real - d_loss_fake
                    optD.step()

                # Generator update
                for p in netD.parameters():
                    p.requires_grad = False  # to avoid computation

                fake_images = image_f()
                g_loss = netD(fake_images)
                g_loss = -g_loss.mean()
                dom_loss = g_loss
                g_cost = -g_loss

                if i% 5 == 0:
                    print(f' loss_fake: {d_loss_fake}, loss_real: {d_loss_real}')
                    print(f'Generator g_loss: {g_loss}')
                '''end'''



            '''ssim loss'''

            '''end'''

            if args.disc:
                loss = sign * objective_f(hook) + args.pw * dom_loss
                # loss = args.pw * dom_loss
            else:
                loss = sign * objective_f(hook)
                # loss = args.pw * dom_loss

            loss.backward()

            # #video the images
            # if i % 5 == 0:
            #     print('1')
            #     image_name = image_name[0].split('\\')[-1].split('.')[0] + '_' + str(i) + '.png'
            #     img_path = os.path.join(args.path_helper['sample_path'], str(image_name))
            #     export(image_f(), img_path)
            # #end
            # if i % 50 == 0:
            #     print('Loss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
            #       % (errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            optimizer.step()
            if i in thresholds:
                image = tensor_to_img_array(image_f())
                # if verbose:
                #     print("Loss at step {}: {:.3f}".format(i, objective_f(hook)))
                if save_image:
                    na = image_name[0].split('\\')[-1].split('.')[0] + '_' + str(i) + '.png'
                    na = date_time + na
                    outpath = args.quickcheck if args.quickcheck else args.path_helper['sample_path']
                    img_path = os.path.join(outpath, str(na))
                    export(image_f(), img_path)
                
                images.append(image)
    except KeyboardInterrupt:
        print("Interrupted optimization at step {:d}.".format(i))
        if verbose:
            print("Loss at step {}: {:.3f}".format(i, objective_f(hook)))
        images.append(tensor_to_img_array(image_f()))

    if save_image:
        na = image_name[0].split('\\')[-1].split('.')[0] + '.png'
        na = date_time + na
        outpath = args.quickcheck if args.quickcheck else args.path_helper['sample_path']
        img_path = os.path.join(outpath, str(na))
        export(image_f(), img_path)
    if show_inline:
        show(tensor_to_img_array(image_f()))
    elif show_image:
        view(image_f())
    return image_f()


def tensor_to_img_array(tensor):
    image = tensor.cpu().detach().numpy()
    image = np.transpose(image, [0, 2, 3, 1])
    return image


def view(tensor):
    image = tensor_to_img_array(tensor)
    assert len(image.shape) in [
        3,
        4,
    ], "Image should have 3 or 4 dimensions, invalid image shape {}".format(image.shape)
    # Change dtype for PIL.Image
    image = (image * 255).astype(np.uint8)
    if len(image.shape) == 4:
        image = np.concatenate(image, axis=1)
    Image.fromarray(image).show()


def export(tensor, img_path=None):
    # image_name = image_name or "image.jpg"
    c = tensor.size(1)
    # if c == 7:
    #     for i in range(c):
    #         w_map = tensor[:,i,:,:].unsqueeze(1)
    #         w_map = tensor_to_img_array(w_map).squeeze()
    #         w_map = (w_map * 255).astype(np.uint8)
    #         image_name = image_name[0].split('/')[-1].split('.')[0] + str(i)+ '.png'
    #         wheat = sns.heatmap(w_map,cmap='coolwarm')
    #         figure = wheat.get_figure()    
    #         figure.savefig ('./fft_maps/weightheatmap/'+str(image_name), dpi=400)
    #         figure = 0
    # else:
    if c == 3:
        vutils.save_image(tensor, fp = img_path)
    else:
        image = tensor[:,0:3,:,:]
        w_map = tensor[:,-1,:,:].unsqueeze(1)
        image = tensor_to_img_array(image)
        w_map = 1 - tensor_to_img_array(w_map).squeeze()
        # w_map[w_map==1] = 0
        assert len(image.shape) in [
            3,
            4,
        ], "Image should have 3 or 4 dimensions, invalid image shape {}".format(image.shape)
        # Change dtype for PIL.Image
        image = (image * 255).astype(np.uint8)
        w_map = (w_map * 255).astype(np.uint8)

        Image.fromarray(w_map,'L').save(img_path)


class ModuleHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = None
        self.features = None


    def hook_fn(self, module, input, output):
        self.module = module
        self.features = output


    def close(self):
        self.hook.remove()


def hook_model(model, image_f):
    features = OrderedDict()
    # recursive hooking function
    def hook_layers(net, prefix=[]):
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():
                if layer is None:
                    # e.g. GoogLeNet's aux1 and aux2 layers
                    continue
                features["_".join(prefix + [name])] = ModuleHook(layer)
                hook_layers(layer, prefix=prefix + [name])

    hook_layers(model)

    def hook(layer):
        if layer == "input":
            out = image_f()
        elif layer == "labels":
            out = list(features.values())[-1].features
        else:
            assert layer in features, f"Invalid layer {layer}. Retrieve the list of layers with `lucent.modelzoo.util.get_model_layers(model)`."
            out = features[layer].features
        assert out is not None, "There are no saved feature maps. Make sure to put the model in eval mode, like so: `model.to(device).eval()`. See README for example."
        return out

    return hook


def vis_image(imgs, pred_masks, gt_masks, save_path, reverse=False, points=None, box=None):
    b, c, h, w = pred_masks.size()
    dev = pred_masks.get_device()
    row_num = min(b, 4)

    if torch.max(pred_masks) > 1 or torch.min(pred_masks) < 0:
        pred_masks = torch.sigmoid(pred_masks)

    # if reverse == True:
    #     pred_masks = 1 - pred_masks
    #     gt_masks = 1 - gt_masks
    # if c == 2:
    #     pred_disc, pred_cup = pred_masks[:,0,:,:].unsqueeze(1).expand(b,3,h,w), pred_masks[:,1,:,:].unsqueeze(1).expand(b,3,h,w)
    #     gt_disc, gt_cup = gt_masks[:,0,:,:].unsqueeze(1).expand(b,3,h,w), gt_masks[:,1,:,:].unsqueeze(1).expand(b,3,h,w)
    #     tup = (imgs[:row_num,:,:,:],pred_disc[:row_num,:,:,:], pred_cup[:row_num,:,:,:], gt_disc[:row_num,:,:,:], gt_cup[:row_num,:,:,:])
    #     # compose = torch.cat((imgs[:row_num,:,:,:],pred_disc[:row_num,:,:,:], pred_cup[:row_num,:,:,:], gt_disc[:row_num,:,:,:], gt_cup[:row_num,:,:,:]),0)
    #     compose = torch.cat((pred_disc[:row_num,:,:,:], pred_cup[:row_num,:,:,:], gt_disc[:row_num,:,:,:], gt_cup[:row_num,:,:,:]),0)
    #     vutils.save_image(compose, fp = save_path, nrow = row_num, padding = 10)
    # else:
    imgs = torchvision.transforms.Resize((h, w))(imgs)
    if imgs.size(1) == 1:
        imgs = imgs[:, 0, :, :].unsqueeze(1).expand(b, 3, h, w)
    pred_masks = pred_masks[:, 0, :, :].unsqueeze(1).expand(b, 3, h, w)
    gt_masks = gt_masks[:, 0, :, :].unsqueeze(1).expand(b, 3, h, w)

    if points != None:
        for i in range(b):
            if args.thd:
                p = np.round(points.cpu() / args.roi_size * args.out_size).to(dtype=torch.int)
            else:
                p = np.round(points.cpu() / args.image_size * args.out_size).to(dtype=torch.int)
            # gt_masks[i,:,points[i,0]-5:points[i,0]+5,points[i,1]-5:points[i,1]+5] = torch.Tensor([255, 0, 0]).to(dtype = torch.float32, device = torch.device('cuda:' + str(dev)))
            gt_masks[i, 0, p[i, 0] - 5:p[i, 0] + 5, p[i, 1] - 5:p[i, 1] + 5] = 0.5
            gt_masks[i, 1, p[i, 0] - 5:p[i, 0] + 5, p[i, 1] - 5:p[i, 1] + 5] = 0.1
            gt_masks[i, 2, p[i, 0] - 5:p[i, 0] + 5, p[i, 1] - 5:p[i, 1] + 5] = 0.4

    if box != None and b == 1:
        box = box.squeeze(0).squeeze(0)
        points1 = torch.tensor(np.array([box[0], box[1]]))
        points2 = torch.tensor(np.array([box[2], box[3]]))
        for i in range(b):
            p1 = np.round(points1.cpu() / args.image_size * args.out_size).to(dtype=torch.int)
            p2 = np.round(points2.cpu() / args.image_size * args.out_size).to(dtype=torch.int)
            p1[0] = 2 if p1[0] - 2 <= 0 else p1[0]
            p1[1] = 2 if p1[1] - 2 <= 0 else p1[1]
            p2[0] = args.out_size - 1 if p2[0] + 2 >= args.out_size - 1 else p2[0]
            p2[1] = args.out_size - 1 if p2[1] + 2 >= args.out_size - 1 else p2[1]

            gt_masks[i, 0, p1[1] - 2:p2[1] + 2, p1[0] - 2:p1[0] + 2] = 0.5  # 左
            gt_masks[i, 0, p1[1] - 2:p2[1] + 2, p2[0] - 2:p2[0] + 2] = 0.5  # 右
            gt_masks[i, 0, p1[1] - 2:p1[1] + 2, p1[0] - 2:p2[0] + 2] = 0.5  # 上
            gt_masks[i, 0, p2[1] - 2:p2[1] + 2, p1[0] - 2:p2[0] + 2] = 0.5  # 下

    tup = (imgs[:row_num, :, :, :], pred_masks[:row_num, :, :, :], gt_masks[:row_num, :, :, :])
    # compose = torch.cat((imgs[:row_num,:,:,:],pred_disc[:row_num,:,:,:], pred_cup[:row_num,:,:,:], gt_disc[:row_num,:,:,:], gt_cup[:row_num,:,:,:]),0)
    compose = torch.cat(tup, 0)
    vutils.save_image(compose, fp=save_path, nrow=row_num, padding=10)

    return

def eval_seg(pred,true_mask_p,threshold):
    '''
    threshold: a int or a tuple of int
    masks: [b,2,h,w]
    pred: [b,2,h,w]
    '''
    b, c, h, w = pred.size()
    if c == 2:
        iou_d, iou_c, disc_dice, cup_dice = 0,0,0,0
        for th in threshold:

            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            disc_pred = vpred_cpu[:,0,:,:].numpy().astype('int32')
            cup_pred = vpred_cpu[:,1,:,:].numpy().astype('int32')

            disc_mask = gt_vmask_p [:,0,:,:].squeeze(1).cpu().numpy().astype('int32')
            cup_mask = gt_vmask_p [:, 1, :, :].squeeze(1).cpu().numpy().astype('int32')
    
            '''iou for numpy'''
            iou_d += iou(disc_pred,disc_mask)
            iou_c += iou(cup_pred,cup_mask)

            '''dice for torch'''
            disc_dice += dice_coeff(vpred[:,0,:,:], gt_vmask_p[:,0,:,:]).item()
            cup_dice += dice_coeff(vpred[:,1,:,:], gt_vmask_p[:,1,:,:]).item()
            
        return iou_d / len(threshold), iou_c / len(threshold), disc_dice / len(threshold), cup_dice / len(threshold)
    else:
        eiou, edice = 0,0
        for th in threshold:

            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            disc_pred = vpred_cpu[:,0,:,:].numpy().astype('int32')

            disc_mask = gt_vmask_p [:,0,:,:].squeeze(1).cpu().numpy().astype('int32')
    
            '''iou for numpy'''
            eiou += iou(disc_pred,disc_mask)

            '''dice for torch'''
            edice += dice_coeff(vpred[:,0,:,:], gt_vmask_p[:,0,:,:]).item()
            
        return eiou / len(threshold), edice / len(threshold)

def batched_mask_to_box(masks: torch.Tensor) -> torch.Tensor:
    """
    Calculates boxes in XYXY format around masks. Return [0,0,0,0] for
    an empty mask. For input shape C1xC2x...xHxW, the output shape is C1xC2x...x4.
    """
    # torch.max below raises an error on empty inputs, just skip in this case
    if torch.numel(masks) == 0:  # 获取tensor中一共有多少元素
        return torch.zeros(*masks.shape[:-2], 4, device=masks.device)

    # Normalize shape to CxHxW
    shape = masks.shape
    h, w = shape[-2:]
    if len(shape) > 2:
        masks = masks.flatten(0, -3)
    else:
        masks = masks.unsqueeze(0)

    # Get top and bottom edges
    in_height, _ = torch.max(masks, dim=-1)
    in_height_coords = in_height * torch.arange(h, device=in_height.device)[None, :]
    bottom_edges, _ = torch.max(in_height_coords, dim=-1)
    in_height_coords = in_height_coords + h * (~in_height)
    top_edges, _ = torch.min(in_height_coords, dim=-1)

    # Get left and right edges
    in_width, _ = torch.max(masks, dim=-2)
    in_width_coords = in_width * torch.arange(w, device=in_width.device)[None, :]
    right_edges, _ = torch.max(in_width_coords, dim=-1)
    in_width_coords = in_width_coords + w * (~in_width)
    left_edges, _ = torch.min(in_width_coords, dim=-1)

    # If the mask is empty the right edge will be to the left of the left edge.
    # Replace these boxes with [0, 0, 0, 0]
    empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
    out = torch.stack([left_edges, top_edges, right_edges, bottom_edges], dim=-1)
    out = out * (~empty_filter).unsqueeze(-1)

    # Return to original shape
    if len(shape) > 2:
        out = out.reshape(*shape[:-2], 4)
    else:
        out = out[0]

    return out

def mask_find_bboxs(mask):
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)  # connectivity参数的默认值为8
    stats = stats[stats[:, 4].argsort()]
    return stats[:-1,:4] # stats[:-1]  # 排除最外层的连通图


def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
    """
    Compute the output size given input size and target long side length.
    """
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)


def apply_boxes(bboxes: np.ndarray, ori_img_size: Tuple[int, ...], target_img_size: Tuple[int, ...]) -> np.ndarray:

    for i in range(len(bboxes)):
        bboxes[i][0] = target_img_size[1] / ori_img_size[1] * bboxes[i][0]
        bboxes[i][1] = target_img_size[0] / ori_img_size[0] * bboxes[i][1]
        bboxes[i][2] = target_img_size[1] / ori_img_size[1] * bboxes[i][2]
        bboxes[i][3] = target_img_size[0] / ori_img_size[0] * bboxes[i][3]

    return bboxes

def computeIoU(box1, box2):
    x1, y1, x2, y2 = box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]  # box1的左上角坐标、右下角坐标
    x3, y3, x4, y4 = box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]  # box1的左上角坐标、右下角坐标

    # 计算交集的坐标
    x_inter1 = max(x1, x3)  # union的左上角x
    y_inter1 = max(y1, y3)  # union的左上角y
    x_inter2 = min(x2, x4)  # union的右下角x
    y_inter2 = min(y2, y4)  # union的右下角y

    # 计算交集部分面积，因为图像是像素点，所以计算图像的长度需要加一
    # 比如有两个像素点(0,0)、(1,0)，那么图像的长度是1-0+1=2，而不是1-0=1
    interArea = max(0, x_inter2 - x_inter1 + 1) * max(0, y_inter2 - y_inter1 + 1)

    # 分别计算两个box的面积
    area_box1 = (x2 - x1 + 1) * (y2 - y1 + 1)
    area_box2 = (x4 - x3 + 1) * (y4 - y3 + 1)

    # 计算IOU，交集比并集，并集面积=两个矩形框面积和-交集面积
    iou = interArea / (area_box1 + area_box2 - interArea)

    return iou

def getIoU(gt_bboxes, bboxes):
    # ious = {
    #     (gt_id, pre_id): computeIoU(gt_bboxes[gt_id], bboxes[pre_id]) for gt_id in range(len(gt_bboxes)) for pre_id in range(len(bboxes))
    # }
    iou = numpy.empty([len(bboxes), len(gt_bboxes)], dtype = float, order = 'C')
    for pre_id in range(len(bboxes)):
        for gt_id in range(len(gt_bboxes)):
            iou[pre_id, gt_id] = computeIoU(bboxes[pre_id], gt_bboxes[gt_id])
    ious = [[iou]]
    # ious = [[ious[gt_id, catId] for catId in catIds] for imgId in p.imgIds]
    return ious

def deal_with_bboxes(bboxes, mode="groundtruth"):
    _instances = collections.defaultdict(list)
    dicts = []
    if mode=="groundtruth":
        for i in range(len(bboxes)):
            dict = {'id': i+1,
                    'image_id': 0,
                    'bbox': bboxes[i],
                    'area': bboxes[i][2] * bboxes[i][3],
                    'iscrowd': 0,
                    'category_id': 0,
                    'ignore': 0}
            dicts.append(dict)
    elif mode== "detection":
        for i in range(len(bboxes)):
            dict = {'image_id': 0,
                    'category_id': 0,
                    'bbox': bboxes[i],
                    'score': 1.0,
                    'segmentation':[bboxes[i][0], bboxes[i][1], bboxes[i][0], bboxes[i][1] + bboxes[i][3],
                                    bboxes[i][0] + bboxes[i][2], bboxes[i][1] + bboxes[i][3],
                                    bboxes[i][0] + bboxes[i][2], bboxes[i][1]
                                    ],
                    'area': bboxes[i][2] * bboxes[i][3],
                    'id': i + 1,
                    'iscrowd': 0}
            dicts.append(dict)
    _instances[(0,0)] = dicts

    def convert_instances_to_cpp(instances, is_det=False):
        # Convert annotations for a list of instances in an image to a format that's fast
        # to access in C++
        instances_cpp = []
        for instance in instances:
            instance_cpp = _C.InstanceAnnotation(
                int(instance["id"]),
                instance["score"] if is_det else instance.get("score", 0.0),
                instance["area"],
                bool(instance.get("iscrowd", 0)),
                bool(instance.get("ignore", 0)),
            )
            instances_cpp.append(instance_cpp)
        return instances_cpp

    instances = [
            [convert_instances_to_cpp(_instances[0, 0])]
        ]
    return instances

def evaluate(params,ious, gt_instances, pre_instances):
    _evalImgs_cpp = _C.COCOevalEvaluateImages(
        params.areaRng, params.maxDets[-1], params.iouThrs, ious, gt_instances, pre_instances
    )
    _paramsEval = copy.deepcopy(params)

    return _evalImgs_cpp, _paramsEval


def accumulate(_evalImgs_cpp, _paramsEval):
    logger = logging.getLogger(__name__)

    logger.info("Accumulating evaluation results...")
    tic = time.time()
    eval_results = _C.COCOevalAccumulate(_paramsEval, _evalImgs_cpp)

    # recall is num_iou_thresholds X num_categories X num_area_ranges X num_max_detections
    eval_results["recall"] = np.array(eval_results["recall"]).reshape(
        eval_results["counts"][:1] + eval_results["counts"][2:]
    )

    # precision and scores are num_iou_thresholds X num_recall_thresholds X num_categories X
    # num_area_ranges X num_max_detections
    eval_results["precision"] = np.array(eval_results["precision"]).reshape(eval_results["counts"])
    eval_results["scores"] = np.array(eval_results["scores"]).reshape(eval_results["counts"])
    toc = time.time()
    logger.info("COCOeval_opt.accumulate() finished in {:0.2f} seconds.".format(toc - tic))

    return eval_results

def summarize(params, eval_results):
    '''
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    '''
    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100 ):
        p = params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap==1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = eval_results['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:,:,:,aind,mind]
        else:
            # dimension of recall: [TxKxAxM]
            s = eval_results['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:,:,aind,mind]
        if len(s[s>-1])==0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s>-1])
        print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
        return mean_s

    def _summarizeDets():
        stats = np.zeros((12,))
        stats[0] = _summarize(1)
        stats[1] = _summarize(1, iouThr=.5, maxDets=params.maxDets[2])
        stats[2] = _summarize(1, iouThr=.75, maxDets=params.maxDets[2])
        stats[3] = _summarize(1, areaRng='small', maxDets=params.maxDets[2])
        stats[4] = _summarize(1, areaRng='medium', maxDets=params.maxDets[2])
        stats[5] = _summarize(1, areaRng='large', maxDets=params.maxDets[2])
        stats[6] = _summarize(0, maxDets=params.maxDets[0])
        stats[7] = _summarize(0, maxDets=params.maxDets[1])
        stats[8] = _summarize(0, maxDets=params.maxDets[2])
        stats[9] = _summarize(0, areaRng='small', maxDets=params.maxDets[2])
        stats[10] = _summarize(0, areaRng='medium', maxDets=params.maxDets[2])
        stats[11] = _summarize(0, areaRng='large', maxDets=params.maxDets[2])
        return stats

    def _summarizeKps():
        stats = np.zeros((10,))
        stats[0] = _summarize(1, maxDets=20)
        stats[1] = _summarize(1, maxDets=20, iouThr=.5)
        stats[2] = _summarize(1, maxDets=20, iouThr=.75)
        stats[3] = _summarize(1, maxDets=20, areaRng='medium')
        stats[4] = _summarize(1, maxDets=20, areaRng='large')
        stats[5] = _summarize(0, maxDets=20)
        stats[6] = _summarize(0, maxDets=20, iouThr=.5)
        stats[7] = _summarize(0, maxDets=20, iouThr=.75)
        stats[8] = _summarize(0, maxDets=20, areaRng='medium')
        stats[9] = _summarize(0, maxDets=20, areaRng='large')
        return stats

    if not eval_results:
        raise Exception('Please run accumulate() first')
    iouType = params.iouType
    if iouType == 'segm' or iouType == 'bbox':
        summarize = _summarizeDets
    elif iouType == 'keypoints':
        summarize = _summarizeKps
    stats = summarize()


def eval_seqg_ob(pred, mask_old, gt_bboxes, threshold, img_size):
    '''
    threshold: a int or a tuple of int
    masks: [b,2,h,w]
    pred: [b,2,h,w]
    '''
    logger = logging.getLogger(__name__)
    b, c, h, w = pred.size()
    params = Params(iouType="bbox")
    params.areaRng = [[0, 10000000000.0], [0, 1024], [1024, 9216], [9216, 10000000000.0]]
    params.iouThrs = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    params.maxDets = [10, 100, 300]
    params.catIds = [0]
    params.imgIds = [0]
    params.areaRngLbl = ['all', 'small', 'medium', 'large']
    params.iouType = 'bbox'
    params.recThrs = np.array([0, 0.01, 0.02, 0.03, 0.04, 0.05,0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13,
                      0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27,
                      0.28, 0.29, 0.3,  0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4,  0.41,
                      0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5,  0.51, 0.52, 0.53, 0.54, 0.55,
                      0.56, 0.57, 0.58, 0.59, 0.6,  0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69,
                      0.7,  0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8,  0.81, 0.82, 0.83,
                      0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9,  0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97,
                      0.98, 0.99, 1.0])
    params.useCats = 1
    params.useSegm = None
    eiou, edice = 0, 0
    # for th in threshold:
    #     data = {"masks": pred.flatten(0, 1)}
    #     data["masks"] = data["masks"] > th
    #     data["boxes"] = batched_mask_to_box(data["masks"])
    gt_bboxes = apply_boxes(gt_bboxes, img_size)
    gt_instances = deal_with_bboxes(gt_bboxes, mode="groundtruth")
    for th in threshold:
        print(th)
        prediction = (pred > th).float().cpu().numpy()
        for i in range(prediction.shape[0]):
            arr = np.zeros((256, 256, 3))
            arr[:, :, 0] = prediction[i][0]
            arr[:, :, 1] = prediction[i][0]
            arr[:, :, 2] = prediction[i][0]
            arr = arr.astype(np.uint8)
            arr_gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            bboxes = mask_find_bboxs(arr_gray)
            # ax = plt.axes()
            # plt.imshow(arr_gray, cmap='bone')
            # for j in bboxes:
            #     if j[2] - j[0] >= 2 and j[3] - j[1] >= 2:
            #         rect = patches.Rectangle((j[0], j[1]), j[2], j[3], linewidth=1, edgecolor='r', facecolor='g')
            #     else:
            #         rect = patches.Rectangle((j[0], j[1]), j[2], j[3], linewidth=1, edgecolor='r', facecolor='r')
            #     ax.add_patch(rect)
            # plt.show()
            ious = getIoU(gt_bboxes, bboxes)
            pre_instances = deal_with_bboxes(bboxes, mode="detection")

            _evalImgs_cpp, _paramsEval = evaluate(params,ious, gt_instances, pre_instances)

            eval_results = accumulate(_evalImgs_cpp, _paramsEval)

            summarize(params, eval_results)

            print()
    print()
    # data["masks"] = data["masks"] > 0
    # data["boxes"] = batched_mask_to_box(data["masks"])
    # del data["masks"]
    # return data


def eval_bboxes(pred, img_name, threshold, img_size):
    '''
    threshold: a int or a tuple of int
    masks: [b,2,h,w]
    pred: [b,2,h,w]
    '''
    save_csv_path = os.path.join("10028_prediction", "csv_file")
    save_vis_picture = os.path.join("10028_prediction", "vis")
    if not os.path.exists(save_csv_path):
        os.makedirs(save_csv_path)
    if not os.path.exists(save_vis_picture):
        os.makedirs(save_vis_picture)
    # plt.imshow(pred[0].permute(1, 2, 0).cpu().numpy(), cmap='bone')
    # plt.show()
    # prediction = (pred > th).float().cpu().numpy()
    prediction = pred.cpu().numpy()
    for i in range(prediction.shape[0]):
        arr = np.zeros((256, 256, 3))
        arr[:, :, 0] = prediction[i][0]
        arr[:, :, 1] = prediction[i][0]
        arr[:, :, 2] = prediction[i][0]
        arr = 255 - arr.astype(np.uint8)
        arr_gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        arr_gray = (arr_gray > 200).astype(np.uint8)
        bboxes = mask_find_bboxs(arr_gray)
        bboxes = apply_boxes(bboxes, (256, 256) ,img_size)
        columns = ["X-Coordinate", "Y-Coordinate", "Diameter"]
        title_csv = pd.DataFrame(columns=columns)
        title_csv.to_csv(save_csv_path + "/" + img_name +".csv", mode="a", index=False, header=1, encoding="utf-8")
        for j in range(len(bboxes)):
            save = pd.DataFrame({"X-Coordinate": bboxes[j][0] + bboxes[j][2] / 2,
                                 "Y-Coordinate": bboxes[j][1] + bboxes[j][3] / 2,
                                 "Diameter": [bboxes[j][3] if bboxes[j][2] >= bboxes[j][3] else bboxes[j][2]]})
            save.to_csv(save_csv_path + "/" + img_name +".csv", mode='a', header=False, index=False)
        ax = plt.axes()
        res = cv2.resize(arr_gray, img_size, interpolation=cv2.INTER_CUBIC)
        plt.imshow(res, cmap='bone')
        for j in bboxes:
            rect = patches.Rectangle((j[0], j[1]), j[2], j[3], linewidth=1, edgecolor='g', fill=False)
            ax.add_patch(rect)
            rect = patches.Rectangle((j[0] + j[2] / 2 - 10, j[1] + j[3] / 2 - 10), 20, 20, linewidth=1, edgecolor='r', fill=True, facecolor= "r")
            ax.add_patch(rect)
            # else:
            #     rect = patches.Rectangle((j[0], j[1]), j[2], j[3], linewidth=1, edgecolor='r', fill=False)
            # ax.add_patch(rect)
        # plt.show()
        plt.savefig(os.path.join(save_vis_picture, img_name + ".png"))
        plt.close()
        print()
    print()
    # data["masks"] = data["masks"] > 0
    # data["boxes"] = batched_mask_to_box(data["masks"])
    # del data["masks"]
    # return data

def data_normal(orign_data):
    d_min = orign_data.min()
    if d_min < 0:
        orign_data += torch.abs(d_min)
        d_min = orign_data.min()
    d_max = orign_data.max()
    dst = d_max - d_min
    norm_data = (orign_data - d_min).true_divide(dst)
    return norm_data

def eval_seg_N_P(pred, true_mask_p, threshold):
    '''
    threshold: a int or a tuple of int
    masks: [b,2,h,w]
    pred: [b,2,h,w]
    '''
    b, c, h, w = pred.size()
    eiou, edice = 0, 0
    # pred = torch.nn.functional.normalize(pred)
    # pred = data_normal(pred)  # 归一化到【0，1】
    pred = torch.sigmoid(pred)  # 归一化到【0，1】
    for th in threshold:
        gt_vmask_p = (true_mask_p >= th).float()
        gt_vmask_n = (true_mask_p < th).float()
        vpred_p = (pred >= th).float()
        vpred_n = (pred < th).float()

        dice3_pred = pred
        dice3_pred[dice3_pred < th] = 0

        vpred_p_cpu = vpred_p.cpu()
        disc_pred_p = vpred_p_cpu[:, 0, :, :].numpy().astype('int32')
        disc_mask_p = gt_vmask_p[:, 0, :, :].squeeze(1).cpu().numpy().astype('int32')

        '''iou for numpy'''
        eiou += iou(disc_pred_p, disc_mask_p)

        '''dice for torch'''
        # 以下都是在pred归一化后计算的
        dice1 = dice_coeff(vpred_p[:, 0, :, :], gt_vmask_p[:, 0, :, :]).item()  # 原dice
        dice2 = dice_coeff_N_P(vpred_p[:, 0, :, :], gt_vmask_p[:, 0, :, :],     # 按照像素true 和 false ，比例算dice（不可用，超过1）
                                vpred_n[:, 0, :, :], gt_vmask_n[:, 0, :, :]) .item()
        dice3_1 = dice_coeff(dice3_pred[:, 0, :, :], true_mask_p[:, 0, :, :]).item()     # 直接用pred值筛选后，计算dice
        # dice3 = dice_coeff_N_P_dice3(dice3_pred[:, 0, :, :], true_mask_p[:, 0, :, :]).item() # 直接用pred值筛选后，计算dice ，和dice3一样，方法不同，可以删掉
        dice2and3 = dice_coeff_N_P_2and3(vpred_p[:, 0, :, :], gt_vmask_p[:, 0, :, :],     # N和P的像素 ，比例算dice
                                vpred_n[:, 0, :, :], gt_vmask_n[:, 0, :, :]) .item()
        edice += dice1
        # print("p: ",dice1,  "p_n: ",dice2, "dice3_1", dice3_1,"dice2and3",dice2and3)

    return eiou / len(threshold), edice / len(threshold)

# @objectives.wrap_objective()
def dot_compare(layer, batch=1, cossim_pow=0):
  def inner(T):
    dot = (T(layer)[batch] * T(layer)[0]).sum()
    mag = torch.sqrt(torch.sum(T(layer)[0]**2))
    cossim = dot/(1e-6 + mag)
    return -dot * cossim ** cossim_pow
  return inner

def init_D(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def pre_d():
    netD = Discriminator(3).to(device)
    # netD.apply(init_D)
    beta1 = 0.5
    dis_lr = 0.00002
    optimizerD = optim.Adam(netD.parameters(), lr=dis_lr, betas=(beta1, 0.999))
    return netD, optimizerD

def update_d(args, netD, optimizerD, real, fake):
    criterion = nn.BCELoss()

    label = torch.full((args.b,), 1., dtype=torch.float, device=device)
    output = netD(real).view(-1)
    # Calculate loss on all-real batch
    errD_real = criterion(output, label)
    # Calculate gradients for D in backward pass
    errD_real.backward()
    D_x = output.mean().item()

    label.fill_(0.)
    # Classify all fake batch with D
    output = netD(fake.detach()).view(-1)
    # Calculate D's loss on the all-fake batch
    errD_fake = criterion(output, label)
    # Calculate the gradients for this batch, accumulated (summed) with previous gradients
    errD_fake.backward()
    D_G_z1 = output.mean().item()
    # Compute error of D as sum over the fake and the real batches
    errD = errD_real + errD_fake
    # Update D
    optimizerD.step()

    return errD, D_x, D_G_z1

def calculate_gradient_penalty(netD, real_images, fake_images):
    eta = torch.FloatTensor(args.b,1,1,1).uniform_(0,1)
    eta = eta.expand(args.b, real_images.size(1), real_images.size(2), real_images.size(3)).to(device = device)

    interpolated = (eta * real_images + ((1 - eta) * fake_images)).to(device = device)

    # define it to calculate gradient
    interpolated = Variable(interpolated, requires_grad=True)

    # calculate probability of interpolated examples
    prob_interpolated = netD(interpolated)

    # calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                            grad_outputs=torch.ones(
                                prob_interpolated.size()).to(device = device),
                            create_graph=True, retain_graph=True)[0]

    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
    return grad_penalty


def random_click(mask, point_labels = 1, inout = 1):
    indices = np.argwhere(mask == inout) # 所有label像素
    return indices[np.random.randint(len(indices))]  # 随机生成一个像素坐标


def generate_click_prompt(img, msk, pt_label = 1):
    # return: prompt, prompt mask
    pt_list = []
    msk_list = []
    b, c, h, w, d = msk.size()
    msk = msk[:,0,:,:,:]
    for i in range(d):
        pt_list_s = []
        msk_list_s = []
        for j in range(b):
            msk_s = msk[j,:,:,i]
            indices = torch.nonzero(msk_s)
            if indices.size(0) == 0:
                # generate a random array between [0-h, 0-h]:
                random_index = torch.randint(0, h, (2,)).to(device = msk.device)
                new_s = msk_s
            else:
                random_index = random.choice(indices)
                label = msk_s[random_index[0], random_index[1]]
                new_s = torch.zeros_like(msk_s)
                # convert bool tensor to int
                new_s = (msk_s == label).to(dtype = torch.float)
                # new_s[msk_s == label] = 1
            pt_list_s.append(random_index)
            msk_list_s.append(new_s)
        pts = torch.stack(pt_list_s, dim=0)
        msks = torch.stack(msk_list_s, dim=0)
        pt_list.append(pts)
        msk_list.append(msks)
    pt = torch.stack(pt_list, dim=-1)
    msk = torch.stack(msk_list, dim=-1)

    msk = msk.unsqueeze(1)

    return img, pt, msk #[b, 2, d], [b, c, h, w, d]



