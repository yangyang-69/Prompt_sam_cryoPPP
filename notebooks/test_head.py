import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import numpy as np
import pandas as pd
import cv2, os, time
import torchvision
from tqdm import tqdm
import itertools, sys
import torch
torch.set_printoptions(profile="full")
from torch import optim
from PIL import Image
from torchvision import models,transforms
from typing import Tuple, List
from unet_model import UNet_model
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Function
from monai.losses import DiceCELoss
import torchvision.utils as vutils
import torch.nn as nn
from segment_anything import sam_model_registry
from segment_anything import build_sam_vit_h,build_sam_vit_b
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def build_all_layer_point_grids(
    n_per_side: int, n_layers: int, scale_per_layer: int
) -> List[np.ndarray]:
    """Generates point grids for all crop layers."""
    points_by_layer = []
    for i in range(n_layers + 1):
        n_points = int(n_per_side / (scale_per_layer**i))
        points_by_layer.append(build_point_grid(n_points))
    return points_by_layer

def build_point_grid(n_per_side: int) -> np.ndarray:
    """Generates a 2D grid of points evenly spaced in [0,1]x[0,1]."""
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
    return points


def iou(outputs: np.array, labels: np.array):
    SMOOTH = 1e-6
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    return iou.mean()

def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).to(device = input.device).zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

def eval_seg(pred, true_mask_p, threshold):

    eiou, edice = 0, 0
    for th in threshold:
        gt_vmask_p = (true_mask_p > th).float()
        vpred = (pred > th).float()
        vpred_cpu = vpred.cpu()
        disc_pred = vpred_cpu[:, 0, :, :].numpy().astype('int32')

        disc_mask = gt_vmask_p[:, 0, :, :].squeeze(1).cpu().numpy().astype('int32')

        '''iou for numpy'''
        eiou += iou(disc_pred, disc_mask)

        '''dice for torch'''
        edice += dice_coeff(vpred[:, 0, :, :], gt_vmask_p[:, 0, :, :]).item()

    return eiou / len(threshold), edice / len(threshold)


def vis_image(imgs, pred_masks, gt_masks, save_path, points=None):
    b, c, h, w = pred_masks.size()
    dev = pred_masks.get_device()
    row_num = min(b, 4)

    if torch.max(pred_masks) > 1 or torch.min(pred_masks) < 0:
        pred_masks = torch.sigmoid(pred_masks)

    imgs = torchvision.transforms.Resize((h, w))(imgs)
    pred_masks = pred_masks[:, 0, :, :].unsqueeze(1).expand(b, 3, h, w)
    gt_masks = gt_masks[:, 0, :, :].unsqueeze(1).expand(b, 3, h, w)
    if points != None:
        for i in range(b):
            p = np.round(points.cpu() / 1024 * 256).to(dtype=torch.int)
            gt_masks[i, 0, p[i, 0] - 5:p[i, 0] + 5, p[i, 1] - 5:p[i, 1] + 5] = 0.5
            gt_masks[i, 1, p[i, 0] - 5:p[i, 0] + 5, p[i, 1] - 5:p[i, 1] + 5] = 0.1
            gt_masks[i, 2, p[i, 0] - 5:p[i, 0] + 5, p[i, 1] - 5:p[i, 1] + 5] = 0.4
    tup = (imgs[:row_num, :, :, :], pred_masks[:row_num, :, :, :], gt_masks[:row_num, :, :, :])
    compose = torch.cat(tup, 0)
    vutils.save_image(compose, fp=save_path, nrow=row_num, padding=10)

    return


class TestDataset():
    def __init__(self,image_path,label_path):

        self.image_path = image_path
        self.label_path = label_path

        self.image_list = sorted(os.listdir(image_path))
        self.label_list = sorted(os.listdir(label_path))


    def __getitem__(self, item):

        image_name = self.image_list[item]
        image = Image.open(os.path.join(self.image_path, image_name)).convert('RGB')
        image = image.resize((1024,1024),Image.ANTIALIAS)
        image = transforms.ToTensor()(image)

        label_name = self.label_list[item]
        label = Image.open(os.path.join(self.label_path, label_name)).convert('L')
        label = label.resize((256, 256), Image.ANTIALIAS)

        label = transforms.ToTensor()(label).long()

        points_scale = np.array(image.shape[1:])[None, ::-1]
        point_grids = build_all_layer_point_grids(
            n_per_side=32,
            n_layers=0,
            scale_per_layer=1,
        )
        points_for_image = point_grids[0] * points_scale
        in_points = torch.as_tensor(points_for_image,device='cuda')
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int,device='cuda')
        points = (in_points, in_labels)

        return image, label, points,image_name

    def __len__(self):

        return len(self.image_list)

class Sam_model(nn.Module):
    def __init__(self,model_type):
        super(Sam_model, self).__init__()

        # self.sam = sam_model_registry[model_type]
        if model_type == 'vit_h':
            self.sam = build_sam_vit_h()
        elif model_type == 'vit_b':
            self.sam = build_sam_vit_b()

    def forward(self, x, points):

        image = self.sam.image_encoder(x)
        se, de = self.sam.prompt_encoder(
            points=points,
            boxes=None,
            masks=None,
        )
        pred, _ = self.sam.mask_decoder(
            image_embeddings=image,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=se,
            dense_prompt_embeddings=de,
            multimask_output=False,
        )

        return pred


class Model(nn.Module):
    def __init__(self,model_type):
        super(Model,self).__init__()

        self.unet = UNet_model(in_channels=3,out_channels=3)
        self.sam = Sam_model(model_type=model_type)

        for p in self.sam.parameters():
            p.requires_grad = False

    def forward(self, x, points):

        denoised_img = self.unet(x)
        img_add = x + denoised_img
        img_add = torch.clamp(img_add, 0, 255)
        masks = self.sam(img_add,points)
        return denoised_img, masks


def compute_dice_coefficient(mask_gt, mask_pred):

    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2 * volume_intersect / volume_sum


if __name__ == '__main__':

    image_path = '/mnt/Data1/yzy/code/Sam/head-prompt/robustness/10028_V/test/images/'
    label_path = '/mnt/Data1/yzy/code/Sam/head-prompt/robustness/10028_V/test/labels/'
    train_data = TestDataset(image_path, label_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = "vit_h"

    model = Model(model_type=model_type)
    model = model.to(device)
    test_data = TestDataset(image_path, label_path)
    test_dataloader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)


    state_dict = torch.load(f'/mnt/Data1/yzy/code/Sam/head-prompt/robustness/head_prompt_10028_robustness_{i}_add.pt')
    model.load_state_dict(state_dict,strict=False)

    loss_list = []
    result_list = []
    dice = []
    model.eval()
    with torch.no_grad():
        for data,label,points,image_name in test_dataloader:

            data = data.to(device=device)
            label = label.to(device=device)

            denoised_img, pred = model(data, points)

            loss = lossfunc(pred, label)
            loss_list.append(loss.item())

            result = eval_seg(pred, label, threshold)
            result_list.append(result)
            dice.append(result[1])

        print('loss:{}'.format(loss_list))
        print('result:{}'.format(result_list))

        print("loss_mean", np.mean(loss_list), "dice_mean", np.mean(dice))



