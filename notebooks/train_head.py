import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from unet_model import UNet_model
# from ori_sam_model import Sam_model
from segment_anything import sam_model_registry
from monai.losses import DiceCELoss
from dataset import TrainDataset,TestDataset
from torch.autograd import Function
import time
import logging
import copy

class Sam_model(nn.Module):
    def __init__(self,model_type,sam_checkpoint):
        super(Sam_model, self).__init__()

        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

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
    def __init__(self,model_type,sam_checkpoint):
        super(Model,self).__init__()

        self.unet = UNet_model(in_channels=3,out_channels=3)
        self.sam = Sam_model(model_type=model_type,sam_checkpoint=sam_checkpoint)

        for n, value in self.sam.named_parameters():
            if 'mask_decoder' in n:
                value.requires_grad = True
            else:
                value.requires_grad = False

    def forward(self, x, points):

        denoised_img = self.unet(x)
        img_add = x + denoised_img
        img_add = torch.clamp(img_add,0,255)
        masks = self.sam(img_add,points)
        return denoised_img, masks

def cal_iou(outputs: np.array, labels: np.array):
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
        eiou += cal_iou(disc_pred, disc_mask)

        '''dice for torch'''
        edice += dice_coeff(vpred[:, 0, :, :], gt_vmask_p[:, 0, :, :]).item()

    return eiou / len(threshold), edice / len(threshold)


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total:{}'.format(total_num))
    print('trainable:{}'.format(trainable_num))
    # return {'Total': total_num, 'Trainable': trainable_num}


def evaluate(model,val_dataloader):
    model.eval()

    val_loss = []
    iou_list = []
    dice_list = []
    with torch.no_grad():

        for image,label,points in val_dataloader:

            image = image.to(device=device)
            label = label.to(device=device)

            denoised_img, pred = model(image, points)

            loss = lossfunc(pred,label)
            val_loss.append(loss.item())
            iou,dice = eval_seg(pred, label, threshold)
            iou_list.append(iou)
            dice_list.append(dice)

        loss_mean = np.average(val_loss)
        iou_mean = np.average(iou_list)
        dice_mean = np.average(dice_list)

    return  loss_mean,iou_mean,dice_mean




def train(model,train_dataloader):

    model.train()
    train_loss = []
    iou_list = []
    dice_list = []
    for image, label, points in train_dataloader:
        image = image.to(device=device)
        label = label.to(device=device)
        optimizer.zero_grad()

        denoised_img, pred = model(image, points)

        loss = lossfunc(pred, label) * 100
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        iou, dice = eval_seg(pred, label, threshold)
        iou_list.append(iou)
        dice_list.append(dice)

    loss_mean = np.average(train_loss)
    iou_mean = np.average(iou_list)
    dice_mean = np.average(dice_list)

    logger.info(
        f"| epoch {epoch:3d} | "f"train loss {loss_mean:5.2f} | "f"iou {iou_mean:3.2f}  | "f"dice {dice_mean:3.2f}"
    )



if __name__ == '__main__':

    path = '/mnt/Data1/yzy/code/Sam/dataset/10947_split/'
    dataset_name = '10947'

    save_path = '/mnt/Data1/yzy/code/Sam/head-prompt/' + dataset_name +'/'

    subset_size = 10
    exp_name = f'fewshot_{subset_size}'

    train_image = path + subset_size + '/train/images/'
    train_label = path + subset_size + '/train/labels/'
    train_data = TrainDataset(train_image, train_label)
    train_dataloader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)

    val_image = path + subset_size + '/valid/images/'
    val_label = path + subset_size + '/valid/labels/'
    val_data = TestDataset(val_image, val_label)
    val_dataloader = DataLoader(dataset=val_data, batch_size=1, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 100
    learning_rate = 0.0001
    best_loss = 100
    best_iou = 1
    best_dice = 1
    logger = logging.getLogger('head-prompt SAM')

    sam_checkpoint = "/mnt/Data1/yzy/code/Sam/model_checkpoint/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    model = Model(model_type=model_type, sam_checkpoint=sam_checkpoint)
    model = model.to(device)
    lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        epoch_start_time = time.time()
        train(model,train_dataloader)
        val_loss,iou,dice = evaluate(model,val_dataloader)

        elapsed = time.time() - epoch_start_time
        logger.info("-" * 89)
        logger.info(
            f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
            f"valid loss {val_loss:5.4f} | "f"iou {iou:3.2f}  | "f"dice {dice:3.2f}" )
        logger.info("-" * 89)

        if (val_loss < best_loss) or (iou > best_iou) or (dice > best_dice) :
            best_loss = val_loss
            best_iou = iou
            best_dice = dice
            torch.save(model.state_dict(), save_path + 'head_prompt_' + dataset_name + '_' + exp_name + "_add.pt")
            logger.info(f"Best model saved!")

    logger.info('Head prompt ' + dataset_name + exp_name + ' training is complete ! ')





