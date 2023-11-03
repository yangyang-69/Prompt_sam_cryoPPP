import torch
import torchvision
import numpy as np
import torchvision.utils as vutils
import matplotlib.pyplot as plt


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



def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_img_mask(image,masks_np):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(masks_np, plt.gca())
    plt.axis('off')
    plt.show()