import os
import numpy as np
import torch.nn as nn
import torchvision
from monai.metrics import DiceMetric
from typing import Dict, Any
from SAM_adapter_conf import settings
from SAM_adapter_conf.SAM_adapter_utils import *
import torch.nn.functional as F

import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from monai.losses import DiceCELoss
from monai.transforms import (
    AsDiscrete,
)
from einops import rearrange
from notebooks.SAM_adapter_conf import SAM_adapter_cfg
from segment_anything import SamPredictor, SamAutomaticMaskGenerator

args = SAM_adapter_cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)
pos_weight = torch.ones([1]).cuda(device=GPUdevice) * 2
criterion_G = torch.nn.BCEWithLogitsLoss(
    pos_weight=pos_weight)  # BCELoss和sigmoid融合  BECloss对输出向量的每个元素单独使用交叉熵损失函数，然后计算平均值
seed = torch.randint(1, 11, (args.b, 7))

torch.backends.cudnn.benchmark = True
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
scaler = torch.cuda.amp.GradScaler()
max_iterations = settings.EPOCH
post_label = AsDiscrete(to_onehot=14)
post_pred = AsDiscrete(argmax=True, to_onehot=14)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print()
    print('total:{}'.format(total_num))
    print('trainable:{}'.format(trainable_num))
    # return {'Total': total_num, 'Trainable': trainable_num}


def get_ptsimgs(img):
    from dataset import build_all_layer_point_grids
    point_grids = build_all_layer_point_grids(
        n_per_side=32,
        n_layers=0,
        scale_per_layer=1,
    )
    points_scale = np.array(img.shape[1:])[None, ::-1]
    points_for_image = point_grids[0] * points_scale  # 1024 * 2
    in_points = torch.as_tensor(points_for_image)
    in_labels = torch.ones(in_points.shape[0], dtype=torch.int)
    # points = (in_points, in_labels)
    pt = points_for_image.tolist()
    point_label = in_labels.tolist()
    return pt, point_label


def postprocess_masks(
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
) -> torch.Tensor:
    masks = F.interpolate(
        masks,
        (1024, 1024),
        mode="bilinear",
        align_corners=False,
    )
    masks = masks[..., : input_size[0], : input_size[1]]
    masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
    return masks


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_img_mask(image, masks_np, name):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(masks_np, plt.gca())
    plt.axis('off')
    plt.savefig(name)
    plt.clf()
    # plt.show()


def mask_to_rle_pytorch(tensor: torch.Tensor) -> List[Dict[str, Any]]:
    """
    Encodes masks to an uncompressed RLE, in the format expected by
    pycoco tools.
    """
    # Put in fortran order and flatten h,w
    tensor = tensor.squeeze(0)
    b, h, w = tensor.shape
    tensor = tensor.permute(0, 2, 1).flatten(1)

    # Compute change indices
    diff = tensor[:, 1:] ^ tensor[:, :-1]
    change_indices = diff.nonzero()

    # Encode run length
    out = []
    for i in range(b):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1]
        cur_idxs = torch.cat(
            [
                torch.tensor([0], dtype=cur_idxs.dtype, device=cur_idxs.device),
                cur_idxs + 1,
                torch.tensor([h * w], dtype=cur_idxs.dtype, device=cur_idxs.device),
            ]
        )
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [] if tensor[i, 0] == 0 else [0]
        counts.extend(btw_idxs.detach().cpu().tolist())
        out.append({"size": [h, w], "counts": counts})
    return out


def rle_to_mask(rle: Dict[str, Any]) -> np.ndarray:
    """Compute a binary mask from an uncompressed RLE."""
    h, w = rle["size"]
    mask = np.empty(h * w, dtype=bool)
    idx = 0
    parity = False
    for count in rle["counts"]:
        mask[idx: idx + count] = parity
        idx += count
        parity ^= True
    mask = mask.reshape(w, h)
    return mask.transpose()  # Put in C order


def remove_small_regions(
        mask: np.ndarray, area_thresh: float, mode: str
) -> Tuple[np.ndarray, bool]:
    """
    Removes small disconnected regions and holes in a mask. Returns the
    mask and an indicator of if the mask has been modified.
    """
    import cv2  # type: ignore

    assert mode in ["holes", "islands"]
    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(
        np.uint8)  # ^异或，hole：working mask标注mask中False位置为1，islands: working mask标注mask中False为0
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask,
                                                                   8)  # 处理不规则连通区域 image : 是要处理的图片，官方文档要求是8位单通道的图像。connectivity : 可以选择是4连通还是8连通。
    # n_labels : 返回值是连通区域的数量。
    # regions : regions是一个与image一样大小的矩形（regions.shape = image.shape），其中每一个连通区域会有一个唯一标识，标识从0开始。
    # stats ：stats会包含5个参数分别为x,y,h,w,s。分别对应每一个连通区域的外接矩形的起始坐标x,y；外接矩形的wide,height；s其实不是外接矩形的面积，实践证明是labels对应的连通区域的像素个数。
    # centroids : 返回的是连通区域的质心。
    sizes = stats[:, -1][1:]  # Row 0 is background label  # 获得stats中除了0（背景）之外的各个区域的像素个数
    small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]  # 筛选出像素个数少于阈值的region index
    if len(small_regions) == 0:
        return mask, False
    fill_labels = [0] + small_regions
    if not correct_holes:
        fill_labels = [i for i in range(n_labels) if i not in fill_labels]
        # If every region is below threshold, keep largest
        if len(fill_labels) == 0:
            fill_labels = [int(np.argmax(sizes)) + 1]
    mask = np.isin(regions, fill_labels)  # 可以方便地判断数组element中的元素是否属于test_elements
    return mask, True


def postprocess_small_regions(
        mask_data, min_area: int
):
    """
    Removes small disconnected regions and holes in masks, then reruns
    box NMS to remove any new duplicates.

    Edits mask_data in place.

    Requires open-cv as a dependency.
    """
    mask_threshold = 0.0
    pred = mask_data
    mask_data = mask_data > mask_threshold
    rles = mask_to_rle_pytorch(mask_data)

    if len(rles) == 0:
        return mask_data

    # Filter small disconnected regions and holes
    for rle in rles:
        mask = rle_to_mask(rle)

        mask, changed = remove_small_regions(mask, min_area, mode="holes")
        mask, changed = remove_small_regions(mask, min_area, mode="islands")

        indexes = np.where(mask == False)
        for i in range(len(indexes[0])):
            pred[0][0][indexes[0][i]][indexes[1][i]] = -1

    return pred


def train_sam(args, net: nn.Module, optimizer, train_loader,
              epoch, writer, schedulers=None, vis=50):
    hard = 0
    epoch_loss = 0
    ind = 0
    BASE_LR = 0.01  # 小
    # train mode
    net.train()
    optimizer.zero_grad()

    epoch_loss = 0
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice

    lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')  # 也可以试试这个 针对seg任务

    # flag = False
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:

        for pack in train_loader:
            imgs = pack['image'].to(dtype=torch.float32, device=GPUdevice)
            masks = pack['label'].to(dtype=torch.float32, device=GPUdevice)
            name = pack['image_meta_dict']['filename_or_obj']
            # print(name)

            if args.prompt_approach == 'box':
                boxes = pack['box']
                box_torch = torch.as_tensor(boxes, dtype=torch.float32, device=GPUdevice)
                boxes = box_torch[None, :]

            else:
                if 'pt' not in pack:
                    imgs, pt, masks = generate_click_prompt(imgs, masks)
                    # pt, point_labels = get_ptsimgs(imgs)
                else:
                    pt = pack['pt']
                    point_labels = pack['p_label']

                showp = pt

                ind += 1
                b_size, c, w, h = imgs.size()
                longsize = w if w >= h else h

                if point_labels[0] != None:  # != -1
                    # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                    point_coords = pt
                    coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                    labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                    if args.prompt_approach == 'random_click':
                        coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]  # 追加一个新维度
                    elif args.prompt_approach == 'points_grids':
                        pass
                    pt = (coords_torch, labels_torch)

            '''init'''
            if hard:
                true_mask_ave = (true_mask_ave > 0.5).float()
                # true_mask_ave = cons_tensor(true_mask_ave)

            imgs = imgs.to(dtype=torch.float32, device=GPUdevice)

            '''Train'''
            # params = []
            for n, value in net.named_parameters():  # named_parameters() 方法可以对一个nn.Module中所有注册的参数进行迭代
                if "prompt_embeddings" in n:
                    value.requires_grad = True
                elif args.token_output_type == "linear" and 'image_encoder.linear' in n:
                    value.requires_grad = True
                elif 'mask_decoder' in n:
                    value.requires_grad = True
                else:
                    value.requires_grad = False

            # for p in net.mask_decoder.parameters():
            #     p.requires_grad = True

            # get_parameter_number(net)

            imge = net.image_encoder(imgs)  # image embeddings

            if args.prompt_approach == 'box':
                se, de = net.prompt_encoder(
                    points=None,  # 用grids作为prompt
                    boxes=boxes,
                    masks=None,
                )
            else:
                se, de = net.prompt_encoder(
                    points=pt,  # 用grids作为prompt
                    boxes=None,
                    masks=None,
                )

            pred, _ = net.mask_decoder(  # batched predicted masks
                image_embeddings=imge,
                image_pe=net.prompt_encoder.get_dense_pe(),
                # get_dense_pe() get positional encoding used to encode point prompts
                sparse_prompt_embeddings=se,
                dense_prompt_embeddings=de,
                multimask_output=False,
            )

            loss = lossfunc(pred, masks)  # pred -> mask  masks -> label

            pbar.set_postfix(**{'loss (batch)': loss.item()})  # 显示指标
            epoch_loss += loss.item()
            loss.backward()

            # nn.utils.clip_grad_value_(net.parameters(), 0.1)
            optimizer.step()  # 更新权重
            optimizer.zero_grad()  # 清除梯度

            '''vis images'''
            if vis:
                if ind % vis == 0:
                    namecat = 'Train'
                    for na in name:
                        namecat = namecat + na.split('/')[-1].split('.')[0] + '+'

                    if args.prompt_approach == 'random_click':
                        vis_image(imgs, pred, masks, os.path.join(args.path_helper['train_sample_path'],
                                                                  namecat + 'epoch+' + str(epoch) + '.jpg'),
                                  reverse=False, points=showp)
                    elif args.prompt_approach == 'points_grids':
                        vis_image(imgs, pred, masks, os.path.join(args.path_helper['train_sample_path'],
                                                                  namecat + 'epoch+' + str(epoch) + '.jpg'),
                                  reverse=False, points=None)
                    elif args.prompt_approach == 'box':
                        vis_image(imgs, pred, masks, os.path.join(args.path_helper['train_sample_path'],
                                                                  namecat + 'epoch+' + str(epoch) + '.jpg'),
                                  reverse=False, box=pack['box'])

            pbar.update()

    return loss


def validation_sam(args, val_loader, epoch, net: nn.Module, clean_dir=True):
    # eval mode
    net.eval()

    mask_type = torch.float32
    n_val = len(val_loader)  # the number of batch
    ave_res, mix_res = (0, 0, 0, 0), (0, 0, 0, 0)
    rater_res = [(0, 0, 0, 0) for _ in range(6)]
    tot = 0
    hard = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice

    lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            imgsw = pack['image'].to(dtype=torch.float32, device=GPUdevice)
            masksw = pack['label'].to(dtype=torch.float32, device=GPUdevice)
            # for k,v in pack['image_meta_dict'].items():
            #     print(k)
            name = pack['image_meta_dict']['filename_or_obj']

            buoy = 0
            if args.evl_chunk:
                evl_ch = int(args.evl_chunk)
            else:
                evl_ch = int(imgsw.size(-1))

            if 'pt' not in pack:  # 改   1：当前情况的结果   2：用默认的情况看效果
                imgsw, ptw, masksw = generate_click_prompt(imgsw, masksw)
            else:
                ptw = pack['pt']
                point_labels = pack['p_label']

            while (buoy + evl_ch) <= imgsw.size(-1):
                imgs = imgsw[..., buoy:buoy + evl_ch]
                masks = masksw[..., buoy:buoy + evl_ch]
                buoy += evl_ch
                ind += 1

                if args.prompt_approach == 'box':
                    boxes = pack['box']
                    box_torch = torch.as_tensor(boxes, dtype=torch.float32, device=GPUdevice)
                    boxes = box_torch[None, :]
                    pass

                else:
                    pt = ptw
                    showp = pt
                    if point_labels[0] != None:
                        # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                        point_coords = pt
                        coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                        labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                        if args.prompt_approach == 'random_click':
                            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]  # 追加一个新维度
                        elif args.prompt_approach == 'points_grids':
                            pass
                        pt = (coords_torch, labels_torch)

                '''init'''
                if hard:
                    true_mask_ave = (true_mask_ave > 0.5).float()
                    # true_mask_ave = cons_tensor(true_mask_ave)
                imgs = imgs.to(dtype=torch.float32, device=GPUdevice)

                '''test'''

                with torch.no_grad():
                    imge = net.image_encoder(imgs)

                    if args.prompt_approach == 'box':
                        se, de = net.prompt_encoder(
                            points=None,  # 用grids作为prompt
                            boxes=boxes,
                            masks=None,
                        )
                    else:
                        se, de = net.prompt_encoder(
                            points=pt,  # 用grids作为prompt
                            boxes=None,
                            masks=None,
                        )

                    pred, _ = net.mask_decoder(
                        image_embeddings=imge,
                        image_pe=net.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=se,
                        dense_prompt_embeddings=de,
                        multimask_output=False,
                    )

                    tot += lossfunc(pred, masks)

                    '''vis images'''
                    if ind % args.vis == 0:
                        namecat = 'Valid'
                        for na in name:
                            img_name = na.split('/')[-1].split('.')[0]
                            namecat = namecat + img_name + '+'

                        if args.prompt_approach == 'random_click':
                            vis_image(imgs, pred, masks, os.path.join(args.path_helper['valid_sample_path'],
                                                                      namecat + 'epoch+' + str(epoch) + '.jpg'),
                                      reverse=False, points=showp)
                        elif args.prompt_approach == 'points_grids':
                            vis_image(imgs, pred, masks, os.path.join(args.path_helper['valid_sample_path'],
                                                                      namecat + 'epoch+' + str(epoch) + '.jpg'),
                                      reverse=False, points=None)
                        elif args.prompt_approach == 'box':
                            vis_image(imgs, pred, masks, os.path.join(args.path_helper['valid_sample_path'],
                                                                      namecat + 'epoch+' + str(epoch) + '.jpg'),
                                      reverse=False, box=pack['box'])

                    mask_old = masks
                    # mask_threshold = 0.5
                    # return_logits = False
                    #
                    # masks = postprocess_masks(pred, (1024, 1024), (1024, 1024))
                    # if not return_logits:
                    #     masks = masks > mask_threshold
                    #
                    # masks_np = masks[0].detach().cpu().numpy()
                    # # true_point = random_click(masks_np[0], inout=True)
                    #
                    # image = torch.squeeze(imgs, dim=0).permute(1, 2, 0)
                    # image = image.detach().cpu().numpy()
                    # show_img_mask(image, masks_np)

                    temp = eval_seg(pred, mask_old, threshold)
                    # temp = get_boxes(pred, mask_old, threshold)
                    # print("\nold_dice:  ", temp)
                    # temp = eval_seg_N_P(pred, mask_old, threshold)
                    # print("new_dice:  ", temp)
                    mix_res = tuple([sum(a) for a in zip(mix_res, temp)])

            pbar.update()

    if args.evl_chunk:
        n_val = n_val * (imgsw.size(-1) // evl_ch)

    return tot / n_val, tuple([a / n_val for a in mix_res])


def Test_sam(args, val_loader, epoch, net: nn.Module, clean_dir=True):
    # eval mode
    net.eval()

    mask_type = torch.float32
    n_val = len(val_loader)  # the number of batch
    ave_res, mix_res = [(0, 0, 0, 0), (0, 0, 0, 0)], [(0, 0, 0, 0), (0, 0, 0, 0)]
    rater_res = [(0, 0, 0, 0) for _ in range(6)]
    tot = 0
    hard = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice
    # metrics = []
    metrics = [[], []]

    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        # lossfunc = criterion_G
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            imgsw = pack['image'].to(dtype=torch.float32, device=GPUdevice)
            masksw = pack['label'].to(dtype=torch.float32, device=GPUdevice)
            # for k,v in pack['image_meta_dict'].items():
            #     print(k)
            if 'pt' not in pack:  # 改   1：当前情况的结果   2：用默认的情况看效果
                imgsw, ptw, masksw = generate_click_prompt(imgsw, masksw)
            else:
                ptw = pack['pt']
                point_labels = pack['p_label']
            name = pack['image_meta_dict']['filename_or_obj']

            buoy = 0
            if args.evl_chunk:
                evl_ch = int(args.evl_chunk)
            else:
                evl_ch = int(imgsw.size(-1))

            while (buoy + evl_ch) <= imgsw.size(-1):
                imgs = imgsw[..., buoy:buoy + evl_ch]
                masks = masksw[..., buoy:buoy + evl_ch]
                buoy += evl_ch
                ind += 1

                if args.prompt_approach == 'box':
                    boxes = pack['box']
                    box_torch = torch.as_tensor(boxes, dtype=torch.float32, device=GPUdevice)
                    boxes = box_torch[None, :]
                    pass

                else:
                    pt = ptw
                    showp = pt
                    if point_labels[0] != None:
                        # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                        point_coords = pt
                        coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                        labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                        if args.prompt_approach == 'random_click':
                            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]  # 追加一个新维度
                        elif args.prompt_approach == 'points_grids':
                            pass
                        pt = (coords_torch, labels_torch)

                '''init'''
                if hard:
                    true_mask_ave = (true_mask_ave > 0.5).float()
                    # true_mask_ave = cons_tensor(true_mask_ave)
                imgs = imgs.to(dtype=torch.float32, device=GPUdevice)

                '''test'''
                with torch.no_grad():
                    imge = net.image_encoder(imgs)

                    if args.prompt_approach == 'box':
                        se, de = net.prompt_encoder(
                            points=None,  # 用box作为prompt
                            boxes=boxes,
                            masks=None,
                        )
                    else:
                        se, de = net.prompt_encoder(
                            points=pt,  # 用grids作为prompt
                            boxes=None,
                            masks=None,
                        )

                    pred, _ = net.mask_decoder(
                        image_embeddings=imge,
                        image_pe=net.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=se,
                        dense_prompt_embeddings=de,
                        multimask_output=False,
                    )

                    tot += lossfunc(pred, masks)

                    '''vis images'''
                    if ind % args.vis == 0:
                        namecat = 'Valid'
                        for na in name:
                            img_name = na.split('/')[-1].split('.')[0]
                            namecat = namecat + img_name + '+'

                        if args.prompt_approach == 'random_click':
                            vis_image(imgs, pred, masks, os.path.join(args.path_helper['test_sample_path'],
                                                                      namecat + 'epoch+' + str(epoch) + '.jpg'),
                                      reverse=False, points=showp)
                        elif args.prompt_approach == 'points_grids':
                            vis_image(imgs, pred, masks, os.path.join(args.path_helper['test_sample_path'],
                                                                      namecat + 'epoch+' + str(epoch) + '.jpg'),
                                      reverse=False, points=None)
                        elif args.prompt_approach == 'box':
                            vis_image(imgs, pred, masks, os.path.join(args.path_helper['test_sample_path'],
                                                                      namecat + 'epoch+' + str(epoch) + '.jpg'),
                                      reverse=False, box=pack['box'])

                mask_old = masks
                mask_threshold = 0.5
                return_logits = False

                masks = postprocess_masks(pred, (1024, 1024), (1024, 1024))
                if not return_logits:
                    masks = masks > mask_threshold

                masks_np = masks[0].detach().cpu().numpy()
                true_point = random_click(masks_np[0], inout=True)

                image = torch.squeeze(imgs, dim=0).permute(1, 2, 0)
                image = image.detach().cpu().numpy()
                show_img_mask(image, masks_np, os.path.join(args.path_helper['test_sample_path'],
                                                            namecat + '_mask_gt' + '.jpg'))

                temp = eval_seg(pred, mask_old, threshold)
                print(temp)
                # temp = eval_seg_N_P(pred, mask_old, threshold)
                # print("p and n:  ", temp)
                metrics[0].append(temp)
                mix_res[0] = tuple([sum(a) for a in zip(mix_res[0], temp)])

                if args.min_mask_region_area > 0:
                    pred_minimal = postprocess_small_regions(pred, args.min_mask_region_area)

                    tot += lossfunc(pred_minimal, masks)

                    '''vis images'''
                    if ind % args.vis == 0:
                        namecat = 'Test'
                        for na in name:
                            img_name = na.split('/')[-1].split('.')[0]
                            namecat = namecat + img_name + '+'

                        if args.prompt_approach == 'random_click':
                            vis_image(imgs, pred_minimal, mask_old, os.path.join(args.path_helper['test_sample_path'],
                                                                                 namecat + "minimal_area" + 'epoch+' + str(
                                                                                     epoch) + '.jpg'),
                                      reverse=False, points=showp)
                        elif args.prompt_approach == 'points_grids':
                            print(namecat + 'epoch+' + str(epoch) + '.jpg')
                            vis_image(imgs, pred_minimal, mask_old, os.path.join(args.path_helper['test_sample_path'],
                                                                                 namecat + "minimal_area" + 'epoch+' + str(
                                                                                     epoch) + '.jpg'),
                                      reverse=False, points=None)

                    mask_threshold = 0.5
                    return_logits = False

                    masks = postprocess_masks(pred_minimal, (1024, 1024), (1024, 1024))
                    if not return_logits:
                        masks = masks > mask_threshold

                    masks_np = masks[0].detach().cpu().numpy()
                    # true_point = random_click(masks_np[0], inout=True)

                    image = torch.squeeze(imgs, dim=0).permute(1, 2, 0)
                    image = image.detach().cpu().numpy()
                    show_img_mask(image, masks_np, os.path.join(args.path_helper['test_sample_path'],
                                                                namecat + "minimal_area" + '_mask_gt' + '.jpg'))

                    temp = eval_seg(pred_minimal, mask_old, threshold)
                    # print(temp)
                    # temp = eval_seg_N_P(pred, mask_old, threshold)
                    # print("p and n:  ",temp)
                    metrics[1].append(temp)
                    mix_res[1] = tuple([sum(a) for a in zip(mix_res[1], temp)])

            pbar.update()

    if args.evl_chunk:
        n_val = n_val * (imgsw.size(-1) // evl_ch)

    print(metrics)
    # print(metrics[0])
    # print(tuple([a / n_val for a in mix_res[0]]))
    '-'# print(tuple([[a / n_val for a in mix_res[1]]))

    return tot / n_val, tuple([a / n_val for a in mix_res[0]])


def get_annotations(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask

    return img

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()  # 利用plt.gca( )进行坐标轴的移动  get current axes
    ax.set_autoscale_on(False)  # 设置是否在下一次绘制或调用 `Axes.autoscale_view'时将自动缩放应用于每个轴。

    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))  # np.ones()函数返回给定形状和数据类型的新数组，其中元素的值设置为1   534*800*3
        color_mask = np.random.random((1, 3)).tolist()[0]  # 代表三个浮点数
        for i in range(3):
            img[:,:,i] = color_mask[i]  # 第一层，第二层，第三层，侧切面
        # ax.imshow(np.dstack((img, m*0.35)))

        box = ann["bbox"]

        x = (ann["bbox"][2] + ann["bbox"][0]) / 2
        y = (ann["bbox"][3] + ann["bbox"][1]) / 2
        r = (ann["bbox"][2] - ann["bbox"][0]) / 2 if (ann["bbox"][2] - ann["bbox"][0]) / 2 < (ann["bbox"][3] - ann["bbox"][1]) / 2 else (ann["bbox"][3] - ann["bbox"][1]) / 2
        # draw_circle = plt.Circle((x, y), r, fill=False)
        # ax.add_artist(draw_circle)
        # ax.add_patch(plt.Circle((x, y), r, fill=False, edgecolor='green', facecolor=(0, 0, 0, 0), lw=5))

        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='red', facecolor=(0, 0, 0, 0), lw=1))
        # plt.scatter(x, y, lw=0, marker='o', edgecolors='r', s=r)

def prepare_plot(image, mask, sam_mask, coords, image_path):
    output_path = r"/mnt/Data1/yzy/code/Sam/LOG"

    plt.figure(figsize=(40, 30))
    plt.subplot(231)
    plt.title('Testing Image', fontsize=14)
    plt.imshow(image, cmap='gray')
    plt.subplot(232)
    plt.title('Original Mask', fontsize=14)
    plt.imshow(mask, cmap='gray')
    plt.subplot(234)
    # plt.title('Attention-UNET Mask', fontsize=14)
    # plt.imshow(predicted_mask, cmap='gray')
    # plt.subplot(235)
    plt.title('SAM Mask', fontsize=14)
    plt.imshow(sam_mask, cmap='gray')
    plt.subplot(236)
    plt.title('Final Picked Particles', fontsize=14)
    plt.imshow(coords, cmap='gray')
    # path = image_path.split("/")[-1]
    # path = path.replace(".jpg", "_result.jpg")
    # plt.savefig(os.path.join(f"{output_path}/results/", path))
    # final_path = os.path.join(f"{output_path}/results/", f'predicted_{path}')
    # cv2.imwrite(final_path, coords)


def Test_sam_circle(args, val_loader, epoch, net: nn.Module, clean_dir=True):
    # eval mode
    import statistics as st

    net.eval()

    n_val = len(val_loader)  # the number of batch
    ave_res, mix_res = [(0, 0, 0, 0), (0, 0, 0, 0)], [(0, 0, 0, 0), (0, 0, 0, 0)]
    rater_res = [(0, 0, 0, 0) for _ in range(6)]
    tot = 0
    hard = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice
    metrics = [[], []]

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            imgsw = pack['image'].to(dtype=torch.float32, device=GPUdevice)
            masksw = pack['label'].to(dtype=torch.float32, device=GPUdevice)

            if 'pt' not in pack:  # 改   1：当前情况的结果   2：用默认的情况看效果
                imgsw, ptw, masksw = generate_click_prompt(imgsw, masksw)
            else:
                ptw = pack['pt']
                point_labels = pack['p_label']
            name = pack['image_meta_dict']['filename_or_obj']
            print(name)

            buoy = 0
            if args.evl_chunk:
                evl_ch = int(args.evl_chunk)
            else:
                evl_ch = int(imgsw.size(-1))

            while (buoy + evl_ch) <= imgsw.size(-1):
                imgs = imgsw[..., buoy:buoy + evl_ch]
                masks_ori = masksw[..., buoy:buoy + evl_ch]
                buoy += evl_ch
                ind += 1

                if args.prompt_approach == 'box':
                    boxes = pack['box']
                    box_torch = torch.as_tensor(boxes, dtype=torch.float32, device=GPUdevice)
                    boxes = box_torch[None, :]
                    pass

                else:
                    pt = ptw
                    showp = pt
                    if point_labels[0] != None:
                        # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                        point_coords = pt
                        coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                        labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                        if args.prompt_approach == 'random_click':
                            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]  # 追加一个新维度
                        elif args.prompt_approach == 'points_grids':
                            pass
                        pt = (coords_torch, labels_torch)

                '''init'''
                if hard:
                    true_mask_ave = (true_mask_ave > 0.5).float()
                    # true_mask_ave = cons_tensor(true_mask_ave)
                imgs = (imgs.to(dtype=torch.float32, device=GPUdevice)[0,:,:,:].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)

                # image = cv2.imread('pathway1.jpg')
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                mask_generator = SamAutomaticMaskGenerator(model=net,
                                                           pred_iou_thresh=0.3,
                                                           stability_score_thresh=0.3,
                                                           points_per_batch=512,
                                                           # crop_nms_thresh=0.1,
                                                           box_nms_thresh=0.99
                                                           )  # 初始化掩码生成器
                masks = mask_generator.generate(imgs)  # 生成该图像的mask

                plt.figure(figsize=(20, 20))
                plt.imshow(imgs)

                sam_mask = get_annotations(masks)
                show_anns(masks)

                plt.axis('off')
                plt.show()

                width, height = 1024, 1024
                input_image_width, input_image_height = 1024, 1024
                sam_mask = cv2.resize(sam_mask, (1024, 1024))

                bboxes = []
                for i in range(0, len(masks)):
                    if masks[i]["predicted_iou"] > 0.5:
                        box = masks[i]["bbox"]
                        bboxes.append(box)

                segment_mask = copy.deepcopy(imgs)

                if len(bboxes) > 1:

                    x_ = st.mode([box[2] for box in bboxes])
                    y_ = st.mode([box[3] for box in bboxes])
                    d_ = np.sqrt(
                        (x_ * width / input_image_width) ** 2 + (y_ * height / input_image_height) ** 2)
                    r_ = int(d_ // 2)
                    th = r_ * 0.20
                    # segment_mask = cv2.cvtColor(segment_mask, cv2.COLOR_GRAY2BGR)
                    for b in bboxes:
                        if b[2] < x_ + th and b[2] > x_ - th / 3 and b[3] < y_ + th and b[3] > y_ - th / 3:
                            x_new, y_new = int((b[0] + b[2] / 2) / input_image_width * width), int(
                                (b[1] + b[3] / 2) / input_image_height * height)
                            coords = cv2.circle(segment_mask, (x_new, y_new), r_, (0, 0, 255), 8)
                    image_path = ""
                    try:
                        prepare_plot(imgs, masks_ori, sam_mask, coords, image_path)
                    except:
                        pass
                else:
                    pass

                # predictor = SamPredictor(net)  # 初始化
                #
                # predictor.set_image(imgs[0,:,:,:])  # Calculates the image embeddings  return None

                # input_boxes = torch.tensor([[0,0,0,0]])
                # for i in range(point_coords.shape[0]):
                #     if i == 0:
                #         input_boxes = torch.tensor([[point_coords[i][0],point_coords[i][0],1,1]])
                #     else:
                #         input_boxes = torch.cat([input_boxes,[point_coords[i][0],point_coords[i][0],1,1]],dim=1)
                #
                # input_boxes = torch.tensor([
                #     [75, 275, 1725, 850],
                #     [425, 600, 700, 875],
                #     [1375, 550, 1650, 800],
                #     [1240, 675, 1400, 750],
                # ], device=predictor.device)
                #
                # input_pc = point_coords[0,:,:]
                # input_pl = point_labels[0]
                #
                # print()
                #
                # point_coords = point_coords[0,:,:].cpu().numpy()
                # point_labels = point_labels.cpu().numpy()
                #
                #
                #
                # masks, _, _ = predictor.predict_torch(
                #     point_coords=input_pc,
                #     point_labels=input_pl,
                #     multimask_output=False,
                # )
                #
                # print(masks.shape)  # (number_of_masks) x H x W

            pbar.update()

    if args.evl_chunk:
        n_val = n_val * (imgsw.size(-1) // evl_ch)

    print(metrics)
    # print(metrics[0])
    # print(tuple([a / n_val for a in mix_res[0]]))
    # print(tuple([[a / n_val for a in mix_res[1]]))

    return tot / n_val, tuple([a / n_val for a in mix_res[0]])