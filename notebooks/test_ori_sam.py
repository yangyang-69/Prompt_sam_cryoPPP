#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Sam_temp 
@File    ：test_ori_sam.py
@Author  ：yang
@Date    ：2023/11/5 14:28 
'''
import os

from numpy import *

from dataset import *
from torch.utils.data import DataLoader
from SAM_conf.SAM_utils import *
import function_token

args = SAM_cfg.parse_args()
GPUdevice = torch.device('cuda', args.gpu_device)
net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=args.distributed)

args.path_helper = set_log_dir('../logs', args.exp_name)
logger = create_logger(args.path_helper['log_path'])
logger.info(args)

'''segmentation data'''
transform_test = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
])

transform_test_seg = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((args.out_size, args.out_size)),

])

if args.dataset == 'CryoPPP':

    turkey_test_dataset = CryopppDataset(args, args.data_path, transform=transform_test,
                                         transform_msk=transform_test_seg, mode='test', prompt=args.prompt_approach)

    nice_test_loader = DataLoader(turkey_test_dataset, batch_size=args.b, shuffle=False, num_workers=8,
                                  pin_memory=True)

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))

''' test original sam and show mask '''
for image in os.listdir(os.path.join(args.data_path, "test", 'images')):
    img = os.path.join(args.data_path, "test", 'images',image)
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(20,20))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    sam_checkpoint = args.sam_ckpt
    model_type = "default"
    import sys
    sys.path.append("..")
    from segment_anything import SamAutomaticMaskGenerator
    mask_generator = SamAutomaticMaskGenerator(net)
    masks = mask_generator.generate(img)

    print(len(masks))
    print(masks[0].keys())

    plt.figure(figsize=(20,20))
    plt.imshow(img)
    show_anns(masks)
    masks_json = masks
    plt.show()

'''begain valuation'''
best_acc = 0.0
best_tol = 1e4

net.eval()
tol, (eiou, edice) = function_token.Test_sam(args, nice_test_loader, 0, net)
logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {0}.')