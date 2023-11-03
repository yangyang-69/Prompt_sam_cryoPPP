import csv
import os
import cv2
from copy import deepcopy
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import pandas as pd
from skimage.transform import rotate
# from SAM_adapter_conf.SAM_adapter_utils import random_click
import random

class TrainDataset():
    def __init__(self, image_path, label_path, is_robustness):

        self.image_path = image_path
        self.label_path = label_path
        self.is_robustness = is_robustness

        self.image_list = sorted(os.listdir(self.image_path))
        self.label_list = sorted(os.listdir(self.label_path))

        if self.is_robustness:
            self.image_list, self.label_list = self.get_images_and_labels_path_for_loop()

    def __getitem__(self, item):

        image_name = self.image_list[item]
        image = Image.open(os.path.join(self.image_path, image_name)).convert('RGB')
        image = image.resize((1024, 1024), Image.ANTIALIAS)
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
        in_points = torch.as_tensor(points_for_image, device='cuda:1')
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device='cuda:1')
        points = (in_points, in_labels)

        return image, label, points

    def __len__(self):

        return len(self.image_list)

    def get_images_and_labels_path_for_loop(self):

        self.label_list_robust = sorted([img for img in random.sample(self.label_list, 5)])
        self.image_list_robust = sorted([self.image_list[self.label_list.index(image)] for image in self.label_list_robust])
        print(f'train list:{self.label_list_robust}')

        return self.image_list_robust, self.label_list_robust



class TestDataset():
    def __init__(self, image_path, label_path, is_robustness):
        self.image_path = image_path
        self.label_path = label_path
        self.is_robustness = is_robustness

        self.image_list = sorted(os.listdir(self.image_path))
        self.label_list = sorted(os.listdir(self.label_path))

        if self.is_robustness:
            self.image_list, self.label_list = self.get_images_and_labels_path_for_loop()

    def __getitem__(self, item):
        image_name = self.image_list[item]
        image = Image.open(os.path.join(self.image_path, image_name)).convert('RGB')
        image = image.resize((1024, 1024), Image.ANTIALIAS)
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
        in_points = torch.as_tensor(points_for_image, device='cuda:1')
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device='cuda:1')
        points = (in_points, in_labels)

        return image, label, points

    def __len__(self):
        return len(self.image_list)

    def get_images_and_labels_path_for_loop(self):
        self.label_list_robust = sorted([img for img in random.sample(self.label_list, 1)])
        self.image_list_robust = sorted(
            [self.image_list[self.label_list.index(image)] for image in self.label_list_robust])
        print(f'val list:{self.label_list_robust}')

        return self.image_list_robust, self.label_list_robust



def get_imagse_and_labels_path(data_path, mode):

    label_list = sorted([os.path.join(data_path, mode, "labels", label_file) for label_file in os.listdir(os.path.join(data_path, mode, "labels"))])
    image_list = sorted([os.path.join(data_path, mode, "images", image_file) for image_file in os.listdir(os.path.join(data_path, mode, "images"))])

    print(mode, "data length:", len(label_list), len(image_list))

    return label_list, image_list

class CryopppDataset(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='train', prompt='random_click',
                 plane=False, iteration = -1, train_sample = []):

        self.train_list = train_sample
        self.valid_list = []
        self.args = args

        if iteration != -1:
            label_list, name_list = self.get_images_and_labels_path_for_loop(data_path, mode)
        else:
            label_list, name_list = get_images_and_labels_path(data_path, mode)

        self.original_size = (256, 256)
        self.target_length = 1024
        self.name_list = name_list
        self.label_list = label_list
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt  # or bboxes
        self.img_size = args.image_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):

        inout = 1
        point_label = 1
        boxes = []
        box_old = []
        pt = np.array([0, 0])
        bboxes = []

        """Get the images"""
        name = self.name_list[index]
        # img_path = os.path.join(self.data_path, self.mode, "images", name)
        img_path = name

        mask_name = self.label_list[index]
        msk_path = mask_name

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')  # ‘L’为灰度图像
        # csv_file_path = os.path.join(self.args.Groundtruth_path,img_path.split("/")[-1].split("_")[0] + ".csv")
        # with open(csv_file_path,"r") as f:
        #     csv_file = pd.read_csv(f)
        #     for i in range(csv_file.shape[0]):  # xywh format
        #         X_Coordinate = csv_file.loc[i, "X-Coordinate"]
        #         Y_Coordinate = csv_file.loc[i, "Y-Coordinate"]
        #         Diameter = csv_file.loc[i, "Diameter"]
        #         bbox = [X_Coordinate-Diameter/2, Y_Coordinate-Diameter/2, Diameter, Diameter]
        #         bboxes.append(bbox)

        newsize = (self.img_size, self.img_size)
        mask = mask.resize(newsize)

        if self.prompt == 'random_click':
            pt = random_click(np.array(mask) / 255, point_label, inout)

        elif self.prompt == 'box':
            img_name = img_path.split('/')[-1]
            with open(os.path.join(self.data_path,"bbox.csv"),mode="r") as box_file:
                reader = csv.reader(box_file)
                for index, row in enumerate(reader):
                    if index != 0 and self.mode == row[0] and img_name == row[1]:
                        boxes = np.array([int(row[2]),int(row[3]),int(row[4]),int(row[5])])

            if boxes.any():
                boxes = boxes[None, :]
                boxes = self.apply_boxes(boxes, self.original_size)
                # box_torch = torch.as_tensor(boxes, dtype=torch.float, device="cuda")
                # boxes = box_torch[None, :]
                pass

        if self.transform:
            state = torch.get_rng_state()  # 返回随机生成器状态(ByteTensor)
            img = self.transform(img)

            # 加的
            # lower_bound, upper_bound = np.percentile(img, 0.5), np.percentile(img, 99.5)
            # image_data_pre = np.clip(img, lower_bound, upper_bound)
            # image_data_pre = (image_data_pre - np.min(image_data_pre)) / (
            #         np.max(image_data_pre) - np.min(image_data_pre)) * 255.0
            # image_data_pre[img == 0] = 0
            # img = np.uint8(image_data_pre)
            # img = transforms.ToTensor()(img)
            # torch.as_tensor(torch.from_numpy(img), dtype=torch.float32)
            # img = torch.tensor(img, dtype=torch.float32)

            torch.set_rng_state(state)  # 设定随机生成器状态

            if self.prompt == 'points_grids':
                point_grids = build_all_layer_point_grids(
                    n_per_side=32,
                    n_layers=0,
                    scale_per_layer=1,
                )
                points_scale = np.array(img.shape[1:])[None, ::-1]
                points_for_image = point_grids[0] * points_scale  # (1024 * 2)
                in_points = torch.as_tensor(points_for_image)
                in_labels = torch.ones(in_points.shape[0], dtype=torch.int)
                # points = (in_points, in_labels)
                pt = points_for_image
                point_label = np.array(in_labels)

            if self.transform_msk:
                mask = self.transform_msk(mask)
                # lower_bound, upper_bound = np.percentile(mask, 0.5), np.percentile(mask, 99.5)
                # image_data_pre = np.clip(mask, lower_bound, upper_bound)
                # image_data_pre = (image_data_pre - np.min(image_data_pre)) / (
                #         np.max(image_data_pre) - np.min(image_data_pre)) * 255.0
                # image_data_pre[mask == 0] = 0
                # mask = np.uint8(image_data_pre)
                # mask = transforms.ToTensor()(mask)

        name = name.split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj': name}
        return {
            'image': img,
            'label': mask,
            'p_label': point_label,
            'pt': pt,
            'box': boxes,
            # 'box_old':box_old,
            'image_meta_dict': image_meta_dict,
            'ground_truth_bboxes': bboxes
        }

    def get_images_and_labels_path_for_loop(self, data_path, mode):

        if mode == 'train' or mode == "valid":
            label_list = sorted([os.path.join(data_path, "training_set", "labels", label_file) for label_file in
                                 os.listdir(os.path.join(data_path, "training_set", "labels"))])
            image_list = sorted([os.path.join(data_path, "training_set", "images", image_file) for image_file in
                                 os.listdir(os.path.join(data_path, "training_set", "images"))])

            if mode == 'train':
                label_train_list = sorted([img for img in random.sample(label_list, 5)])
                image_train_list = sorted([image_list[label_list.index(image)] for image in label_train_list])

                print(mode, "data length:", len(label_train_list), len(image_train_list))

                self.train_list = image_train_list

                print("train_dataset:")
                for i in range(len(self.train_list)):
                    print(self.train_list[i].split("/")[-1])

                return label_train_list, image_train_list

            elif mode == "valid":
                image_train_list = sorted([img for img in random.sample(image_list, 1) if img not in self.train_list])
                label_train_list = sorted([label_list[image_list.index(image)] for image in image_train_list])

                self.valid_list = image_train_list

                print("\nvalid_dataset:")
                for i in range(len(self.valid_list)):
                    print(self.valid_list[i].split("/")[-1])

                print(mode, "data length:", len(label_train_list), len(image_train_list))

                return label_train_list, image_train_list

        elif mode == "test":

            label_list = sorted([os.path.join(data_path, "testing_set", "labels", label_file) for label_file in
                                 os.listdir(os.path.join(data_path, "testing_set", "labels"))])
            image_list = sorted([os.path.join(data_path, "testing_set", "images", image_file) for image_file in
                                 os.listdir(os.path.join(data_path, "testing_set", "images"))])

            print(mode, "data length:", len(label_list), len(image_list))

            return label_list, image_list

    def apply_boxes(self, boxes: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array shape Bx4. Requires the original image size
        in (H, W) format.
        """
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


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

def get_images_and_labels_path(data_path, mode):

    label_list = sorted([os.path.join(data_path, mode, "labels", label_file) for label_file in os.listdir(os.path.join(data_path, mode, "labels"))])
    image_list = sorted([os.path.join(data_path, mode, "images", image_file) for image_file in os.listdir(os.path.join(data_path, mode, "images"))])

    print(mode, "data length:", len(label_list), len(image_list))

    return label_list, image_list
