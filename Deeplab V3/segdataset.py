"""
Author: Manpreet Singh Minhas
Contact: msminhas at uwaterloo ca
"""
import os
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image
import albumentations as albu
import cv2

import torch
import torch.utils.data as torchdata
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset

import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2

class SegmentationDataset(VisionDataset):
    """A PyTorch dataset for image segmentation task.
    The dataset is compatible with torchvision transforms.
    The transforms passed would be applied to both the Images and Masks.
    """
    def __init__(self,
                 root: str,
                 image_folder: str,
                 mask_folder: str,
                 transforms: Optional[Callable] = None,
                 seed: int = None,
                 fraction: float = None,
                 subset: str = None) -> None:
        """
        Args:
            root (str): Root directory path.
            image_folder (str): Name of the folder that contains the images in the root directory.
            mask_folder (str): Name of the folder that contains the masks in the root directory.
            transforms (Optional[Callable], optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.ToTensor`` for images. Defaults to None.
            seed (int, optional): Specify a seed for the train and test split for reproducible results. Defaults to None.
            fraction (float, optional): A float value from 0 to 1 which specifies the validation split fraction. Defaults to None.
            subset (str, optional): 'Train' or 'Test' to select the appropriate set. Defaults to None.
            image_color_mode (str, optional): 'rgb' or 'grayscale'. Defaults to 'rgb'.
            mask_color_mode (str, optional): 'rgb' or 'grayscale'. Defaults to 'grayscale'.

        Raises:
            OSError: If image folder doesn't exist in root.
            OSError: If mask folder doesn't exist in root.
            ValueError: If subset is not either 'Train' or 'Test'
            ValueError: If image_color_mode and mask_color_mode are either 'rgb' or 'grayscale'
        """
        super().__init__(root, transforms)
        image_folder_path = Path(self.root) / image_folder
        mask_folder_path = Path(self.root) / mask_folder
        self.transform = transforms

        if not image_folder_path.exists():
            raise OSError(f"{image_folder_path} does not exist.")
        if not mask_folder_path.exists():
            raise OSError(f"{mask_folder_path} does not exist.")

        self.image_folder_path = image_folder_path
        self.mask_folder_path = mask_folder_path
        #self.dict = {0:0,1:1,2:1,3:1,4:1,5:1,6:1,7:1,8:2}

        if not fraction:
            self.image_names = os.listdir(image_folder_path)
        else:
            pass # 나중에 필요하면 아래 수정 바람

            #
            # if subset not in ["Train", "Test"]:
            #     raise (ValueError(
            #         f"{subset} is not a valid input. Acceptable values are Train and Test."
            #     ))
            # self.fraction = fraction
            # self.image_list = np.array(sorted(image_folder_path.glob("*")))
            # #self.mask_list = np.array(sorted(mask_folder_path.glob("*")))
            # if seed:
            #     np.random.seed(seed)
            #     indices = np.arange(len(self.image_list))
            #     np.random.shuffle(indices)
            #     self.image_list = self.image_list[indices]
            #     #self.mask_list = self.mask_list[indices]
            # if subset == "Train":
            #     self.image_names = self.image_list[:int(
            #         np.ceil(len(self.image_list) * (1 - self.fraction)))]
            #     # self.mask_names = self.mask_list[:int(
            #     #     np.ceil(len(self.mask_list) * (1 - self.fraction)))]
            # else:
            #     self.image_names = self.image_list[
            #         int(np.ceil(len(self.image_list) * (1 - self.fraction))):]
            #     # self.mask_names = self.mask_list[
            #     #     int(np.ceil(len(self.mask_list) * (1 - self.fraction))):]

    def __len__(self) -> int:
        return len(self.image_names)

    # def get_one_hot_encoded_mask(self, mask_img):
    #     one_hot_mask = np.zeros((mask_img.shape[0], mask_img.shape[1], 3),dtype=np.uint8)
    #
    #     back = (mask_img == 0)
    #     non_crops = (mask_img == 8)
    #     crops =np.logical_not(back)*np.logical_not(non_crops)
    #
    #     one_hot_mask[:, :, 0] = np.where(back, 1, 0)
    #     one_hot_mask[:, :, 1] = np.where(crops, 1, 0)
    #     one_hot_mask[:, :, 2] = np.where(non_crops, 1, 0)
    #
    #     return one_hot_mask

    def get_one_hot_encoded_mask(self, mask_img):
        one_hot_mask = np.zeros((mask_img.shape[0], mask_img.shape[1]),dtype=np.uint8)

        back = (mask_img == 0)
        non_crops = (mask_img == 8)
        crops =np.logical_not(back)*np.logical_not(non_crops)

        one_hot_mask[back] = 0
        one_hot_mask[crops] = 1
        one_hot_mask[non_crops] = 2
        # one_hot_mask[:, :, 0] = np.where(back, 1, 0)
        # one_hot_mask[:, :, 1] = np.where(crops, 1, 0)
        # one_hot_mask[:, :, 2] = np.where(non_crops, 1, 0)

        return one_hot_mask

    def __getitem__(self, index: int) -> Any:
        name = self.image_names[index]
        image_path = os.path.join(self.image_folder_path,name)
        mask_path = os.path.join(self.mask_folder_path,name)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        origin_mask =cv2.imread(mask_path,0)
        mask = self.get_one_hot_encoded_mask(origin_mask)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
        else:
            raise "no transform"

        return transformed

if __name__ == "__main__":
    batch_size = 16
    workers = 4

    train_transform = A.Compose(
        [
            A.PadIfNeeded(min_height=256, min_width=256),
            A.RandomCrop(256, 256),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.PadIfNeeded(min_height=256, min_width=256),
            A.CenterCrop(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    dataset = SegmentationDataset(root='/home/bang/Desktop/jeju/dataset/제주 월동작물 자동탐지 드론 영상/Training',
                                  image_folder='Image1',mask_folder='Mask', transforms=train_transform)
    dataloder = torchdata.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                     num_workers=workers, pin_memory=True)

    for step, data in enumerate(dataloder):
        img, mask = data['image'].cuda(), data['mask'].cuda()
        #print('image',img.shape)
        #print('mask',mask.shape)