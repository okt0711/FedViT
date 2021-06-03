import os
import numpy as np
import torch
from torchvision.datasets.vision import VisionDataset
import cv2

from torch.utils.data import DataLoader
# from torch.utils.data.sampler import SubsetRandomSampler 
import torchvision.transforms as transforms
from ..torch_model.utils import ReScale, ToYUVTensor, ToRGBTensor, RandomRotate90, img2patch, patch2img
from ..torch_model.jpegmodules.quantize import Quantizer
from .base_dataset import BaseDataset


class ToFloatTensor():
    def _toFloatTensor(self, ndarray):
        return torch.from_numpy(ndarray.copy().astype(np.float32)).permute(2,0,1)
    def __call__(self, ndarray):
        return self._toFloatTensor(ndarray)

class DenoiseDataset(BaseDataset):
    def __init__(self, root, cropsize, mean, std, valid=True):
        super().__init__(root=root, task='denoising', mean=mean, std=std)

        augment = [
        transforms.RandomCrop(size=cropsize),
        RandomRotate90(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        ]

        # noise is added when the data is numpy.ndarray in BaseDataset
        if not valid:
            self.target_transform = transforms.Compose(
                augment + [ToFloatTensor(), ReScale(maxi=255)]
            )
            self.transform = transforms.Compose(
                augment + [ToFloatTensor(), ReScale(maxi=255)]
            )
        else:
            # for eval
            self.target_transform = transforms.Compose(
                    [ToFloatTensor(), ReScale(maxi=255)]
            )
            self.transform = transforms.Compose(
                [ToFloatTensor(), ReScale(maxi=255)]
            )

def denoise_dataloader( dataroot,
                        cropsize,
                        batch_size, 
                        num_workers,
                        mean=None,
                        std=None,
                        valid=False
                        ):

    dataset = DenoiseDataset(dataroot, cropsize, mean=mean, std=std, valid=valid)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            drop_last=(not valid),
                            shuffle=(not valid),
                            num_workers=num_workers
                            )

    data = {'loader':dataloader,
            'num':dataset.__len__()}

    return data
