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

class DeblurDataset(BaseDataset):
    def __init__(self, root, cropsize, valid=True):
        super().__init__(root=root, task='deblurring')

        augment = [
        transforms.RandomCrop(size=cropsize),
        RandomRotate90(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        ]

        # noise is added when the data is numpy.ndarray in BaseDataset
        if not valid:
            self.target_transform = transforms.Compose(
                augment + [transforms.ToTensor(), ReScale(maxi=1)]
            )
            self.transform = transforms.Compose(
                augment + [transforms.ToTensor(), ReScale(maxi=1)]
            )
        else:
            # for eval
            self.target_transform = transforms.Compose(
                    [transforms.ToTensor(), ReScale(maxi=1),]
            )
            self.transform = transforms.Compose(
                [transforms.ToTensor(), ReScale(maxi=1)]
            )

def deblur_dataloader(dataroot,
                      cropsize,
                      batch_size, 
                      num_workers,
                      valid=False
                      ):

    dataset = DeblurDataset(dataroot, cropsize, valid=valid)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            drop_last=(not valid),
                            shuffle=(not valid),
                            num_workers=num_workers
                            )

    data = {'loader':dataloader,
            'num':dataset.__len__()}

    return data
