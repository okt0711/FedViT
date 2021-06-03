import os
import tarfile
import collections
from torchvision.datasets.vision import VisionDataset
import xml.etree.ElementTree as ET
from PIL import Image
from typing import Any, Callable, Dict, Optional, Tuple, List
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
import warnings

import copy
import numpy as np
import torch
import random
from .voc_dataloader import _VOCBase

class CustomVOC(_VOCBase):
    def __init__(self, root, image_set, transform, target_transform):
        super(CustomVOC, self).__init__(root=root,
                                        image_set=image_set,
                                        transform=transform,
                                        target_transform=target_transform)

    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years ``"2007"`` to ``"2012"``.
        image_set (string, optional): Select the image_set to use, ``"train"``, ``"trainval"`` or ``"val"``. If
            ``year=="2007"``, can also be ``"test"``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """    


    _SPLITS_DIR = "Segmentation"
    _TARGET_DIR = "SegmentationClass"
    _TARGET_FILE_EXT = ".png"

    # @property
    # def masks(self) -> List[str]:
        # return self.targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert("RGB")
        target = copy.deepcopy(img)
        img_path = self.images[index]

        seed = np.random.randint(2147483647) # make a seed with numpy generator 

        if self.transform is not None:
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            img = self.transform(img)
        if self.target_transform is not None:
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            target = self.target_transform(target)

        return img, target, img_path