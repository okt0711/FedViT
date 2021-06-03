import os
import glob
import copy
import random
import numpy as np
import torch
from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, Dict, Optional, Tuple, List
import cv2
from PIL import Image

def check_validity(directory):
    # now, this will raise Error unless .jpg or .png files are in directory.
    if 'VOC' in directory:
        directory = os.path.join(directory, "JPEGImages")

    jpg_list = glob.glob(os.path.join(directory, '*.jpg'))
    png_list = glob.glob(os.path.join(directory, '*.png'))
    if len(jpg_list) == 0 and len(png_list) == 0:
        raise FileNotFoundError

class BaseDataset(VisionDataset):
    def __init__(self, 
                 root,
                 task,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 **kwargs
                 ):
        
        self.transform = transform
        self.target_transform = target_transform
        self.task = task
        self.root = root
        self.files = []

        if 'VOC' in self.root:
            # This is only for the validation.
            img_list = 'inference/data/VOCList/val.txt'
            with open(os.path.join(img_list), "r") as f:
                file_names = [x.strip() for x in f.readlines()]
                image_dir = os.path.join(self.root, "JPEGImages")
                self.files = [os.path.join(image_dir, x + ".jpg") for x in file_names]

        # Load proper file lists
        if self.task == 'deblocking':
            try:
                self.Q = kwargs['Q']
            except KeyError:
                print('DEBLOCK DATASET: You should give Q. Set the default value (50).')
                self.Q = 50

            try:
                dataFiles = sorted(os.listdir(root))
                check_validity(root)
            except FileNotFoundError:
                print('Warning: No such file or directory. Instead, load sample images.')
                root = 'inference/data/samples/deblocking/'
                dataFiles = sorted(os.listdir(root))

            for idata, dataName in enumerate(dataFiles):
                self.files.append(os.path.join(root, dataName))
            
        elif self.task == 'denoising':
            try:
                self.mean = kwargs['mean']
                self.std = kwargs['std']
            except KeyError:
                print('DENOISE DATASET: You should give both mean and std. Set the default values (0, 30.0).')
                self.mean = 0
                self.std = 30.0

            try:
                dataFiles = sorted(os.listdir(root))
                check_validity(root)
            except FileNotFoundError:
                print('Warning: No such file or directory. Instead, load sample images.')
                root = 'inference/data/samples/denoising/'
                dataFiles = sorted(os.listdir(root))

            for idata, dataName in enumerate(dataFiles):
                self.files.append(os.path.join(root, dataName))

        elif self.task == 'deblurring':
            # initialize files
            self.files = []
            # define new lists
            self.in_files = []
            self.target_files = []
            input_dir = os.path.join(self.root, 'blur')
            target_dir = os.path.join(self.root, 'sharp')

            try:
                in_dataFiles = sorted(os.listdir(input_dir))
                target_dataFiles = sorted(os.listdir(target_dir))
                check_validity(input_dir)
                check_validity(target_dir)

            except FileNotFoundError:
                print('Warning: No such file or directory. Instead, load sample images.')
                input_dir = 'inference/data/samples/deblurring/blur'
                target_dir = 'inference/data/samples/deblurring/sharp'
                in_dataFiles = sorted(os.listdir(input_dir))
                target_dataFiles = sorted(os.listdir(target_dir))
            
            for idata, dataName in enumerate(in_dataFiles):
                self.in_files.append(os.path.join(input_dir, dataName))
            for idata, dataName in enumerate(target_dataFiles):
                self.target_files.append(os.path.join(target_dir, dataName))

        elif self.task == 'deraining':
            # initialize files
            self.files = []
            # define new lists
            self.in_files = []
            self.target_files = []
            input_dir = os.path.join(self.root, 'input')
            target_dir = os.path.join(self.root, 'target')

            try:
                in_dataFiles = sorted(os.listdir(input_dir))
                target_dataFiles = sorted(os.listdir(target_dir))
                check_validity(input_dir)
                check_validity(target_dir)
            except FileNotFoundError:
                print('Warning: No such file or directory. Instead, load sample images.')
                input_dir = 'inference/data/samples/deraining/input'
                target_dir = 'inference/data/samples/deraining/target'
                in_dataFiles = sorted(os.listdir(input_dir))
                target_dataFiles = sorted(os.listdir(target_dir))
            
            for idata, dataName in enumerate(in_dataFiles):
                self.in_files.append(os.path.join(input_dir, dataName))
            for idata, dataName in enumerate(target_dataFiles):
                self.target_files.append(os.path.join(target_dir, dataName))

        else:
            dataFiles = sorted(os.listdir(root))
            for idata, dataName in enumerate(dataFiles):
                self.files.append(os.path.join(root, dataName))

        
    def __len__(self):
        if self.task in ['deblocking', 'denoising']:
            return len(self.files)
        elif self.task in ['deraining', 'deblurring']:
            assert len(self.in_files) == len(self.target_files), print('Input and Target dataset should be the same number.')
            return len(self.in_files)
        else:
            return 0

    def __getitem__(self, index):

        if self.task == 'deblocking':
            img_path = self.files[index]
            in_ = Image.open(img_path).convert("RGB")
            target_ = copy.deepcopy(in_)
            # quantization is done by transform.
            
        elif self.task == 'denoising':
            img_path = self.files[index]
            target_ = cv2.imread(img_path, cv2.IMREAD_COLOR)
            target_ = target_[:, :, ::-1]  # change BGR to RBG
            noisy_img = target_ + np.random.normal(self.mean, self.std, target_.shape)

            in_ = np.clip(noisy_img, 0, 255)

        elif self.task in ['deblurring','deraining']:
            img_path = self.in_files[index]
            target_path = self.target_files[index]
            in_ = Image.open(img_path).convert("RGB")
            target_ = Image.open(target_path).convert("RGB")

        seed = np.random.randint(2147483647) # make a seed with numpy generator 

        if self.transform is not None:
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            in_ = self.transform(in_)
        if self.target_transform is not None:
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            target_ = self.target_transform(target_)

        return in_, target_, img_path
    
