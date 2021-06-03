import argparse

import cv2
import os
import ntpath
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from omegaconf import OmegaConf
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from torchvision.utils import save_image
from tqdm import tqdm as tqdm
from inference.inference_model import InferenceModel
from inference.data.deblock_dataloader import deblock_dataloader
from inference.torch_model.utils import *

def to_np(tensor):
    """Convert torch.tensor to numpy.ndarray
    The shape will be changed as [B, C, H, W] -> [B, H, W, C]
    """
    if tensor.device == 'cpu':
        return tensor[0,...].permute(1,2,0).numpy()
    else:
        return tensor[0,...].permute(1,2,0).cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', type=str, default=0)
    parser.add_argument('--Q', type=str, default=10)
    args = parser.parse_args()

    DEVICE = torch.device("cuda:{}".format(args.gpu) if (torch.cuda.is_available() and args.gpu != 'None') else "cpu")

    save_dir = './inference/results/deblocking/'
    os.makedirs(save_dir, exist_ok=True)

    config_path = './inference/deblocking_inference_config.yaml'
    config = OmegaConf.load(config_path)

    model = InferenceModel(**config).to(DEVICE)
    model.eval()

    valid_data = deblock_dataloader(**config.data_config, Q=int(args.Q), valid=True)

    avg_in_psnr = 0
    avg_out_psnr = 0
    avg_in_ssim = 0
    avg_out_ssim = 0

    with torch.no_grad():
        for in_, target_, img_path in tqdm(valid_data['loader']):
            
            in_ = in_.to(DEVICE)
            target_ = target_.to(DEVICE)            
            target_ = target_[:, :, :in_.shape[2], :in_.shape[3]]

            # split into patches
            patches, orig_shape, padded_shape, unfolded_shape = img2patch(in_, k=64, s=32)
            # model forward
            out_patches = model(patches)
            # aggregate to image
            out_ = patch2img(out_patches, 64, 32, orig_shape, padded_shape, unfolded_shape)
            # scale to [0.0, 255.0]
            in_ = torch.clamp((in_/2) + 0.5, 0, 1) * 255
            out_ = torch.clamp((out_/2) + 0.5, 0, 1) * 255
            target_ = torch.clamp((target_/2) + 0.5, 0, 1) * 255

            # save the output image
            img_name = ntpath.basename(img_path[0])
            
            in_save_path = os.path.join(save_dir, 'input_' + img_name)
            save_image(in_/255, in_save_path, nrow=1)
            out_save_path = os.path.join(save_dir, 'result_' + img_name)
            save_image(out_/255, out_save_path, nrow=1)

            # convert to numpy
            in_np = to_np(in_)
            out_np = to_np(out_)
            target_np = to_np(target_)

            # Calculate PSNR
            in_psnr = psnr(target_np, in_np, data_range=255.0)
            out_psnr = psnr(target_np, out_np, data_range=255.0)
            avg_in_psnr += in_psnr/valid_data['num']
            avg_out_psnr += out_psnr/valid_data['num']

            # Calculate SSIM
            in_ssim = ssim(target_np, in_np, data_range=255.0, multichannel=True, gaussian_weights=True, use_sample_covariance=False)
            out_ssim = ssim(target_np, out_np, data_range=255.0, multichannel=True, gaussian_weights=True, use_sample_covariance=False)
            avg_in_ssim += in_ssim/valid_data['num']
            avg_out_ssim += out_ssim/valid_data['num']

            del in_, target_, patches, out_patches, out_, in_np, out_np, target_np

        print('Average PSNR | Input: {} dB | Output: {} dB'.format(avg_in_psnr, avg_out_psnr))
        print('Average SSIM | Input: {} | Output: {}'.format(avg_in_ssim, avg_out_ssim))


if __name__ == '__main__':
    main()
