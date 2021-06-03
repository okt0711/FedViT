import cv2
import torch
import torch.nn as nn
import numpy as np
import random
from .dct import dct_2d, idct_2d
from .matrices import zigzag_idx, Q10, Q50, Q90

class Quantizer:
    def __init__(self, target_Q=None):
        if target_Q == [10, 50]: self.Qs = [Q10, Q50] # for random quality training
        elif target_Q == 10: self.Qs = [Q10]
        elif target_Q == 50: self.Qs = [Q50]
        else:
            raise Exception('Quantizer only works for Q=10 and Q=50 (got {}). You can customize it.'.format(target_Q))

    def __call__(self, img):
        return self.quantize(img)
    
    def __repr__(self):
        return self.__class__.__name__ + '()'

    def random_lv_choose(self):
        idx = random.randint(1, len(self.Qs))-1
        return self.Qs[idx]

    @torch.no_grad()
    def quantize(self, img):
        # split into patches
        patches = self.divide_to_blocks(img, patch_size=8)

        # Change image scale from [0,255] to [-128,127] to reduce the
        # dynamic range requirements for the DCT computing.
        patches -= 128

        # apply DCT
        transformed = dct_2d(patches, norm='ortho')

        # quantize using randomly choosed Quantization matrix
        Q = self.random_lv_choose().to(img.device)

        quantized = torch.round(torch.div(transformed, Q))
        quantized = torch.mul(quantized, Q)

        # apply iDCT
        output = idct_2d(quantized, norm='ortho')
        output += 128

        # aggregate
        return self.back_to_image(output, width=img.size(-1))

    @torch.no_grad()
    def divide_to_blocks(self, img, patch_size):
        # C, H, W
        pn = int(img.size(2)/patch_size)**2
        H, W = img.size(1), img.size(2)
        patches = []

        curY = 0
        for i in range(patch_size, H+1, patch_size):
            curX = 0
            for j in range(patch_size, W+1, patch_size):
                patches.append(img[ :, curY:i, curX:j])
                curX = j
            curY = i

        # create patch_num dimension: shape (ch, pn, ps, ps)
        patches = torch.stack(patches, dim=1)

        return patches
    
    @torch.no_grad()
    def back_to_image(self, patches, width):
        # C, Pn, Ps, Ps = 3, 256, 8, 8

        n, p = patches.size(1), patches.size(2)

        currow = 0
        rows = []
        for i in range(int(width/p), n+1, int(width/p)):
            partial_patch = patches[ :, currow:i, :, :]
            partial = []
            for j in range(partial_patch.size(-3)):
                partial.append(partial_patch[ :, j, :, :])

            rows.append(torch.cat(partial, dim=-1))
            currow = i

        img = torch.cat(rows, dim=-2)
        return img