import cv2
import torch
import torch.nn as nn
import numpy as np
import random
from .dct import dct_2d, idct_2d
from .matrices import zigzag_idx, Q10, Q50, Q90


class JPEGEncoder(nn.Module):
    def __init__(self, device, patch_size, macro_patch_size, quantize, Q_matrix, israndomQ):
        super().__init__()
        # //TODO check mode and keep_prob (if mode != bert, keep_prob should be 1. log as warning.)
        self.isquantize = quantize
        self.israndomQ = israndomQ
        self.p = patch_size
        self.mp = macro_patch_size

        if self.isquantize:
            Qs = {'Q10': Q10, 'Q50': Q50, 'Q90': Q90}
            self.Q = Qs[Q_matrix.default].to(device)
        else:
            self.Q = torch.ones((8, 8)).to(device)
    
    def __call__(self, batch):
        return self.encode(batch)

    def concat_channelwise(self, codes):
        codes = codes.permute(2, 0, 1, 3)  # (pn, bs, c, p*p)
        codes = codes.reshape(codes.size(0), codes.size(1), -1)  # (pn, bs, c*p*p)
        return codes

    def encode(self, img, keep_prob=1.0):
        """
        This function encodes the input image size lower than or equal to 128.
        Compressing image tensors using JPEG encoding. 
        The compressed output will be the input to the transformer.

        <Shape info>
        input:
            img: (Batch_size, Channels, Height, Width)
        return:
            flatten: (Patch_num, Batch_size, Channels*Patch_size**2)
            no_quan_flatten: (Patch_num, Batch_size, Channels*Patch_size**2)
            mask : (Patch_num, Batch_size, Channels*Patch_size**2)
        """
        assert 0 < keep_prob <= 1, print('keep_prob should be in (0,1].')

        patches = self.divide_to_blocks(img, patch_size=self.p)
        # Change image scale from [0,255] to [-128,127] to reduce the
        # dynamic range requirements for the DCT computing.
        patches -= 128
        masked, mask = self.random_masking(patches, keep_prob)
        transformed = self.discrete_cosine_transform(img=masked)

        if self.isquantize:
            transformed = self.quantize(transformed, self.israndomQ)

        flatten = self.zigzag_flatten(transformed)
        flatten = self.concat_channelwise(flatten)

        no_quan_transformed = self.discrete_cosine_transform(img=patches)
        no_quan_flatten = self.zigzag_flatten(no_quan_transformed)
        no_quan_flatten = self.concat_channelwise(no_quan_flatten)

        if mask is not None:
            mask = self.zigzag_flatten(mask)
            mask = self.concat_channelwise(mask)

        return flatten, no_quan_flatten, mask

    @torch.no_grad()
    def zigzag_flatten(self, img):
        """
        Flatten image patches in a zigzag manner.
        You can change the order by changning the predifined index matrix.

        e.g)
        [[15, 14, 10, 9]
         [13, 11, 8 , 0]
         [12, 0 , 0 , 0]
         [0 , 0 , 0 , 0]]   -> [15,14,13,12,11,10,9,8,0,...,0]

        <shape info>
        input:
            img: (Batch_size, Channels, Patch_num, p, p)
        return: 
            flatten: (Batch_size, Channels, Patch_num, p*p)
        """

        # case for 8x8 patch
        idx = zigzag_idx
        flatten = []

        p = img.size(-1)
        for i in range(p*p):
            posi = torch.where(idx == i)
            idx_row = posi[0].item()
            idx_col = posi[1].item()
            flatten.append(img[:, :, :, idx_row, idx_col])

        flatten = torch.stack(flatten, dim=-1)
        return flatten

    @torch.no_grad()
    def divide_to_blocks(self, img, patch_size):
        """
        Divide image tensor into patches.
        Depending on argument macro, patch size will be selected between mp and p.

        <shape info>
        input:
            img: (Batch_size, Channels, Height, Width) 
        return: 
            patches: (Batch_size, Channels, Patch_num, Patch_size, Patch_size) 
        """
        bs = img.size(0)
        pn = int(img.size(2)/patch_size)**2

        H, W = img.size(-2), img.size(-1)
        patches = []

        curY = 0
        for i in range(patch_size, H+1, patch_size):
            curX = 0
            for j in range(patch_size, W+1, patch_size):
                patches.append(img[:, :, curY:i, curX:j])
                curX = j
            curY = i

        # create patch_num dimension: shape (bs, ch, pn, ps, ps)
        patches = torch.stack(patches, dim=2)

        return patches

    @torch.no_grad()
    def random_masking(self, patches, keep_prob):
        """
        This function zero out some portion of patches with probability 'keep_prob'.
        Implemented as element-wise tensor multiplication.
        Masking is independently applied to across batch-dimension.

        <shape info>
        input:
            patches: (Batch_size, channels, Patch_num, Patch_size, Patch_size)
        return:
            patches: (Batch_size, channels, Patch_num, Patch_size, Patch_size)
            mask: (Batch_size, channels, Patch_num, Patch_size, Patch_size)

        In the case of macro encoding, the exact size of each dimension is slightly different.
        But, you don't have to consider it seriously. This makes sense as a module.
        """
        bs, ch, pn, ps, ps = patches.size()

        if keep_prob < 1.0:
            # shape: (bs, pn, 3, ps, ps)
            patches = patches.permute(0, 2, 1, 3, 4)
            # shape: (bs, pn, 3*ps*ps)
            patches = patches.reshape(patches.size(0), patches.size(1), -1)
            remove_num = int(pn*(1-keep_prob))
            remove_indices = []

            for _ in range(bs):
                remove_indices.append(np.random.choice(pn, remove_num, replace=False))

            remove_indices = torch.tensor(remove_indices).to(patches.device)  # shape: (bs, remove_num)
            mask = torch.ones((bs, pn)).to(patches.device)
            mask = mask.scatter(1, remove_indices, 0).unsqueeze(-1)
            # shape: (bs, pn, 3*ps*ps)
            mask = mask.repeat([1, 1, patches.size(-1)])
            patches = torch.mul(patches, mask)

            # shape: (bs, pn, 3, ps, ps)
            patches = patches.view(bs, pn, -1, ps, ps)
            # shape: (bs, 3, pn, ps, ps)
            patches = patches.permute(0, 2, 1, 3, 4)

            # for mask return,
            # mask = mask.view(bs, ch, pn, ps, ps)
            mask = mask.view(bs, pn, -1, ps, ps)
            mask = mask.permute(0, 2, 1, 3, 4)
        else:
            # do nothing.
            mask = None

        return patches, mask

    def discrete_cosine_transform(self, img):
        """
        Apply the discrete cosine transform.
        If self.quantize = True, divide the transformed matrix by the quantization matrix Q.

        <shape info>
        input:
            img: (Batch_size, Channels, Patch_num, Patch_size, Patch_size) 
        return: 
            output: (Batch_size, Channels, Patch_num, Patch_size, Patch_size)
        """

        output = dct_2d(img, norm='ortho')
        return output

    def quantize(self, dct_img, israndomQ=False):
        """
        Quantize the discrete cosine transformed image using a specified matrix.
        Note that the size of quantization matrix (Q matrix) is the same as p (patch size).
        Here, we need to implement element-wise division.
        Don't warry about the zero dividing because the defined Q matirx is the positive definite.

        If israndomQ==Ture, the model will quantize using Q10, Q50 and Q90 randomly regardless to self.Q.
        You can set this at config file.

        <shape info>
        input:
            dct_img: (Batch_size, Channels, Patch_num, Patch_size, Patch_size) 
        return: 
            quantized: (Batch_size, Channels, Patch_num, Patch_size, Patch_size) 
        """
        if not israndomQ:
            quantized = torch.round(torch.div(dct_img, self.Q))
        else:
            q_list = [Q10, Q50, Q90]
            idx = random.randint(1, 3)-1
            quantized = torch.round(
                torch.div(dct_img, q_list[idx].to(dct_img.device)))

        return quantized


class JPEGDecoder(nn.Module):
    def __init__(self, img_height, img_width):
        super().__init__()
        self.H = img_height
        self.W = img_width

    def __call__(self, compressed, p=8):
        return self.decode(compressed, p)

    def split_channelwise(self, codes, p, mp=None):
        # mp is for macro version
        if mp is None:
            codes = codes.view(codes.size(0), codes.size(1), -1, p**2)  # (pn, bs, ch, p*p)
            codes = codes.permute(1, 2, 0, 3)  # (bs, ch, pn, p*p)
        else:
            last_dim = int(mp/p)**2 * p**2
            codes = codes.view(codes.size(0), codes.size(1), -1, last_dim)
            codes = codes.permute(1, 2, 0, 3)  # (bs, 3, mpn, (mp/p)^2*p*p)

        return codes

    def decode(self, compressed, p=8):
        """
        This function decodes the compressed code made by 'encode' funcion of the encoder.
        If you compressed the input image using 'macro_encode', you should use 'macro_decode'.
        Reconstructing image tensors using JPEG decoding. 
        The compressed input will be the output of the transformer.

        <shape info>
        input:
            compressed: (Patch_num, Batch_size, Channels*p*p) size of torch.Tensor
        return : 
            img: (Batch_size, Channels, Height, Width) size of torch.Tensor
        """
        
        compressed = self.split_channelwise(compressed, p, mp=None)
        
        block = self.zigzag_block(compressed)
        # block = torch.mul(block, Q90.to(compressed.device))

        inversed = self.inverse_discrete_cosine_transform(block)
        inversed += 128
        img = self.back_to_image(inversed, width=self.W)
        img = torch.clamp(img, min=0, max=255)

        return img

    def zigzag_block(self, code):
        """
        Reconstruct image patches in a zigzag manner.

        e.g)
        [[15, 14, 10, 9]
         [13, 11, 8 , 0]
         [12, 0 , 0 , 0]
         [0 , 0 , 0 , 0]]   <- [15,14,13,12,11,10,9,8,0,...,0]

        <shape info>
        input:
            code: (Batch_size, Channels, Patch_num, Patch_size**2)
        return: 
            patches: (Batch_size, Channels, Patch_num, Patch_size, Patch_size)
        """

        # case for 8x8 patch
        idx = zigzag_idx.type(torch.LongTensor)
        rows = []
        for row in idx:
            one_row = []
            for one_idx in row:
                one_row.append(code[:, :, :, one_idx])
            rows.append(torch.stack(one_row, dim=-1))
        patches = torch.stack(rows, dim=-2)
        return patches

    def back_to_image(self, patches, width):
        """
        Aggregate patch tensors into images.
        Assume that the output image has the same size of width and height.

        <shape info>
        input:
            patches: (Batch_size, Channels, Patch_num, Patch_size, Patch_size)
        return: 
            img: (Batch_size, Channels, Height, Width)
        """
        n, p = patches.size(-3), patches.size(-2)

        currow = 0
        rows = []
        for i in range(int(width/p), n+1, int(width/p)):
            partial_patch = patches[:, :, currow:i, :, :]
            partial = []
            for j in range(partial_patch.size(-3)):
                partial.append(partial_patch[:, :, j, :, :])

            rows.append(torch.cat(partial, dim=-1))
            currow = i

        img = torch.cat(rows, dim=-2)
        return img

    def inverse_discrete_cosine_transform(self, compressed):
        """
        Apply the inverse discrete cosine transform.

        <shape info>
        input:
            compressed: (Batch_size, Channels, Patch_num, Patch_size, Patch_size) 
        return: 
            output: (Batch_size, Channels, Patch_num, Patch_size, Patch_size)
        """
        output = idct_2d(compressed, norm='ortho')
        return output
