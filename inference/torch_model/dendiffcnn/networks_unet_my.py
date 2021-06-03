import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np


###############################################################################
# Functions
###############################################################################
def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('LinearAttention') != -1:
        # import ipdb; ipdb.set_trace()
        for ms in m._modules:
            init.normal_(m._modules[ms].weight, 0.0, 0.02)
        return
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('LinearAttention') != -1:
        # import ipdb; ipdb.set_trace()
        for ms in m._modules:
            init.normal_(m._modules[ms].weight, 0.0, 0.02)
        return
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
    for ms in net._modules:
        if ms == 'embd':
            net._modules[ms].weight.data.uniform_()
            continue

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def define_G(input_nc, which_model_netG, init_type='normal', gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert (torch.cuda.is_available())
    if which_model_netG == 'DenDiff_head':
        from .denoising_diffusion_mysc import Unet_encoder
        netG = Unet_encoder(input_nc, dim = 128, dim_mults = (1, 2, 4))
    elif which_model_netG == 'DenDiff_tail':
        from .denoising_diffusion_mysc import Unet_decoder
        netG = Unet_decoder(input_nc, dim = 128, dim_mults = (1, 2, 4), skip='add')
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    init_weights(netG, init_type=init_type)
    return netG

def define_T(num_feat, dfeat, enc_layer, which_model_netT, init_type='normal', gpu_ids=[]):
    netT = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert (torch.cuda.is_available())
    if which_model_netT == 'cnnTransformer':
        netG = cnnTransformer(num_feat, dfeat, enc_layer)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netT)
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    init_weights(netG, init_type=init_type)
    return netG

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

from einops.layers.torch import Rearrange
class cnnTransformer(nn.Module):
    def __init__(self, num_feat, dfeat, enc_layers=6):
        super(cnnTransformer, self).__init__()


        self.enc_sequences = Rearrange('b c h w -> (h w) b c')
        self.pos_embedding = nn.Parameter(torch.randn(num_feat, 1, dfeat))
        encoder_layer = nn.TransformerEncoderLayer(d_model=dfeat, nhead=8, dim_feedforward=1024, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=enc_layers)
        self.dec_sequences = Rearrange('(h w) b c -> b c h w', h=16)

    def forward(self, input):
        x = self.enc_sequences(input)
        x = x + self.pos_embedding
        x = self.transformer(x)
        x = self.dec_sequences(x)
        return x
