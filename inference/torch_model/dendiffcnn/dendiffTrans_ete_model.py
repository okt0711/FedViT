import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks_unet_my as networks_unet

class TransformerModel(BaseModel):
    def name(self):
        return 'TransformerModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.inputSize
        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)

        # load/define networks
        self.netG_H = networks_unet.define_G(opt.input_nc, 'DenDiff_head', opt.init_type, self.gpu_ids)
        self.netG_T = networks_unet.define_G(opt.input_nc, 'DenDiff_tail', opt.init_type, self.gpu_ids)
        self.netT = networks_unet.define_T(16*16, 512, 8, opt.which_model_netT, opt.init_type, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_H, 'G_H', which_epoch)
            self.load_network(self.netG_T, 'G_T', which_epoch)
            self.load_network(self.netT, 'T', which_epoch)


        if self.isTrain:
            self.old_lr = opt.lr

            self.criterionAE = torch.nn.MSELoss()

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_H.parameters(), self.netT.parameters(), self.netG_T.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)

            for optimizer in self.optimizers:
                self.schedulers.append(networks_unet.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks_unet.print_network(self.netG_H)
        networks_unet.print_network(self.netT)
        print('-----------------------------------------------')

    def set_input(self, input):
        input_A = input['A']
        input_B = input['B']

        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['path']

    def set_input_test(self, input):
        input_A = input['A']
        self.input_A.resize_(input_A.size()).copy_(input_A)

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)

    def test(self):
        with torch.no_grad():
            real_A = Variable(self.input_A)
            X_enc, t = self.netG_H(real_A)
            x = self.netT(X_enc)
            X_dec = self.netG_T(x, t)

        self.X_dec = X_dec.data

        # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_G(self):

        X_enc, t, h = self.netG_H(self.real_A)
        x = self.netT(X_enc)
        X_dec = self.netG_T(x, t, h)

        loss_ae = self.criterionAE(X_dec, self.real_B) # * 5
        loss = loss_ae
        loss.backward()

        self.X_dec = X_dec.data
        self.loss_AE = loss_ae.item()
        self.loss = loss.item()

    def optimize_parameters(self, step):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        ret_errors = OrderedDict([('AE', self.loss_AE),('Tot', self.loss)])
        return ret_errors

    def get_current_visuals(self):
        real_A = util.tensor2im(self.input_A[0])
        real_B = util.tensor2im(self.input_B[0])
        fake_B = util.tensor2im(self.X_dec[0])

        ret_visuals = OrderedDict([('real_A', real_A), ('real_B', real_B), ('fake_B', fake_B)])
        return ret_visuals

    def get_current_data(self):
        ret_visuals = OrderedDict([('real_A', self.input_A), ('fake_B', self.X_dec)])
        return ret_visuals

    def save(self, label):
        self.save_network(self.netT, 'T', label, self.gpu_ids)
        self.save_network(self.netG_H, 'G_H', label, self.gpu_ids)
        self.save_network(self.netG_T, 'G_T', label, self.gpu_ids)
