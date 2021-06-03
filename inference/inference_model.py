import os
import glob
import torch
import torch.nn as nn
from inference.torch_model.dendiffcnn.denoising_diffusion_mysc import Unet_encoder
from inference.torch_model.dendiffcnn.denoising_diffusion_mysc import Unet_decoder
from inference.torch_model.transformer.transformer import cnnTransformer

class InferenceModel(nn.Module):
    def __init__(self, model_config, inference_config, data_config):
        super().__init__()

        self.model_config = model_config
        self.infer_config = inference_config
        self.date_config = data_config

        self.task = self.infer_config.task

        self.head = Unet_encoder(input_nc=3, dim=128, dim_mults=(1, 2, 4))
        self.tail = Unet_decoder(input_nc=3, dim=128, dim_mults = (1, 2, 4), skip='add')
        self.body = cnnTransformer(**model_config.transformer_config)

        ckpt_root = inference_config.ckpt_root
        bodycycle = inference_config.body_load_cycle
        HTprefix = os.path.join(ckpt_root, self.task)
        Bperfix = os.path.join(ckpt_root, 'task_agnostic_body')
        try:
            # to load both .pt and .pth files
            head_query = os.path.join(HTprefix, 'FedViT_'+self.task+'_head') + '*'
            tail_query = os.path.join(HTprefix, 'FedViT_'+self.task+'_tail') + '*'
            head_path = glob.glob(head_query)[0]
            tail_path = glob.glob(tail_query)[0]
            self.head.load_state_dict(torch.load(head_path))
            self.tail.load_state_dict(torch.load(tail_path))
            self.body.load_state_dict(torch.load(os.path.join(Bperfix, 'FedViT_body_cycle'+str(bodycycle)+'.pth')))
            
        except:
            # do not load.
            print('Checkpoints do not exist. Randomly initializing.')

    def forward(self, batch):
        head_out, t, s = self.head(batch)
        body_out = self.body(head_out)
        output = self.tail(body_out, t, s)
        return output
