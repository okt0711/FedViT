import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

class cnnTransformer(nn.Module):
    def __init__(self,
                 name:str,
                 n_token:int,
                 n_embed:int,
                 n_head:int,
                 n_hid:int,
                 n_layer:int,
                 dropout:float=0.5):

        super(cnnTransformer, self).__init__()

        self.enc_sequences = Rearrange('b c h w -> (h w) b c')
        self.pos_embedding = nn.Parameter(torch.randn(n_token, 1, n_embed))
        encoder_layer = nn.TransformerEncoderLayer(d_model=n_embed, 
                                                   nhead=n_head,
                                                   dim_feedforward=n_hid,
                                                   dropout=dropout)

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.dec_sequences = Rearrange('(h w) b c -> b c h w', h=16)

    def forward(self, input):
        x = self.enc_sequences(input)
        x = x + self.pos_embedding
        x = self.transformer(x)
        x = self.dec_sequences(x)
        return x

