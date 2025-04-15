import torch
import torch.nn as nn
from .utils_dl import get_activation

class Res_MLP(nn.Module):
    def __init__(self, d_in, d_block, d_hidden, n_block_l, n_block_h, d_out, act:str, act_param=None, skip=True):
        super(Res_MLP, self).__init__()

        self.activation = get_activation(act, act_param)
        self.skip = skip

        # projection layer
        self.input_layer = nn.Linear(d_in, d_block)
        # for shared low-level feature extraction
        self.blocks_0 = nn.ModuleList(
            [nn.Sequential(nn.Linear(d_block, d_hidden), self.activation, nn.Linear(d_hidden, d_block)) for _ in range(n_block_l)])
        # for high-level feature extraction
        self.blocks_1 = nn.ModuleList(
            [nn.Sequential(nn.Linear(d_block, d_hidden), self.activation, nn.Linear(d_hidden, d_block)) for _ in range(n_block_h)])
        self.blocks_2 = nn.ModuleList(
            [nn.Sequential(nn.Linear(d_block, d_hidden), self.activation, nn.Linear(d_hidden, d_block)) for _ in range(n_block_h)])
        # output layer
        self.output_layer_1 = nn.Sequential(self.activation, nn.Linear(d_block, d_out))
        self.output_layer_2 = nn.Sequential(self.activation, nn.Linear(d_block, d_out))

    def forward(self, x):

        x = self.input_layer(x)
        x = self.activation(x)

        for block in self.blocks_0:
            if self.skip:
                x = x + block(x)
            else:
                x = block(x)
        x1, x2 = x.clone(), x.clone()
        del x

        # Lg branch
        for block in self.blocks_1:
            if self.skip:
                x1 = x1 + block(x1)
            else:
                x1 = block(x1)
        Lg = self.output_layer_1(x1)
        
        # Ld branch
        for block in self.blocks_2:
            if self.skip:
                x2 = x2 + block(x2)
            else:
                x2 = block(x2)
        Ld = self.output_layer_2(x2)
        
        return Lg, Ld