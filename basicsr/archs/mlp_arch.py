import torch
import torch.nn as nn
import math
from basicsr.utils.registry import ARCH_REGISTRY



class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale=2, num_feat=10, num_out_ch=3, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)


@ARCH_REGISTRY.register()
class MLP(nn.Module):
    def __init__(self, d_image, upscale, n_fc, width=0, n_param=0, branch_layer_out_dim=[], act='relu', dropout=0):
        super(MLP, self).__init__()



        # activation func
        if act == 'relu':
            activation = nn.ReLU()
        elif act == 'lrelu':
            activation = nn.LeakyReLU()
        elif act == 'linear':
            activation = nn.Identity()
        else:
            raise NotImplementedError
        
        n_middle = n_fc - 2
        if width == 0:
            # Given total num of parameters budget, calculate the width: n_middle * width^2 + width * (d_image + n_class) = n_param 
            assert n_param > 0
            Delta = (d_image + upscale) * (d_image + upscale) + 4 * n_middle * n_param
            width = (math.sqrt(Delta) - d_image - upscale) / 2 / n_middle
            width = int(width)
            print("FC net width = %s" % width)

        # build the stem net
        net = [nn.Linear(d_image, width), activation]
        for i in range(n_middle):
            net.append(nn.Linear(width, width))
            if dropout and n_middle - i <= 2: # the last two middle fc layers will be applied with dropout
                net.append(nn.Dropout(dropout))
            net.append(activation)

        # net.append(nn.Linear(width, n_class))
        self.net = nn.Sequential(*net)

        self.upsample = UpsampleOneStep(scale=upscale, num_feat=width, num_out_ch=3)
        
        # build branch layers
        branch = []
        for x in branch_layer_out_dim:
            branch.append(nn.Linear(width, x))
        self.branch = nn.Sequential(*branch) # so that the whole model can be put on cuda
        self.branch_layer_ix = []
    
    def forward(self, img, branch_out=False, mapping=False):
        '''
            <branch_out>: if output the internal features
            <mapping>: if the internal features go through a mapping layer
        '''
        if not branch_out:

            img_bs, img_dp, img_h, img_w = img.size()
            img = img.view(img_bs, -1)

            # return self.net(img)
            mlp_out = self.net(img)
            mlp_out = mlp_out.view(img_bs, img_dp, img_h, img_w)
            upsample_out = self.upsample(mlp_out)
            return upsample_out
        else:
            out = []
            start = 0
            y = img.view(img.size(0), -1)
            keys = [int(x) for x in self.branch_layer_ix]
            for i in range(len(keys)):
                end = keys[i] + 1
                y = self.net[start:end](y)
                y_branch = self.branch[i](y) if mapping else y
                out.append(y_branch)
                start = end
            y = self.net[start:](y)
            out.append(y)
            return out


# Refer to: A Signal Propagation Perspective for Pruning Neural Networks at Initialization (ICLR 2020).
# https://github.com/namhoonlee/spp-public/blob/32bde490f19b4c28843303f1dc2935efcd09ebc9/spp/network.py#L108
# def mlp_7_linear(**kwargs):
#     return FCNet(d_image=1024, upscale=2, n_fc=7, width=100, act='relu')

if __name__ == '__main__':
    upscale = 2
    height=64
    width=64
    model = MLP(
        d_image=64, upscale=2, n_fc=7, width=100, act='relu')
    print(model)

    x = torch.randn((1, 3, height, width))
    x = model(x)
    print(x.shape)