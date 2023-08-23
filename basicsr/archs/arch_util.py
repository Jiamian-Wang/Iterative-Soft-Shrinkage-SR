import collections.abc
import math
import copy
import torch
import torchvision
import warnings
from distutils.version import LooseVersion
from itertools import repeat
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm

# from basicsr.ops.dcn import ModulatedDeformConvPack, modulated_deform_conv
from basicsr.utils import get_root_logger


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


def reshape_1d(matrix, m):
    # If not a nice multiple of m, fill with zeroes.
    if matrix.shape[1] % m > 0:
        mat = torch.cuda.FloatTensor(matrix.shape[0], matrix.shape[1] + (m-matrix.shape[1]%m)).fill_(0)
        mat[:, :matrix.shape[1]] = matrix
        shape = mat.shape
        return mat.view(-1,m),shape
    else:
        return matrix.view(-1,m), matrix.shape

class SparseConv(nn.Conv2d):
    """" implement N:M sparse convolution layer """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', N=2, M=8, args=None, **kwargs):
        self.N = N
        self.M = M

        if padding != 0 and kernel_size // 2 != padding:
            pass
        else:
            padding = kernel_size // 2

        super(SparseConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                         bias, padding_mode, **kwargs)

        self.priority_scores = nn.Parameter(torch.ones(self.M - 1), requires_grad=True)

        self.op_per_position = (kernel_size ** 2) * in_channels * out_channels

        self.pruning_units = nn.Parameter(
            torch.ones(self.M, self.weight.shape[0], self.weight.shape[1], self.weight.shape[2], self.weight.shape[3]),
            requires_grad=True)

        self.final_binary_scores = None
        self.final_priority_scores = None

    def compute_costs(self):
        scores = torch.clamp(self.priority_scores, 0, 1.03)
        # Priority-Ordered Pruning
        scores = torch.cumprod(scores, dim=0)

        binary_scores = (scores > 0.5).int()
        # Straight-Through Estimator
        binary_scores = binary_scores - scores.detach() + scores

        current_N = 1 + binary_scores.sum()

        spatial_size = self.h * self.w

        pruned_costs = (current_N / self.M) * self.op_per_position * spatial_size
        original_costs = self.op_per_position * spatial_size

        return pruned_costs, original_costs

    # Determine pruning units using magnitude of weights
    # Reference: https://github.com/NM-sparsity/NM-sparsity
    def update_pruning_units(self):
        weight = self.weight.clone()
        shape = weight.shape
        tensor_type = weight.type()
        weight = weight.float().contiguous()

        weight = weight.permute(2, 3, 0, 1).contiguous().view(shape[2] * shape[3] * shape[0], shape[1])
        mat, mat_shape = reshape_1d(weight, self.M)

        _, idxes = abs(mat).sort()
        ws = []
        mask = torch.cuda.IntTensor(weight.shape).fill_(0).view(-1, self.M)
        for n in range(1, self.M + 1):
            mask.scatter_(1, idxes[:, -n].unsqueeze(-1), n)

        mask.view(weight.shape)
        mask = mask.view(shape[2], shape[3], shape[0], shape[1]).permute(2, 3, 0, 1).contiguous()
        mask = mask.view(shape).type(tensor_type)

        pruning_units = []
        for i in range(self.M):
            # preserve only (i+1)-th largest weight (w.r.t magnitude) in each group of M weights
            preserved_weights = (mask == i + 1).int()
            pruning_units.append(preserved_weights)

        self.pruning_units.data = torch.stack(pruning_units, 0)

    # Fix scores when the target budget is reached
    def fix_scores(self):
        scores = torch.clamp(self.priority_scores, 0, 1.03)
        prob = torch.cumprod(scores, 0)
        binary = (prob > 0.5).int()
        self.final_binary_scores = binary
        self.final_priority_scores = copy.deepcopy(self.priority_scores)

        self.priority_scores.requires_grad = False

    # To eliminate the effect of moving average of optimizer
    def reassign_scores(self):
        if self.final_priority_scores is not None:
            self.priority_scores.data = self.final_priority_scores.data

    # Fix pruning masks for inference
    def get_fixed_mask(self):
        scores = torch.clamp(self.priority_scores, 0, 1.03)
        prob = torch.cumprod(scores, 0)
        binary = (prob > 0.5).int()

        mask = self.pruning_units[0]
        mask = mask + torch.sum(self.pruning_units[1:] * binary.view(-1, 1, 1, 1, 1), dim=0)
        self.mask = mask

    def forward(self, x):
        if self.training:
            if self.final_binary_scores is None:
                scores = torch.clamp(self.priority_scores, 0, 1.03)
                prob = torch.cumprod(scores, 0)
                binary = (prob > 0.5).int()
                mn_mask = binary - prob.detach() + prob

                mask = self.pruning_units[0]

                mask = mask + torch.sum(self.pruning_units[1:] * mn_mask.view(-1, 1, 1, 1, 1), dim=0)
            else:
                mask = self.pruning_units[0]

                mask = mask + torch.sum(self.pruning_units[1:] * self.final_binary_scores.view(-1, 1, 1, 1, 1), dim=0)

            out = F.conv2d(x, self.weight * mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            out = F.conv2d(x, self.weight * self.mask, self.bias, self.stride, self.padding, self.dilation, self.groups)

        self.h = out.shape[-2]
        self.w = out.shape[-1]

        return out

class ResidualBlockNoBN_SLS(nn.Module):
    """Residual block without BN.

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False, N=2, M=32):
        super(ResidualBlockNoBN_SLS, self).__init__()
        self.res_scale = res_scale
        self.conv1 = SparseConv(num_feat, num_feat, 3, 1, 1, bias=True, N=N, M=M)
        self.conv2 = SparseConv(num_feat, num_feat, 3, 1, 1, bias=True, N=N, M=M)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class Upsample_SLS(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat, N, M ):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(SparseConv(num_feat, 4 * num_feat, 3, 1, 1, N=N, M=M))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(SparseConv(num_feat, 9 * num_feat, 3, 1, 1, N=N, M=M))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample_SLS, self).__init__(*m)




def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

    # TODO, what if align_corners=False
    return output


def resize_flow(flow, size_type, sizes, interp_mode='bilinear', align_corners=False):
    """Resize a flow according to ratio or shape.

    Args:
        flow (Tensor): Precomputed flow. shape [N, 2, H, W].
        size_type (str): 'ratio' or 'shape'.
        sizes (list[int | float]): the ratio for resizing or the final output
            shape.
            1) The order of ratio should be [ratio_h, ratio_w]. For
            downsampling, the ratio should be smaller than 1.0 (i.e., ratio
            < 1.0). For upsampling, the ratio should be larger than 1.0 (i.e.,
            ratio > 1.0).
            2) The order of output_size should be [out_h, out_w].
        interp_mode (str): The mode of interpolation for resizing.
            Default: 'bilinear'.
        align_corners (bool): Whether align corners. Default: False.

    Returns:
        Tensor: Resized flow.
    """
    _, _, flow_h, flow_w = flow.size()
    if size_type == 'ratio':
        output_h, output_w = int(flow_h * sizes[0]), int(flow_w * sizes[1])
    elif size_type == 'shape':
        output_h, output_w = sizes[0], sizes[1]
    else:
        raise ValueError(f'Size type should be ratio or shape, but got type {size_type}.')

    input_flow = flow.clone()
    ratio_h = output_h / flow_h
    ratio_w = output_w / flow_w
    input_flow[:, 0, :, :] *= ratio_w
    input_flow[:, 1, :, :] *= ratio_h
    resized_flow = F.interpolate(
        input=input_flow, size=(output_h, output_w), mode=interp_mode, align_corners=align_corners)
    return resized_flow


# TODO: may write a cpp file
def pixel_unshuffle(x, scale):
    """ Pixel unshuffle.

    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.

    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


# class DCNv2Pack(ModulatedDeformConvPack):
#     """Modulated deformable conv for deformable alignment.
#
#     Different from the official DCNv2Pack, which generates offsets and masks
#     from the preceding features, this DCNv2Pack takes another different
#     features to generate offsets and masks.
#
#     ``Paper: Delving Deep into Deformable Alignment in Video Super-Resolution``
#     """
#
#     def forward(self, x, feat):
#         out = self.conv_offset(feat)
#         o1, o2, mask = torch.chunk(out, 3, dim=1)
#         offset = torch.cat((o1, o2), dim=1)
#         mask = torch.sigmoid(mask)
#
#         offset_absmean = torch.mean(torch.abs(offset))
#         if offset_absmean > 50:
#             logger = get_root_logger()
#             logger.warning(f'Offset abs mean is {offset_absmean}, larger than 50.')
#
#         if LooseVersion(torchvision.__version__) >= LooseVersion('0.9.0'):
#             return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
#                                                  self.dilation, mask)
#         else:
#             return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding,
#                                          self.dilation, self.groups, self.deformable_groups)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            'mean is more than 2 std from [a, b] in nn.init.trunc_normal_. '
            'The distribution of values may be incorrect.',
            stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        low = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [low, up], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * low - 1, 2 * up - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution.

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py

    The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# From PyTorch
def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple
