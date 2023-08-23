import importlib
from copy import deepcopy
from os import path as osp

from basicsr.utils import get_root_logger, scandir
from basicsr.utils.registry import ARCH_REGISTRY

__all__ = ['build_network']

# automatically scan and import arch modules for registry
# scan all the files under the 'archs' folder and collect files ending with '_arch.py'
arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder) if v.endswith('_arch.py')]
# import all the arch modules
_arch_modules = [importlib.import_module(f'basicsr.archs.{file_name}') for file_name in arch_filenames]


def build_network(opt):
    opt = deepcopy(opt)
    # print('opt', opt) #  OrderedDict OrderedDict([('type', 'SwinIR'), ('upscale', 3), ('in_chans', 3), ('img_size', 48), ('window_size', 8), ('img_range', 1.0), ('depths', [6, 6, 6, 6, 6, 6]), ('embed_dim', 180), ('num_heads', [6, 6, 6, 6, 6, 6]), ('mlp_ratio', 2), ('upsampler', 'pixelshuffle'), ('resi_connection', '1conv')])
    # opt OrderedDict([('type', 'SwinIR'), ('upscale', 3), ('in_chans', 3), ('img_size', 48), ('window_size', 8), ('img_range', 1.0), ('depths', [6, 6, 6, 6, 6, 6]), ('embed_dim', 180), ('num_heads', [6, 6, 6, 6, 6, 6]), ('mlp_ratio', 2), ('upsampler', 'pixelshuffle'), ('resi_connection', '1conv')])
    network_type = opt.pop('type')
    net = ARCH_REGISTRY.get(network_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Network [{net.__class__.__name__}] is created.')
    return net
