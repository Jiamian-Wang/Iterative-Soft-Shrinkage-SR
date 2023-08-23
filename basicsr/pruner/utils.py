import torch, torch.nn as nn
from collections import OrderedDict
from fnmatch import fnmatch, fnmatchcase
import math, numpy as np, copy
import re
tensor2list = lambda x: x.data.cpu().numpy().tolist()
tensor2array = lambda x: x.data.cpu().numpy()
totensor = lambda x: torch.Tensor(x)

def get_pr_layer(base_pr, layer_name, layer_index, skip=[], compare_mode='local'):
    """ 'base_pr' example: '[0-4:0.5, 5:0.6, 8-10:0.2]', 6, 7 not mentioned, default value is 0
    """
    if compare_mode in ['global']:
        pr = 1e-20 # a small positive value to indicate this layer will be considered for pruning, will be replaced
    elif compare_mode in ['local']:
        pr = base_pr[layer_index]

    # if layer name matchs the pattern pre-specified in 'skip', skip it (i.e., pr = 0)
    for p in skip:
        if fnmatch(layer_name, p):
            pr = 0
    return pr

def get_pr_model(layers, base_pr, skip=[], compare_mode='local'):
    """Get layer-wise pruning ratio for a model.
    """
    pr = OrderedDict()
    if isinstance(base_pr, str):
        ckpt = torch.load(base_pr)
        pruned, kept = ckpt['pruned_wg'], ckpt['kept_wg']
        for name in pruned:
            num_pruned, num_kept = len(pruned[name]), len(kept[name])
            pr[name] = float(num_pruned) / (num_pruned + num_kept)
        print(f"==> Load base_pr model successfully and inherit its pruning ratio: '{base_pr}'.")
    elif isinstance(base_pr, (float, list)):
        if compare_mode in ['global']:
            assert isinstance(base_pr, float)
            pr['model'] = base_pr
        for name, layer in layers.items():
            pr[name] = get_pr_layer(base_pr, name, layer.index, skip=skip, compare_mode=compare_mode)
        print(f"==> Get pr (pruning ratio) for pruning the model, done (pr may be updated later).")
    else:
        raise NotImplementedError
    return pr

def get_constrained_layers(layers, constrained_pattern):
    constrained_layers = []
    for name, _ in layers.items():
        for p in constrained_pattern:
            if fnmatch(name, p):
                constrained_layers += [name]
    return constrained_layers

def adjust_pr(layers, pr, pruned, kept, num_pruned_constrained, constrained):
    """The real pr of a layer may not be exactly equal to the assigned one (i.e., raw pr) due to various reasons (e.g., constrained layers). 
    Adjust it here, e.g., averaging the prs for all constrained layers. 
    """
    pr, pruned, kept = copy.deepcopy(pr), copy.deepcopy(pruned), copy.deepcopy(kept)

    # @WJM: to compute global pr, this may yield different pr value in N:M case
    global_wg_num, global_prunedwg_num, global_pr = 0, 0, 0
    for name, layer in layers.items():
        if name in constrained:
            # -- averaging within all constrained layers to keep the total num of pruned weight groups still the same
            num_pruned = int(num_pruned_constrained / len(constrained))
            # --
            pr[name] = num_pruned / len(layer.score)
            order = pruned[name] + kept[name]
            pruned[name], kept[name] = order[:num_pruned], order[num_pruned:]
        else:
            num_pruned = len(pruned[name])
            pr[name] = num_pruned / len(layer.score)

            global_prunedwg_num += num_pruned
            global_wg_num += len(layer.score)
    global_pr = global_prunedwg_num / global_wg_num

    return pr, global_pr,  pruned, kept

def set_same_pruned(model, pr, N2M_granularity,  pruned_wg, kept_wg, constrained, wg='filter', criterion='l1-norm', sort_mode='min'):
    """Set pruned wgs of some layers to the same indices.
    """
    pruned_wg, kept_wg = copy.deepcopy(pruned_wg), copy.deepcopy(kept_wg)
    pruned = None
    for name, m in model.named_modules():
        if name in constrained:
            if pruned is None:
                score = get_score_layer(m, wg=wg, criterion=criterion)['score']
                pruned, kept = pick_pruned_layer(score=score, N2M_granularity=N2M_granularity, pr=pr[name], sort_mode=sort_mode)
                pr_first_constrained = pr[name]
            assert pr[name] == pr_first_constrained
            pruned_wg[name], kept_wg[name] = pruned, kept
    return pruned_wg, kept_wg

def get_score_layer(module, wg='filter', criterion='l1-norm'):
    r"""Get importance score for a layer.

    Return:
        out (dict): A dict that has key 'score', whose value is a numpy array
    """
    # -- define any scoring scheme here as you like
    shape = module.weight.data.shape
    if wg == "channel":
        l1 = module.weight.abs().mean(dim=[0, 2, 3]) if len(shape) == 4 else module.weight.abs().mean(dim=0)
    elif wg == "filter":
        l1 = module.weight.abs().mean(dim=[1, 2, 3]) if len(shape) == 4 else module.weight.abs().mean(dim=1)
    elif wg == "weight":
        l1 = module.weight.abs().flatten()
    # --

    out = {}
    out['l1-norm'] = tensor2array(l1)
    out['wn_scale'] = tensor2array(module.wn_scale.abs()) if hasattr(module, 'wn_scale') else [1e30] * module.weight.size(0)
    # 1e30 to indicate this layer will not be pruned because of its unusually high scores
    out['score'] = out[criterion]
    return out


def pick_pruned_layer(score, N2M_granularity, pr=None,  threshold=None, sort_mode='min', weight_shape=None):
    r"""Get the indices of pruned weight groups in a layer.

    Return:
        pruned (list)
        kept (list)
    """
    score = np.array(score)
    num_total = len(score)
    max_pruned = int(num_total * 0.995)  # This 0.995 is empirically set
    if sort_mode in ['rand']:
        assert pr is not None
        num_pruned = min(math.ceil(pr * num_total), max_pruned)  # do not prune all
        order = np.random.permutation(num_total).tolist()

        pruned, kept = order[:num_pruned], order[num_pruned:]

    elif sort_mode in ['min', 'max', 'ascending', 'descending']:
        num_pruned = math.ceil(pr * num_total) if threshold is None else len(np.where(score < threshold)[0])
        num_pruned = min(num_pruned, max_pruned)  # Do not prune all
        if sort_mode in ['min', 'ascending']:
            order = np.argsort(score).tolist()
        elif sort_mode in ['max', 'descending']:
            order = np.argsort(score)[::-1].tolist()

        pruned, kept = order[:num_pruned], order[num_pruned:]

    elif re.match('min_\d+:\d+', sort_mode):
        # See https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/
        # Currently, only unstructured pruning supports such M:N sparsity pattern
        # E.g., 'mode' = 'min_2:4'
        N_raw, M_raw = [int(x) for x in sort_mode.split('_')[1].split(':')]

        # @WJM: use the N2M granularity, e.g., 180, to help determine a new M
        # in case there is a M value that is not dividable by #weights
        assert N2M_granularity > M_raw, 'N2M_granularity(GCP) should be larger than M'
        M, N = M_raw, N_raw
        while N2M_granularity % M: # when M is not dividable by N2M_granularity
            M += 1
            N += 1
        score = score.reshape(-1, M)

        indices_all = np.argsort(score, axis=-1)
        indices_p = indices_all[:, :N]
        pruned = []
        for row, col in enumerate(indices_p):
            pruned += (row * M + col).tolist()
        # out = np.array(out)

        indices_k = indices_all[:, N:]
        kept = []
        for row, col in enumerate(indices_k):
            kept += (row * M + col).tolist()

    else:
        raise NotImplementedError

    assert isinstance(pruned, list) and isinstance(kept, list)
    return pruned, kept

# @WJM: add an efficient version
def fast_pick_pruned_layer(score, N2M_granularity, pr=None,  threshold=None, sort_mode='min', weight_shape=None):
    r"""Get the indices of pruned weight groups in a layer.

    Return:
        pruned (list)
        kept (list)
    """
    score = np.array(score)
    num_total = len(score)
    if num_total >= 32400:
        score_sample = np.random.choice(score, size=32400, replace=False)
        num_total = 32400
    else:
        score_sample = score
    # max_pruned = int( num_total * 0.995)  # This 0.995 is empirically set
    if sort_mode in ['rand']:
        assert pr is not None
        num_pruned = min(math.ceil(pr * num_total), max_pruned)  # do not prune all
        order = np.random.permutation(num_total).tolist()

        pruned, kept = order[:num_pruned], order[num_pruned:]

    elif sort_mode in ['min', 'max', 'ascending', 'descending']:
        num_pruned = math.ceil(pr * num_total) if threshold is None else len(np.where(score < threshold)[0])
        # num_pruned = min(num_pruned, max_pruned)  # Do not prune all
        if sort_mode in ['min', 'ascending']:
            # order = np.argsort(score).tolist()
            score_sort = np.sort(score_sample)
            threshold = score_sort[num_pruned]
            pruned = np.argwhere(score < threshold).squeeze().tolist()
            kept = list(set(range(len(score))) - set(pruned))
        elif sort_mode in ['max', 'descending']:
            # order = np.argsort(score)[::-1].tolist()
            raise NotImplementedError
        # pruned, kept = order[:num_pruned], order[num_pruned:]

    elif re.match('min_\d+:\d+', sort_mode):
        # See https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/
        # Currently, only unstructured pruning supports such M:N sparsity pattern
        # E.g., 'mode' = 'min_2:4'
        N_raw, M_raw = [int(x) for x in sort_mode.split('_')[1].split(':')]

        # @WJM: use the N2M granularity, e.g., 180, to help determine a new M
        # in case there is a M value that is not dividable by #weights
        assert N2M_granularity > M_raw, 'N2M_granularity(GCP) should be larger than M'
        M, N = M_raw, N_raw
        while N2M_granularity % M: # when M is not dividable by N2M_granularity
            M += 1
            N += 1
        score = score.reshape(-1, M)

        indices_all = np.argsort(score, axis=-1)
        indices_p = indices_all[:, :N]
        pruned = []
        for row, col in enumerate(indices_p):
            pruned += (row * M + col).tolist()
        # out = np.array(out)

        indices_k = indices_all[:, N:]
        kept = []
        for row, col in enumerate(indices_k):
            kept += (row * M + col).tolist()

    else:
        raise NotImplementedError

    assert isinstance(pruned, list) and isinstance(kept, list)
    return pruned, kept


def pick_pruned_model(model, layers, raw_pr, N2M_granularity, wg='filter', criterion='l1-norm', compare_mode='local', sort_mode='min', constrained=[], align_constrained=False):
    r"""Pick pruned weight groups for a model.
    Args:
        layers: an OrderedDict, key is layer name

    Return:
        pruned (OrderedDict): key is layer name, value is the pruned indices for the layer
        kept (OrderedDict): key is layer name, value is the kept indices for the layer
    """
    assert sort_mode in ['rand', 'min', 'max'] or re.match('min_\d+:\d+', sort_mode)
    assert compare_mode in ['global', 'local']
    pruned_wg, kept_wg = OrderedDict(), OrderedDict()
    all_scores, num_pruned_constrained = [], 0


    # iter to get importance score for each layer
    for name, module in model.named_modules():
        if name in layers:
            layer = layers[name]
            out = get_score_layer(module, wg=wg, criterion=criterion)
            score = out['score']
            layer.score = score
            if raw_pr[name] > 0: # pr > 0 indicates we want to prune this layer so its score will be included in the <all_scores>
                all_scores = np.append(all_scores, score)

            # local pruning
            if compare_mode in ['local']:
                assert isinstance(raw_pr, dict)
                pruned_wg[name], kept_wg[name] = pick_pruned_layer(score, N2M_granularity, raw_pr[name], sort_mode=sort_mode)
                if name in constrained: 
                    num_pruned_constrained += len(pruned_wg[name])
    
    # global pruning
    if compare_mode in ['global']:
        num_total = len(all_scores)
        num_pruned = min(math.ceil(raw_pr['model'] * num_total), num_total - 1) # do not prune all
        if sort_mode == 'min':
            threshold = sorted(all_scores)[num_pruned] # in ascending order
        elif sort_mode == 'max':
            threshold = sorted(all_scores)[::-1][num_pruned] # in decending order
        print(f'#all_scores: {len(all_scores)} threshold:{threshold:.6f}')

        for name, layer in layers.items():
            if raw_pr[name] > 0:
                if sort_mode in ['rand']:
                    pass
                elif sort_mode in ['min', 'max']:
                    pruned_wg[name], kept_wg[name] = pick_pruned_layer(layer.score, N2M_granularity, pr=None, threshold=threshold, sort_mode=sort_mode)
            else:
                pruned_wg[name], kept_wg[name] = [], list(range(len(layer.score)))
            if name in constrained:
                num_pruned_constrained += len(pruned_wg[name])
    
    # adjust pr/pruned/kept
    pr, global_pr, pruned_wg, kept_wg = adjust_pr(layers, raw_pr, pruned_wg, kept_wg, num_pruned_constrained, constrained)
    print(f'==> Adjust pr/pruned/kept, done.')

    if align_constrained:
        pruned_wg, kept_wg = set_same_pruned(model, pr, set_same_pruned, pruned_wg, kept_wg, constrained,
                                                wg=wg, criterion=criterion, sort_mode=sort_mode)
    
    return pr, global_pr, pruned_wg, kept_wg


# @WJM: for N:M sparsity
def get_N2M_granularity(model, layers, pick_pruned):
    r"""Get the N:M granularity for the model by iterating the layers (counting the #weights/layer)
    only takes effect when pick_pruned is N:M mode"""
    numel_ls = []
    if re.match('min_\d+:\d+', pick_pruned):
        for name, m in model.named_modules():
            if name in layers:
                numel_ls.append(m.weight.numel()) # count the #weights in this layer
        # compute GCM
        N2M_granularity = np.gcd.reduce(np.array(numel_ls))
        print('>>> N2M_granularity is', N2M_granularity)
    else:
        N2M_granularity = 0
    return N2M_granularity

def get_next_learnable(layers, layer_name, n_conv_within_block=3):
    r"""Get the next learnable layer for the layer of 'layer_name', chosen from 'layers'.
    """
    current_layer = layers[layer_name]

    # for standard ResNets on ImageNet
    if hasattr(current_layer, 'block_index'):
        block_index = current_layer.block_index
        if block_index == n_conv_within_block - 1:
            return None
    
    for name, layer in layers.items():
        if layer.type == current_layer.type and layer.index == current_layer.index + 1:
            return name
    return None

def get_prev_learnable(layers, layer_name):
    r"""Get the previous learnable layer for the layer of 'layer_name', chosen from 'layers'.
    """
    current_layer = layers[layer_name]

    # for standard ResNets on ImageNet
    if hasattr(current_layer, 'block_index'):
        block_index = current_layer.block_index
        if block_index in [None, 0, -1]: # 1st conv, 1st conv in a block, 1x1 shortcut layer
            return None
    
    for name, layer in layers.items():
        if layer.index == current_layer.index - 1:
            return name
    return None

def get_next_bn(model, layer_name):
    r"""Get the next bn layer for the layer of 'layer_name', chosen from 'model'.
    Return the bn module instead of its name.
    """
    just_passed = False
    for name, module in model.named_modules():
        if name == layer_name:
            just_passed = True
        if just_passed and isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            return module
    return None

def replace_module(model, name, new_m):
    """Replace the module <name> in <model> with <new_m>
    E.g., 'module.layer1.0.conv1' ==> model.__getattr__('module').__getattr__("layer1").__getitem__(0).__setattr__('conv1', new_m)
    """
    obj = model
    segs = name.split(".")
    for ix in range(len(segs)):
        s = segs[ix]
        if ix == len(segs) - 1: # the last one
            if s.isdigit():
                obj.__setitem__(int(s), new_m)
            else:
                obj.__setattr__(s, new_m)
            return
        if s.isdigit():
            obj = obj.__getitem__(int(s))
        else:
            obj = obj.__getattr__(s)

def get_kept_filter_channel(layers, layer_name, pr, kept_wg, wg='filter'):
    """Considering layer dependency, get the kept filters and channels for the layer of 'layer_name'.
    """
    current_layer = layers[layer_name]
    if wg in ["channel"]:
        kept_chl = kept_wg[layer_name]
        next_learnable = get_next_learnable(layers, layer_name)
        kept_filter = list(range(current_layer.module.weight.size(0))) if next_learnable is None else kept_wg[next_learnable]
    elif wg in ["filter"]:
        kept_filter = kept_wg[layer_name]
        prev_learnable = get_prev_learnable(layers, layer_name)
        if (prev_learnable is None) or pr[prev_learnable] == 0: 
            # In the case of SR networks, tail, there is an upsampling via sub-pixel. 'self.pr[prev_learnable_layer] == 0' can help avoid it. 
            # Not using this, the code will report error.
            kept_chl = list(range(current_layer.module.weight.size(1)))
        else:
            kept_chl = kept_wg[prev_learnable]
    
    # sort to make the indices be in ascending order 
    kept_filter.sort()
    kept_chl.sort()
    return kept_filter, kept_chl

def get_masks(layers, pruned_wg):
    """Get masks for unstructured pruning.
    """
    masks = OrderedDict()
    for name, layer in layers.items():
        mask = torch.ones(layer.shape).cuda().flatten()
        mask[pruned_wg[name]] = 0
        masks[name] = mask.view(layer.shape)
    return masks