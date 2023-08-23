import datetime
import logging
import math
import time
import torch
import numpy as np
from os import path as osp
from collections import OrderedDict

from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import build_model
from basicsr.utils import (AvgTimer, MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str,
                           init_tb_logger, init_wandb_logger, make_exp_dirs, mkdir_and_rename, scandir,)
from basicsr.utils.options import copy_opt_file, dict2str, parse_options

from basicsr.pruner import pruner_dict

def apply_mask_forward(model, mask):
    for name, m in model.named_modules():
        if name in mask:

            m.weight.data.mul_(mask[name])



def check_model(model):
    for name in model.state_dict():
        print(name)

def init_tb_loggers(opt):
    # initialize wandb logger before tensorboard logger to allow proper sync
    if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project')
                                                     is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, ('should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join(opt['root_path'], 'tb_logger', opt['name']))
    return tb_logger


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loaders = None, []
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = build_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
            train_loader = build_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info('Training statistics:'
                        f'\n\tNumber of train images: {len(train_set)}'
                        f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                        f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                        f'\n\tWorld size (gpu number): {opt["world_size"]}'
                        f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                        f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')
        elif phase.split('_')[0] == 'val':
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            logger.info(f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
            val_loaders.append(val_loader)
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loaders, total_epochs, total_iters


def load_resume_state(opt):
    resume_state_path = None
    if opt['auto_resume']:
        state_path = osp.join('experiments', opt['name'], 'training_states')
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                opt['path']['resume_state'] = resume_state_path
    else:
        if opt['path'].get('resume_state'):
            resume_state_path = opt['path']['resume_state']

    if resume_state_path is None:
        resume_state = None
    else:
        device_id = torch.cuda.current_device()
        resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))
        check_resume(opt, resume_state['iter'])
    return resume_state


def train_pipeline(root_path):

    grad_prune_dict, grad_FT_dict = {}, {}

    # parse options, set distributed setting, set ramdom seed
    opt, args = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # load resume states if necessary
    resume_state = load_resume_state(opt)
    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))
            pass

    # copy the yml file to the experiment root
    copy_opt_file(args.opt, opt['path']['experiments_root'])

    # WARNING: should not use get_root_logger in the above codes, including the called functions
    # Otherwise the logger will not be properly initialized
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    # initialize wandb and tb loggers
    tb_logger = init_tb_loggers(opt)

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = result

    # create model
    model = build_model(opt)
    if resume_state:  # resume training
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        # print('for L1 norm pruning, need to specify the pre-trained model with [pretrain_network_g] and [resume_state]')
        # raise NotImplementedError
        start_epoch = 0
        current_iter = 0



    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f"Wrong prefetch_mode {prefetch_mode}. Supported ones are: None, 'cuda', 'cpu'.")



    # @WJM: validate right before the pruning
    # if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
    if len(val_loaders) > 1:
        logger.warning('Multiple validation datasets are *only* supported by SRModel.')
    for val_loader in val_loaders:
        model.validation(val_loader, 0, tb_logger, opt['val']['save_img'])


    # @WJM: prune the model
    logger.info('Start pruning using L1 norm (one-shot pruning)')
    assert args.prune_method in ['L1', 'GReg-1', 'ASSL', 'IHT', 'IST','IHT-fast', 'IST-fast', 'IST-GReg', 'IST-GReg-fast'], 'ERROR: invalid args.prune_method!'
    class passer:pass
    passer.grad_prune_dict = grad_prune_dict

    passer.train_sampler = train_sampler
    passer.val_loaders = val_loaders
    passer.prefetcher = prefetcher
    passer.msg_logger = msg_logger
    passer.tb_logger = tb_logger
    pruner = pruner_dict[args.prune_method].Pruner(model=model, opt=opt, args=args, logger=logger, passer=passer)
    model_deepcopy = pruner.prune()  # get the pruned model as initialization for later finetuning
    # @WJM: zero out pruned weights
    if args.prune_method and args.wg == 'weight':
        apply_mask_forward(model.net_g, pruner.masks)


    # @WJM: validate right after the pruning
    # if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
    if len(val_loaders) > 1:
        logger.warning('Multiple validation datasets are *only* supported by SRModel.')
    for val_loader in val_loaders:
        model.validation(val_loader, 0, tb_logger, opt['val']['save_img'])


    # training (FT)
    logger.info(f'Start FT training from epoch: {start_epoch}, iter: {current_iter}')
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()

    for epoch in range(start_epoch, total_epochs + 1):
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_timer.record()

            current_iter += 1
            if current_iter > total_iters:
                break
            # update learning rate
            model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            # training
            model.feed_data(train_data)
            model.optimize_parameters(current_iter)

            if current_iter % args.grad_save_iters == 0:
                # compute the per-layer grad norm and var
                model_grad_status = []
                for name, m in model.net_g.named_modules():
                    if name in pruner.layers and pruner.pr[name] > 0:
                        layer_grad_status = []
                        grad_abs_vec = m.weight.grad.abs().flatten()
                        grad_norm = torch.norm(grad_abs_vec).detach().cpu().numpy()
                        grad_var = torch.var(grad_abs_vec).detach().cpu().numpy()
                        grad_mean = torch.mean(grad_abs_vec).detach().cpu().numpy()
                        grad_std = torch.std(grad_abs_vec).detach().cpu().numpy()
                        layer_grad_status.append(grad_norm)
                        layer_grad_status.append(grad_var)
                        layer_grad_status.append(grad_mean)
                        layer_grad_status.append(grad_std)
                        model_grad_status.append(layer_grad_status)

                # save the per-layer grad norm and var into the dict
                print('>>>@FT: model_grad_status shape, should be [#layer, 2(norm, var)]', np.array(model_grad_status).shape) # verified [103,2]
                grad_FT_dict[current_iter] = np.array(model_grad_status)

                # save the current dict into the file for visualization
                if current_iter % args.grad_write2file_iters == 0:
                    np.save(osp.join(opt['path']['log'], 'grad_FT_dict_%d.npy' % current_iter), grad_FT_dict)

            # @WJM: zero out the pruned weights
            if args.prune_method and args.wg == 'weight':
                apply_mask_forward(model.net_g, pruner.masks)


            iter_timer.record()
            if current_iter == 1:
                # reset start time in msg_logger for more accurate eta_time
                # not work in resume mode
                msg_logger.reset_start_time()
            # log
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            # save models and training states
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter + pruner.total_iter)

            # validation
            if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
                if len(val_loaders) > 1:
                    logger.warning('Multiple validation datasets are *only* supported by SRModel.')
                for val_loader in val_loaders:
                    model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])

                    # model.save(epoch, model.best_metric_results[val_loader.dataset.opt['name']]['psnr']['iter'])
                    model.save(epoch, current_iter + pruner.total_iter)

            data_timer.start()
            iter_timer.start()
            train_data = prefetcher.next()
        # end of iter

    # end of epoch

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if opt.get('val') is not None:
        for val_loader in val_loaders:
            model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
