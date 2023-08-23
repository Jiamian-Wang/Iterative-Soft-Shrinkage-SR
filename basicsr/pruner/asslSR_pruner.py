import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
from decimal import Decimal
import os, copy, time, pickle, numpy as np, math
from .meta_pruner import MetaPruner
# import utility
import matplotlib.pyplot as plt
from tqdm import tqdm
from fnmatch import fnmatch, fnmatchcase
from .utils import get_score_layer, pick_pruned_layer
pjoin = os.path.join
tensor2list = lambda x: x.data.cpu().numpy().tolist()
tensor2array = lambda x: x.data.cpu().numpy()
totensor = lambda x: torch.Tensor(x)

import logging
from os import path as osp
from basicsr.utils import get_root_logger, get_time_str

class Pruner(MetaPruner):
    def __init__(self, model, opt, args, logger, passer):
        super(Pruner, self).__init__(model.net_g, args, logger, passer)
        # loader = passer.loader
        # ckp = passer.ckp
        # loss = passer.loss
        self.logprint = self.logger.info if logger else print
        self.train_sampler = passer.train_sampler
        self.prefetcher = passer.prefetcher
        self.val_loaders = passer.val_loaders
        self.tb_logger = passer.tb_logger
        self.msg_logger = passer.msg_logger

        # ************************** variables from RCAN ************************** 
        # self.scale = args.scale
        # self.ckp = ckp
        # self.loader_train = loader.loader_train
        # self.loader_test = loader.loader_test
        self.model = model
        self.epoch = 0
        self.opt = opt
        # self.loss = loss
        # self.optimizer = utility.make_optimizer(args, self.model)
        # self.error_last = 1e8
        # **************************************************************************

        # Reg related variables
        self.reg = {}
        self.delta_reg = {}
        self._init_reg()
        self.iter_update_reg_finished = {}
        self.iter_finish_pick = {}
        self.iter_stabilize_reg = math.inf
        self.hist_mag_ratio = {}
        self.w_abs = {}
        self.wn_scale = {}

        # init prune_state
        self.prune_state = 'update_reg'
        if args.greg_mode in ['part'] and args.same_pruned_wg_layers and args.same_pruned_wg_criterion in ['reg']:
            self.prune_state = "ssa" # sparsity structure alignment
            self._get_kept_wg_L1(model=self.model.net_g, align_constrained=False)
        
        # init pruned_wg/kept_wg if they can be determined right at the begining
        if args.greg_mode in ['part'] and self.prune_state in ['update_reg']:
            self._get_kept_wg_L1(model=self.model.net_g, align_constrained=False) # this will update the 'self.kept_wg', 'self.pruned_wg', 'self.pr'

    def _init_reg(self):
        for name, m in self.model.net_g.named_modules():
            if name in self.layers:
                if self.args.wg == 'weight':
                    self.reg[name] = torch.zeros_like(m.weight.data).flatten().cuda()
                else:
                    shape = m.weight.data.shape
                    self.reg[name] = torch.zeros(shape[0], shape[1]).cuda()

    def _greg_1(self, m, name):
        if self.pr[name] == 0:
            return True
        
        pruned = self.pruned_wg[name]
        if self.args.wg == "channel":
            self.reg[name][:, pruned] += self.args.reg_granularity_prune
        elif self.args.wg == "filter":
            self.reg[name][pruned, :] += self.args.reg_granularity_prune
        elif self.args.wg == 'weight':
            self.reg[name][pruned] += self.args.reg_granularity_prune
        else:
            raise NotImplementedError

        # when all layers are pushed hard enough, stop
        return self.reg[name].max() > self.args.reg_upper_limit

    def _greg_penalize_all(self, m, name):
        if self.pr[name] == 0:
            return True
        
        if self.args.wg == "channel":
            self.reg[name] += self.args.reg_granularity_prune
        elif self.args.wg == "filter":
            self.reg[name] += self.args.reg_granularity_prune
        elif self.args.wg == 'weight':
            self.reg[name] += self.args.reg_granularity_prune
        else:
            raise NotImplementedError

        # when all layers are pushed hard enough, stop
        return self.reg[name].max() > self.args.reg_upper_limit

    def _update_reg(self, skip=[]):
        for name, m in self.model.net_g.named_modules():
            if name in self.layers:                
                if name in self.iter_update_reg_finished.keys():
                    continue
                if name in skip:
                    continue

                # get the importance score (L1-norm in this case)
                out = get_score_layer(m, wg='weight', criterion='l1-norm')
                # self.w_abs[name], self.wn_scale[name] = out['l1-norm'], out['wn_scale']
                self.w_abs[name] = out['l1-norm']
                
                # update reg functions, two things:
                # (1) update reg of this layer (2) determine if it is time to stop update reg
                if self.args.greg_mode in ['part']:
                    finish_update_reg = self._greg_1(m, name)
                elif self.args.greg_mode in ['all']:
                    finish_update_reg = self._greg_penalize_all(m, name)

                # check prune state
                if finish_update_reg:
                    # after 'update_reg' stage, keep the reg to stabilize weight magnitude
                    self.iter_update_reg_finished[name] = self.total_iter
                    self.logprint(f"==> {self.layer_print_prefix[name]} -- Just finished 'update_reg'. Iter {self.total_iter}. pr {self.pr[name]}")

                    # check if all layers finish 'update_reg'
                    prune_state = "stabilize_reg"
                    for n, mm in self.model.net_g.named_modules():
                        if isinstance(mm, self.LEARNABLES):
                            if n not in self.iter_update_reg_finished:
                                prune_state = ''
                                break
                    if prune_state == "stabilize_reg":
                        self.prune_state = 'stabilize_reg'
                        self.iter_stabilize_reg = self.total_iter
                        self.logprint("==> All layers just finished 'update_reg', go to 'stabilize_reg'. Iter = %d" % self.total_iter)

    def _apply_reg(self):
        for name, m in self.model.net_g.named_modules():
            if name in self.layers and self.pr[name] > 0:
                reg = self.reg[name]  # [N, C]
                # m.wn_scale.grad += reg[:, 0] * m.wn_scale
                m.weight.grad += reg.view_as(m.weight.data) * m.weight
                # bias = False if isinstance(m.bias, type(None)) else True
                # if bias:
                #     m.bias.grad += reg[:, 0] * m.bias

    # def _merge_wn_scale_to_weights(self):
    #     '''Merge the learned weight normalization scale to the weights.
    #     '''
    #     for name, m in self.model.named_modules():
    #         if name in self.layers and hasattr(m, 'wn_scale'):
    #             m.weight.data = F.normalize(m.weight.data, dim=(1,2,3)) * m.wn_scale.view(-1,1,1,1)
    #             self.logprint(f'Merged weight normalization scale to weights: {name}')

    def _resume_prune_status(self, ckpt_path):
        raise NotImplementedError

    # def _save_model(self, filename):
    #     savepath = f'{self.ckp.dir}/model/{filename}'
    #     ckpt = {
    #         'pruned_wg': self.pruned_wg,
    #         'kept_wg': self.kept_wg,
    #         'model': self.model,
    #         'state_dict': self.model.state_dict(),
    #     }
    #     torch.save(ckpt, savepath)
    #     return savepath

    def prune(self):
        self.total_iter = 0

        log_file = osp.join(self.opt['path']['log'], f"train_{self.opt['name']}_{get_time_str()}.log")
        logger = get_root_logger(logger_name='prune', log_level=logging.INFO, log_file=log_file)

        if self.args.resume_path:  # args.resume_path is  None
            self._resume_prune_status(self.args.resume_path)
            self._get_kept_wg_L1(model=self.model.net_g, )  # get pruned and kept wg from the resumed model
            self.logprint("Resume model successfully: '{}'. Iter = {}. prune_state = {}".format(
                self.args.resume_path, self.total_iter, self.prune_state))
        
        while True:
            finish_prune = self.train() # there will be a break condition to get out of the infinite loop
            if finish_prune:
                return copy.deepcopy(self.model.net_g)

# ************************************************ The code below refers to RCAN ************************************************ #
    def train(self):

        self.train_sampler.set_epoch(self.epoch)
        self.prefetcher.reset()
        train_data = self.prefetcher.next()

        while train_data is not None:
            # data_timer.record()

            self.total_iter += 1

            # comment out, use the stopping criterion adapted from ASSL
            # self.current_iter += 1
            # if self.current_iter > self.total_iters:
            #     break

            self.model.keep_pruned_learning_rate()
            # training
            self.model.feed_data(train_data)

            # self.model.optimize_parameters(self.total_iter)
            l_total, loss_dict = self.model.optimize_parameters_beforeASSLfunc()



            # @mst: print
            if self.total_iter % self.args.print_interval == 0:
                self.logprint(f"Iter {self.total_iter} [prune_state: {self.prune_state} method: {self.args.prune_method} compare_mode: {self.args.compare_mode} greg_mode: {self.args.greg_mode}]  " + "-"*40)

            # @mst: regularization loss: sparsity structure alignment (SSA)
            if self.prune_state in ['ssa']:
                # n = len(self.constrained_layers)
                # soft_masks = torch.zeros(n, self.args.n_feats, requires_grad=True).cuda()
                # hard_masks = torch.zeros(n, self.args.n_feats, requires_grad=False).cuda()
                cnt = 0 # changed, to count the number of layers
                loss_reg_layer_sum = 0
                for name, m in self.model.net_g.named_modules():
                    if name in self.constrained_layers:
                        cnt += 1
                        # @WJM: if too many weights, sample to compute a threshold
                        if torch.numel(m.weight.data) > 5000:
                            # flatten weights
                            flatten_weights = torch.flatten(m.weight.data)
                            # take subset indecies
                            sample_indecies = torch.randperm(torch.numel(flatten_weights))[:30000]
                            # take subset
                            sample_weights = flatten_weights[sample_indecies]
                        else:
                            sample_weights = torch.flatten(m.weight.data)
                        _, indices = torch.sort(sample_weights)
                        # n_wg = m.weight.size(0)
                        n_wg = torch.numel(sample_weights)
                        n_pruned = min(math.ceil(self.pr[name] * n_wg), n_wg - 1) # do not prune all
                        thre = sample_weights[indices[n_pruned]]
                        soft_mask = torch.sigmoid(m.weight.data - thre)
                        # hard_masks[cnt] = m.wn_scale >= thre
                        loss_reg_layer_sum += -torch.inner((torch.flatten(soft_mask)), torch.flatten(soft_mask).t())
                # loss_reg = -torch.mm(soft_masks, soft_masks.t()).mean()
                loss_reg = loss_reg_layer_sum / cnt
                # loss_reg_hard = -torch.mm(hard_masks, hard_masks.t()).mean().data # only as an analysis metric, not optimized
                if self.total_iter % self.args.print_interval == 0:
                    logstr = f'Iter {self.total_iter} loss_recon {l_total.item():.4f} loss_reg (*{self.args.lw_spr}) {loss_reg.item():6f}'
                    self.logprint(logstr)
                l_total += loss_reg * self.args.lw_spr

                # for constrained Conv layers, at prune_state 'ssa', do not update their regularization co-efficients
                if self.total_iter % self.args.update_reg_interval == 0:
                    self._update_reg(skip=self.constrained_layers)

            l_total.backward()

            
            # @mst: update reg factors and apply them before optimizer updates
            if self.prune_state in ['update_reg'] and self.total_iter % self.args.update_reg_interval == 0:
                self._update_reg()

            # after reg is updated, print to check
            # if self.total_iter % self.args.print_interval == 0:
            #     self._print_reg_status()
        
            if self.args.apply_reg: # reg can also be not applied, as a baseline for comparison
                self._apply_reg()

            self.model.optimize_parameters_afterASSLfunc(loss_dict)

            # if (batch + 1) % self.args.print_every == 0:
            #     self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
            #         (batch + 1) * self.args.batch_size,
            #         len(self.loader_train.dataset),
            #         self.loss.display_loss(batch),
            #         timer_model.release(),
            #         timer_data.release()))
            # timer_data.tic()


            # including (1)msg_logger, (2)log, (3)save models (4)validation (5)prefetcher
            # iter_timer.record()
            if self.total_iter == 1:
                # reset start time in msg_logger for more accurate eta_time
                # not work in resume mode
                self.msg_logger.reset_start_time()
            # log
            if self.total_iter % self.opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': self.epoch, 'iter': self.total_iter}
                log_vars.update({'lrs': self.model.get_current_learning_rate()})
                # log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
                log_vars.update(self.model.get_current_log())
                self.msg_logger(log_vars)
            # save models and training states
            if self.total_iter % self.opt['logger']['save_checkpoint_freq'] == 0:
                self.logprint('Saving models and training states.')
                self.model.save(self.epoch, self.total_iter)
            # validation
            if self.opt.get('val') is not None and (self.total_iter % self.opt['val']['val_freq'] == 0):
                if len(self.val_loaders) > 1:
                    logger.warning('Multiple validation datasets are *only* supported by SRModel.')
                for val_loader in self.val_loaders:
                    self.model.validation(val_loader, self.total_iter, self.tb_logger, self.opt['val']['save_img'])
            # data_timer.start()
            # iter_timer.start()
            train_data = self.prefetcher.next()

            # @mst: at the end of 'ssa', switch prune_state to 'update_reg'
            if self.prune_state in ['ssa'] and self.total_iter == self.args.iter_ssa:
                self._get_kept_wg_L1(model=self.model.net_g, align_constrained=False) # this will update the pruned_wg/kept_wg for constrained Conv layers
                self.prune_state = 'update_reg'
                self.logprint(f'==> Iter {self.total_iter} prune_state "ssa" is done, get pruned_wg/kept_wg, switch to {self.prune_state}.')

            # @mst: exit of reg pruning loop
            if self.prune_state in ["stabilize_reg"] and self.total_iter - self.iter_stabilize_reg == self.args.stabilize_reg_interval:
                self.logprint(f"==> 'stabilize_reg' is done. Iter {self.total_iter}.About to prune and build new model. Testing...")

                if self.args.greg_mode in ['all']:
                    self._get_kept_wg_L1(model=self.model.net_g, align_constrained=False)
                    self.logprint(f'==> Get pruned_wg/kept_wg.')

                # self._merge_wn_scale_to_weights()
                self._prune_and_build_new_model()
                # path = self._save_model('model_just_finished_prune.pt')
                # self.logprint(f"==> Pruned and built a new model. Ckpt saved: '{path}'. Testing...")
                self.model.save(self.epoch, self.total_iter)
                return True

        # self.loss.end_log(len(self.loader_train))
        # self.error_last = self.loss.log[-1, -1]
        # self.optimizer.schedule() # use fixed LR in pruning
        self.epoch += 1

    # def _print_reg_status(self):
    #     self.logprint('************* Regularization Status *************')
    #     for name, m in self.model.named_modules():
    #         if name in self.layers and self.pr[name] > 0:
    #             logstr = [self.layer_print_prefix[name]]
    #             logstr += [f"reg_status: min {self.reg[name].min():.5f} ave {self.reg[name].mean():.5f} max {self.reg[name].max():.5f}"]
    #             out = get_score_layer(m, wg='wight', criterion='wn_scale')
    #             w_abs, wn_scale = out['l1-norm'], out['wn_scale']
    #             pruned, kept = pick_pruned_layer(score=wn_scale, pr=self.pr[name], sort_mode='min')
    #             avg_mag_pruned, avg_mag_kept = np.mean(w_abs[pruned]), np.mean(w_abs[kept])
    #             avg_scale_pruned, avg_scale_kept = np.mean(wn_scale[pruned]), np.mean(wn_scale[kept])
    #             logstr += ["average w_mag: pruned %.6f kept %.6f" % (avg_mag_pruned, avg_mag_kept)]
    #             logstr += ["average wn_scale: pruned %.6f kept %.6f" % (avg_scale_pruned, avg_scale_kept)]
    #             logstr += [f'Iter {self.total_iter}']
    #             logstr += [f'cstn' if name in self.constrained_layers else 'free']
    #             logstr += [f'pr {self.pr[name]}']
    #             self.logprint(' | '.join(logstr))
    #     self.logprint('*************************************************')
        


    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]