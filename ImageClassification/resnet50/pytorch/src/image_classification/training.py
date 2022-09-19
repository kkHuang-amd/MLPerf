import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from . import logger as log
from . import resnet as models
from . import gbn_resnet as gbn_models
from . import utils
import logging
import sys

import time
import statistics
from lars import LARS
from torch.nn.parallel import DistributedDataParallel as torch_DDP

#MLLogs v1.1
from mlperf_logger import configure_logger, log_start, log_end, log_event, set_seeds, get_rank, barrier
from mlperf_logging.mllog import constants    
import horovod.torch as hvd
import deepspeed
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

try:
    from apex import amp
    from apex.fp16_utils import *
    from apex.parallel import DistributedDataParallel as apex_DDP
    from apex.parallel.LARC import LARC
    from apex.optimizers import FusedLAMB, FusedSGD
except:
    pass


ACC_METADATA = {'unit': '%','format': ':.2f'}
IPS_METADATA = {'unit': 'img/s', 'format': ':.2f'}
TIME_METADATA = {'unit': 's', 'format': ':.5f'}
LOSS_METADATA = {'format': ':.5f'}



def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

class ModelAndLoss(nn.Module):
    def __init__(self, arch, loss, nhwc=False, pretrained_weights=None, cuda=True, fp16=False, memory_format=torch.contiguous_format, amp=False, local_rank=0, get_logs=False):
        super(ModelAndLoss, self).__init__()
        self.arch = arch
    
        print("=> creating model '{}'".format(arch))
        if nhwc:
            model = gbn_models.build_resnet(arch[0], arch[1], local_rank=local_rank, get_logs=get_logs)
        else:
            model = models.build_resnet(arch[0], arch[1])

        if pretrained_weights is not None:
            print("=> using pre-trained model from a file '{}'".format(arch))
            model.load_state_dict(pretrained_weights)

        if cuda:
            model = model.to("cuda", non_blocking=True)
            model = model.to(memory_format = memory_format)

        if fp16:
            model = network_to_half(model)

        self.use_amp = amp
        if self.use_amp:
            self.grad_scaler = GradScaler()
        # define loss function (criterion) and optimizer
        criterion = loss()

        if cuda:
            criterion = criterion.to("cuda", non_blocking=True)

        self.model = model
        self.loss = criterion

    def forward(self, data, target):
        if self.use_amp:
            with autocast():
                output = self.model(data)
                loss = self.loss(output, target)
        else:
            output = self.model(data)
            loss = self.loss(output, target)

        return loss, output

    def distributed(self, local_rank, nhwc, gpus):
        if nhwc:
            self.model = apex_DDP(self.model, gradient_predivide_factor=gpus/8.0, delay_allreduce=True, retain_allreduce_buffers=True)
        else:
            self.model = torch_DDP(self.model, device_ids=[local_rank], find_unused_parameters=True, broadcast_buffers=False, bucket_cap_mb=10)

    def load_model_state(self, state):
        if not state is None:
            self.model.load_state_dict(state)


def get_optimizer(model_and_loss, parameters, fp16, lr, momentum, weight_decay, local_rank=0,
                  nesterov=False,
                  state=None,
                  static_loss_scale=1., dynamic_loss_scale=False,
                  bn_weight_decay = False, get_logs= False, use_larc=False, use_lamb=False, use_lars=False):

    if bn_weight_decay:
        print(" ! Weight decay applied to BN parameters ")
        if use_larc:
            optimizer=torch.optim.Adam([v for n, v in parameters], lr, weight_decay=weight_decay)
            optimizer=LARC(optimizer)
        elif use_lamb:
            optimizer = FusedLAMB([v for n, v in parameters], lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
        else: 
            optimizer = torch.optim.SGD([v for n, v in parameters], lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    else:
        print(" ! Weight decay NOT applied to BN parameters ")
        bn_params = []
        rest_params = []
        for n, v in parameters:
            if 'bn' in n or 'bias' in n:
                bn_params.append(v)
            else:
                rest_params.append(v)
        print(len(bn_params))
        print(len(rest_params))
        if use_larc: 
            optimizer = torch.optim.SGD(model_and_loss.model.parameters(),
                                    lr,
                                    momentum=momentum,
                                    weight_decay=weight_decay,
                                    nesterov = nesterov)
            optimizer = LARC(optimizer)
           
        elif use_lamb:
            optimizer = FusedLAMB([{'params': bn_params, 'weight_decay' : 0},
                                     {'params': rest_params, 'weight_decay' : weight_decay}], lr, weight_decay=weight_decay)
        elif use_lars:
            if get_logs and local_rank == 0:
                log_event(key=constants.OPT_NAME, value='lars')
                log_event(key=constants.LARS_EPSILON, value=0)
                log_event(key=constants.LARS_OPT_WEIGHT_DECAY, value=weight_decay)
                log_event(key='lars_opt_momentum', value=momentum)
                log_event(key='lars_opt_base_learning_rate', value=lr)

            from fused_lars.optimizers import fused_lars
            optimizer = fused_lars.Fused_LARS([{'params': bn_params, 'weight_decay' : 0, 'is_skipped': True},
                              {'params': rest_params, 'weight_decay' : weight_decay, 'is_skipped': False}],
                              lr, momentum=momentum, weight_decay=weight_decay, trust_coefficient=0.001, eps=0.0, set_grad_none=True)
        else:
            optimizer = FusedSGD(model_and_loss.model.parameters(),
                                    lr,
                                    momentum=momentum,
                                    weight_decay=weight_decay,
                                    nesterov = nesterov)
    if fp16:
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=static_loss_scale,
                                   dynamic_loss_scale=dynamic_loss_scale,
                                   verbose=False)

    if not state is None:
        optimizer.load_state_dict(state)
    
    return optimizer


def lr_policy(lr_fn, logger=None):
    if logger is not None:
        logger.register_metric('lr', log.IterationMeter(), log_level=1)
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
    
        if logger is not None:
            logger.log_metric('lr', lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    

    return _alr


def lr_step_policy(base_lr, steps, decay_factor, warmup_length, local_rank=0, logger=None, get_logs=False):
    if get_logs and local_rank == 0:
        log_event(key=constants.OPT_BASE_LR, value=base_lr)
        log_event(key=constants.OPT_LR_DECAY_STEPS, value=steps)
        log_event(key=constants.OPT_LR_WARMUP_STEPS, value=warmup_length)
        log_event(key=constants.OPT_WEIGHT_DECAY, value=decay_factor)

    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            lr = base_lr
            for s in steps:
                if epoch >= s:
                    lr *= decay_factor
        return lr

    return lr_policy(_lr_fn, logger=logger)

def lr_poly_policy(base_lr, warmup_length, epochs, steps_per_epoch, local_rank=0,
                   use_lars= False, logger=None, get_logs=False, end_lr=0.0001):
    if get_logs and local_rank == 0:
        log_event(key=constants.OPT_BASE_LR, value=base_lr)
        log_event(key=constants.OPT_LR_WARMUP_STEPS, value=warmup_length)

    warmup_steps = warmup_length * steps_per_epoch
    train_steps = epochs * steps_per_epoch
    decay_steps = train_steps - warmup_steps + 1

    if get_logs and local_rank == 0:
        log_event(key=constants.OPT_LR_DECAY_STEPS, value=warmup_steps)
        log_event(key=constants.OPT_LR_DECAY_BOUNDARY_EPOCHS, value=train_steps)
        log_event(key=constants.OPT_LR_DECAY_BOUNDARY_STEPS, value=decay_steps)
        log_event(key="lars_opt_learning_rate_decay_poly_power", value=2)
        log_event(key="lars_opt_end_learning_rate", value=0.0001)

    def _lr_fn(iteration, epoch):
        global_step = (epoch * steps_per_epoch) + iteration
        if epoch < warmup_length:
            lr = base_lr * global_step / warmup_steps
        else:
            global_step = min(global_step - warmup_steps, decay_steps)
            lr = ((base_lr - end_lr) * ((1 - global_step / decay_steps) ** 2)) + end_lr
        return lr

    return lr_policy(_lr_fn, logger=logger)

def lr_linear_policy(base_lr, warmup_length, epochs, local_rank=0, logger=None, get_logs=False):
    if get_logs and local_rank == 0:
        log_event(key=constants.OPT_BASE_LR, value=base_lr)
        log_event(key=constants.OPT_LR_WARMUP_STEPS, value=warmup_length)

    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = base_lr * (1-(e/es))
        return lr

    return lr_policy(_lr_fn, logger=logger)


def lr_cosine_policy(base_lr, warmup_length, epochs, local_rank=0, logger=None, get_logs=False):
    if get_logs and local_rank == 0:
        log_event(key=constants.OPT_BASE_LR, value=base_lr)
        log_event(key=constants.OPT_LR_WARMUP_STEPS, value=warmup_length)

    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn, logger=logger)



def lr_exponential_policy(base_lr, warmup_length, epochs, local_rank=0, final_multiplier=0.001, logger=None, get_logs=False):
    if get_logs and local_rank == 0:
        log_event(key=constants.OPT_BASE_LR, value=base_lr)
        log_event(key=constants.OPT_LR_WARMUP_STEPS, value=warmup_length)

    es = epochs - warmup_length
    epoch_decay = np.power(2, np.log2(final_multiplier)/es)

    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            lr = base_lr * (epoch_decay ** e)
        return lr

    return lr_policy(_lr_fn, logger=logger)



def get_train_step(model_and_loss, optimizer, fp16, use_amp = False, batch_size_multiplier = 1):
    def _step(input, target, optimizer_step = True):
        input_var = Variable(input)
        target_var = Variable(target)
        loss, output = model_and_loss(input_var, target_var)
        prec1, prec5 = utils.accuracy(output.data, target, topk=(1, 5))

        reduced_loss = utils.reduce_tensor(loss.data)

        if deepspeed.comm.is_initialized():
            model_and_loss.model.backward(loss)
        elif fp16:
            optimizer.backward(loss)
        elif use_amp:
            model_and_loss.grad_scaler.scale(loss).backward()
            #with amp.scale_loss(loss, optimizer) as scaled_loss:
            #    scaled_loss.backward()
        else:
            loss.backward()

        if hvd.is_initialized():
            optimizer.synchronize()

        if optimizer_step:
            opt = optimizer
            for param_group in opt.param_groups:
                for param in param_group['params']:
                    param.grad /= batch_size_multiplier

            if deepspeed.comm.is_initialized():
                model_and_loss.model.step()
            elif use_amp:
                model_and_loss.grad_scaler.step(optimizer)
                model_and_loss.grad_scaler.update()
            else:
                optimizer.step()

            if deepspeed.comm.is_initialized():
                model_and_loss.model.optimizer.zero_grad()
            else:
                optimizer.zero_grad()

        torch.cuda.synchronize()

        return reduced_loss, prec1, prec5
    

    return _step

throughput=[]
def train(train_loader, model_and_loss, optimizer, lr_scheduler, fp16, logger, epoch, local_rank=0, use_amp=False, prof=-1, batch_size_multiplier=1, register_metrics=True, get_logs=False):
    if register_metrics and logger is not None:
        logger.register_metric('train.top1', log.AverageMeter(), log_level = 0)
        logger.register_metric('train.top5', log.AverageMeter(), log_level = 0)
        logger.register_metric('train.loss', log.AverageMeter(), log_level = 0)
        logger.register_metric('train.compute_ips', log.AverageMeter(), log_level=0)
        logger.register_metric('train.total_ips', log.AverageMeter(), log_level=0)
        logger.register_metric('train.data_time', log.AverageMeter(), log_level=1)
        logger.register_metric('train.compute_time', log.AverageMeter(), log_level=1)
    
    step = get_train_step(model_and_loss, optimizer, fp16, use_amp = use_amp, batch_size_multiplier = batch_size_multiplier)
    
    if get_logs and local_rank == 0:
        log_start(key=constants.EPOCH_START, metadata={'epoch_num': epoch + 1})
    
    #model_and_loss.train()
    end = time.time()

    if deepspeed.comm.is_initialized():
        model_and_loss.model.optimizer.zero_grad()
    else:
        optimizer.zero_grad()

    data_iter = enumerate(train_loader)
    if logger is not None:
        data_iter = logger.iteration_generator_wrapper(data_iter)
    for i, (input, target) in data_iter:
        bs = input.size(0)
        if deepspeed.comm.is_initialized():
            lr_scheduler(model_and_loss.model.optimizer, i, epoch)
        else:
            lr_scheduler(optimizer, i, epoch)
        data_time = time.time() - end

        if prof > 0:
            if i >= prof:
                break
        def is_gradient_accumulation_boundary():
            return ((i + 1) % batch_size_multiplier) == 0

        optimizer_step = ((i + 1) % batch_size_multiplier) == 0
        loss, prec1, prec5 = step(input, target, optimizer_step = optimizer_step)

        it_time = time.time() - end
        
        throughput.append(calc_ips(bs, it_time - data_time))

        if logger is not None:
            logger.log_metric('train.top1', to_python_float(prec1))
            logger.log_metric('train.top5', to_python_float(prec5))
            logger.log_metric('train.loss', to_python_float(loss))
            logger.log_metric('train.compute_ips', calc_ips(bs, it_time - data_time))
            logger.log_metric('train.total_ips', calc_ips(bs, it_time))
            logger.log_metric('train.data_time', data_time)
            logger.log_metric('train.compute_time', it_time - data_time)

        end = time.time()
    
    global imgs
    imgs=statistics.mean(throughput)

    if get_logs and local_rank == 0:
        log_end(key=constants.EPOCH_STOP, metadata={'epoch_num': epoch + 1})
        log_event(key='throughput', value={'images_sec': imgs}, metadata={'epoch_num' : epoch + 1})
    throughput.clear()

def get_val_step(model_and_loss):
    def _step(input, target):
        input_var = Variable(input)
        target_var = Variable(target)

        with torch.no_grad():
            loss, output = model_and_loss(input_var, target_var)

        prec1, prec5 = utils.accuracy(output.data, target, topk=(1, 5))

        reduced_loss = utils.reduce_tensor(loss.data)
        prec1 = utils.reduce_tensor(prec1)
        prec5 = utils.reduce_tensor(prec5)

        torch.cuda.synchronize()

        return reduced_loss, prec1, prec5

    return _step


val_accuracy=[]
def validate(val_loader, model_and_loss, fp16, logger, epoch, accuracy_threshold, local_rank=0, prof=-1, register_metrics=True, get_logs=False): 
    if register_metrics and logger is not None:
        logger.register_metric('val.top1',         log.AverageMeter(), log_level = 0)
        logger.register_metric('val.top5',         log.AverageMeter(), log_level = 0)
        logger.register_metric('val.loss',         log.AverageMeter(), log_level = 0)
        logger.register_metric('val.compute_ips',  log.AverageMeter(), log_level = 1)
        logger.register_metric('val.total_ips',    log.AverageMeter(), log_level = 1)
        logger.register_metric('val.data_time',    log.AverageMeter(), log_level = 1)
        logger.register_metric('val.compute_time', log.AverageMeter(), log_level = 1)

    step = get_val_step(model_and_loss)
    
    if get_logs and local_rank == 0:
        log_start(key=constants.EVAL_START, metadata={'epoch_num' : epoch + 1})

    top1 = log.AverageMeter()
    # switch to evaluation mode
    model_and_loss.eval()

    end = time.time()

    data_iter = enumerate(val_loader)
    if not logger is None:
        data_iter = logger.iteration_generator_wrapper(data_iter, val=True)

   
    for i, (input, target) in data_iter:
        bs = input.size(0)
        data_time = time.time() - end
        if prof > 0:
            if i > prof:
                break

        loss, prec1, prec5 = step(input, target)
        
        val_accuracy.append(to_python_float(prec1))

        it_time = time.time() - end

        top1.record(to_python_float(prec1), bs)
        if logger is not None:
            logger.log_metric('val.top1', to_python_float(prec1))
            logger.log_metric('val.top5', to_python_float(prec5))
            logger.log_metric('val.loss', to_python_float(loss))
            logger.log_metric('val.compute_ips', calc_ips(bs, it_time - data_time))
            logger.log_metric('val.total_ips', calc_ips(bs, it_time))
            logger.log_metric('val.data_time', data_time)
            logger.log_metric('val.compute_time', it_time - data_time)
            

        end = time.time()
    
    global acc
    acc=statistics.mean(val_accuracy)
    acc = float(acc)/100
    
    if acc>=accuracy_threshold:
        print("The epoch at we get threshold accuracy is ", epoch + 1)
        print('val.top1: ',acc)
        return epoch
    
    val_accuracy.clear()    

    #Abhinav - prevent calling this on each iteration
    model_and_loss.train()
    
    if get_logs and local_rank == 0:
        log_event(key=constants.EVAL_ACCURACY, value=acc, metadata={'epoch_num' : epoch + 1})
        log_end(key=constants.EVAL_STOP, metadata={'epoch_num' : epoch + 1})

    return top1.get_val()

    
# Train loop {{{
def calc_ips(batch_size, time):
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else hvd.size() if hvd.is_initialized() else deepspeed.comm.get_world_size() if deepspeed.comm.is_initialized() else 1
    tbs = world_size * batch_size
    return tbs/time

def train_loop(model_and_loss, optimizer, lr_scheduler, train_loader, val_loader, epochs, fp16, logger,
               should_backup_checkpoint,accuracy_threshold, eval_period, eval_offset, local_rank=0, global_rank=0, use_amp=False,
               batch_size_multiplier = 1,
               best_prec1 = 0, start_epoch = 0, prof = -1, skip_training = False, skip_validation = False, save_checkpoints = True, checkpoint_dir='./', get_logs=False):
    
       

    prec1 = -1
    start_epoch_num = start_epoch + 1
    epoch_iter = range(start_epoch, epochs)
    start_time=time.time()
    if logger is not None:
        epoch_iter = logger.epoch_generator_wrapper(epoch_iter)
    for epoch in epoch_iter:
        epoch_num = epoch + 1
        torch.cuda.empty_cache() # Abhinav added for larger batch sizes
        model_and_loss.train() # Optimization for one time call to train()
        
        if get_logs and local_rank == 0:
            log_start(key=constants.BLOCK_START, metadata={'first_epoch_num': start_epoch_num,  'epoch_count':epoch_num})

        if not skip_training:
            train(train_loader, model_and_loss, optimizer, lr_scheduler, fp16, logger, epoch, local_rank=local_rank, use_amp = use_amp, prof = prof, register_metrics=epoch==start_epoch, batch_size_multiplier=batch_size_multiplier, get_logs=get_logs)

        if epoch - eval_offset >= 0 and (epoch - eval_offset) % eval_period == 0:
            if not skip_validation:
                prec1 = validate(val_loader, model_and_loss, fp16, logger, epoch, accuracy_threshold, local_rank=local_rank, prof = prof, register_metrics=epoch==eval_offset, get_logs=get_logs)
                if acc>=accuracy_threshold:
                    print(" %s is the runtime in seconds" % (time.time() - start_time))
                    #status = 'success' if acc>=accuracy_threshold else 'failed'
                    if get_logs and local_rank==0:
                        log_event(key=constants.EVAL_ACCURACY, value=acc, metadata={'epoch_num' : epoch + 1})
                        log_end(key=constants.EVAL_STOP, metadata={'epoch_num' : epoch})
                        log_end(key=constants.RUN_STOP, metadata={'status': 'success'})
                        log_end(key=constants.BLOCK_STOP, metadata={'first_epoch_num': start_epoch_num, 'epoch_count':epoch_num})
                    return True

        

        if save_checkpoints and global_rank == 0:
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

            if should_backup_checkpoint(epoch):
                backup_filename = 'checkpoint-{}.pth.tar'.format(epoch + 1)
            else:
                abackup_filename = None
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'arch': model_and_loss.arch,
                'state_dict': model_and_loss.model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint_dir=checkpoint_dir, backup_filename=backup_filename)
           
    
        if get_logs and local_rank==0:
            log_end(key=constants.BLOCK_STOP, metadata={'first_epoch_num': start_epoch_num, 'epoch_count':epoch_num})
# }}}

