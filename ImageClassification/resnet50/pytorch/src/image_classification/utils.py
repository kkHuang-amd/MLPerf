import os
import numpy as np
import torch
import shutil
import torch.distributed as dist
import deepspeed
import horovod.torch as hvd


def should_backup_checkpoint(args):
    def _sbc(epoch):
        return args.gather_checkpoints and (epoch < 10 or epoch % 10 == 0)
    return _sbc


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', checkpoint_dir='./', backup_filename=None):
    if (not torch.distributed.is_initialized() and not hvd.is_initialized()) or (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0) or (hvd.is_initialized() and hvd.rank() == 0):
        filename = os.path.join(checkpoint_dir, filename)
        print("SAVING {}".format(filename))
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(checkpoint_dir, 'model_best.pth.tar'))
        if backup_filename is not None:
            shutil.copyfile(filename, os.path.join(checkpoint_dir, backup_filename))


def timed_generator(gen):
    start = time.time()
    for g in gen:
        end = time.time()
        t = end - start
        yield g, t
        start = time.time()


def timed_function(f):
    def _timed_function(*args, **kwargs):
        start = time.time()
        ret = f(*args, **kwargs)
        return ret, time.time() - start
    return _timed_function


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    #print("****************************\nres={}\n**************************\n".format(res))
    return res


def reduce_tensor(tensor):
    if hvd.is_initialized():
        rt = hvd.allreduce(tensor)
    elif deepspeed.comm.is_initialized():
        rt = tensor.clone()
        deepspeed.comm.all_reduce(rt, op=deepspeed.comm.ReduceOp.AVG)
    elif torch.distributed.is_initialized():
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= torch.distributed.get_world_size()
    else:
        rt = tensor
    return rt
