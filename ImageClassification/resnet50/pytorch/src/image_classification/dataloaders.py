import os
import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from functools import partial
import horovod.torch as hvd
DATA_BACKEND_CHOICES = ['pytorch']

def get_pytorch_train_loader(data_path, batch_size, num_classes, one_hot, workers=5, _worker_init_fn=None, fp16=False, memory_format=torch.contiguous_format):
    traindir = os.path.join(data_path, 'train')
    train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                ]))

    if hvd.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    elif torch.distributed.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=torch.distributed.get_world_size(), rank=torch.distributed.get_rank())
    elif deepspeed.comm.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=deepspeed.comm.get_world_size(), rank=deepspeed.comm.get_rank())

    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=workers, worker_init_fn=_worker_init_fn, pin_memory=True, sampler=train_sampler, collate_fn=partial(fast_collate, memory_format), drop_last=False)

    return PrefetchedWrapper(train_loader, num_classes, fp16, one_hot), len(train_loader)

def get_pytorch_val_loader(data_path, batch_size, num_classes, one_hot, workers=5, _worker_init_fn=None, fp16=False, memory_format=torch.contiguous_format):
    valdir = os.path.join(data_path, 'val')
    val_dataset = datasets.ImageFolder(
            valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                ]))

    if hvd.is_initialized():
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    elif torch.distributed.is_initialized():
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=torch.distributed.get_world_size(), rank=torch.distributed.get_rank())
    elif deepspeed.comm.is_initialized():
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=deepspeed.comm.get_world_size(), rank=deepspeed.comm.get_rank())

    val_loader = torch.utils.data.DataLoader(
            val_dataset,
            sampler=val_sampler,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, worker_init_fn=_worker_init_fn, pin_memory=True,
            collate_fn=partial(fast_collate, memory_format))
    return PrefetchedWrapper(val_loader, num_classes, fp16, one_hot), len(val_loader)


def fast_collate(memory_format, batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8 ).contiguous(memory_format=memory_format)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        tens = torch.from_numpy(nump_array)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array)

    return tensor, targets


class PrefetchedWrapper(object):
    def prefetched_loader(stream, loader, num_classes, fp16, one_hot):
        mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        if fp16:
            mean = mean.half()
            std = std.half()

        first = True

        for next_input, next_target in loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)
                if fp16:
                    next_input = next_input.half()
                    if one_hot:
                        next_target = expand(num_classes, torch.half, next_target)
                else:
                    next_input = next_input.float()
                    if one_hot:
                        next_target = expand(num_classes, torch.float, next_target)

                next_input = next_input.sub_(mean).div_(std)

            if not first:
                yield input, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target

        yield input, target

    def __init__(self, dataloader, num_classes, fp16, one_hot):
        self.dataloader = dataloader
        self.fp16 = fp16
        self.epoch = 0
        self.one_hot = one_hot
        self.num_classes = num_classes
        self.stream = torch.cuda.Stream()
    def __iter__(self):
        if (self.dataloader.sampler is not None and
            isinstance(self.dataloader.sampler,
                       torch.utils.data.distributed.DistributedSampler)):

            self.dataloader.sampler.set_epoch(self.epoch)
        self.epoch += 1
        return PrefetchedWrapper.prefetched_loader(self.stream, self.dataloader, self.num_classes, self.fp16, self.one_hot)

