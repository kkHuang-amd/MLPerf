# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
helper class that supports empty tensors on some nn functions.

Ideally, add support directly in PyTorch to empty tensors in
those functions.

This can be removed once https://github.com/pytorch/pytorch/issues/12013
is implemented
"""

import math
import torch
from torch.nn.modules.utils import _ntuple


class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None



class Conv2d(torch.nn.Conv2d):
    def forward(self, x):
        if x.numel() > 0:
            before_pad = x.shape[0]
            if before_pad>8:
                #print('>>>MYDEBUG conv in',before_pad,x.size(),flush=True)
                new_bs=32*((before_pad+32-1)//32)
                x = torch.nn.functional.pad(x, (0,0,0,0,0,0,0,new_bs-before_pad))
                #print('>>>MYDEBUG conv pad in',before_pad,x.size(),flush=True)
            out = torch.nn.Conv2d.forward(self,x)
            if before_pad>8:
                #print('>>>MYDEBUG conv pad out',before_pad,out.size(),flush=True)
                out = out[:before_pad]
                #print('>>>MYDEBUG conv out',before_pad,out.size(),flush=True)
            return out
        # get output shape
        output_shape = [
            (i + 2 * p - (di * (k - 1) + 1)) // d + 1
            for i, p, di, k, d in zip(
                x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
            )
        ]
        output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)

class ConvTranspose2d(torch.nn.ConvTranspose2d):
    def forward(self, x):
        if x.numel() > 0:
            before_pad = x.shape[0]
            if before_pad>8:
                #print('>>>MYDEBUG convtranspose in',before_pad,x.size(),flush=True)
                new_bs=32*((before_pad+32-1)//32)
                x = torch.nn.functional.pad(x, (0,0,0,0,0,0,0,new_bs-before_pad))
                #print('>>>MYDEBUG convtranspose pad in',before_pad,x.size(),flush=True)
            out = super(ConvTranspose2d, self).forward(x)
            if before_pad>8:
                #print('>>>MYDEBUG convtranspose pad out',before_pad,out.size(),flush=True)
                out = out[:before_pad]
                #print('>>>MYDEBUG convtranspose out',before_pad,out.size(),flush=True)
            return out
        # get output shape

        output_shape = [
            (i - 1) * d - 2 * p + (di * (k - 1) + 1) + op
            for i, p, di, k, d, op in zip(
                x.shape[-2:],
                self.padding,
                self.dilation,
                self.kernel_size,
                self.stride,
                self.output_padding,
            )
        ]
        output_shape = [x.shape[0], self.bias.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)


def interpolate(
    input, size=None, scale_factor=None, mode="nearest", align_corners=None
):
    if input.numel() > 0:
        return torch.nn.functional.interpolate(
            input.float(), size, scale_factor, mode, align_corners
        )

    def _check_size_scale_factor(dim):
        if size is None and scale_factor is None:
            raise ValueError("either size or scale_factor should be defined")
        if size is not None and scale_factor is not None:
            raise ValueError("only one of size or scale_factor should be defined")
        if (
            scale_factor is not None
            and isinstance(scale_factor, tuple)
            and len(scale_factor) != dim
        ):
            raise ValueError(
                "scale_factor shape must match input shape. "
                "Input is {}D, scale_factor size is {}".format(dim, len(scale_factor))
            )

    def _output_size(dim):
        _check_size_scale_factor(dim)
        if size is not None:
            return size
        scale_factors = _ntuple(dim)(scale_factor)
        # math.floor might return float in py2.7
        return [
            int(math.floor(input.size(i + 2) * scale_factors[i])) for i in range(dim)
        ]

    output_shape = tuple(_output_size(2))
    output_shape = input.shape[:-2] + output_shape
    return _NewEmptyTensorOp.apply(input, output_shape)
