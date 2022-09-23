# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
helper class that supports empty tensors on some nhwc functions
"""

import math
import torch
from torch.nn.modules.utils import _ntuple
from maskrcnn_benchmark.layers.nhwc import conv
#from maskrcnn_benchmark.layers.nhwc import transforms
from maskrcnn_benchmark.layers.nhwc import UpSampleNearest2d


class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None

#@torch.jit.script
def _conv_npad(x):
    before_pad = x.shape[0]
    new_bs=32*((before_pad+32-1)//32)
    return torch.nn.functional.pad(x, (0,0,0,0,0,0,0,new_bs-before_pad))

class Conv2d_NHWC(conv.Conv2d_NHWC):
    #@torch.jit.ignore
    def forward(self, x):
        if x.numel() > 0:
            before_pad = x.shape[0]
            if before_pad>8:
                #print('>>>MYDEBUG conv in',before_pad,x.size(),flush=True)
                new_bs=32*((before_pad+32-1)//32)
                x = torch.nn.functional.pad(x, (0,0,0,0,0,0,0,new_bs-before_pad))
                #x=_conv_npad(x)
                #print('>>>MYDEBUG conv pad in',before_pad,x.size(),flush=True)
            #start = torch.cuda.Event(enable_timing=True)
            #end = torch.cuda.Event(enable_timing=True)
            #start.record()
            out=conv.Conv2d_NHWC.forward(self, x)
            #end.record()
            #torch.cuda.synchronize()
            #print('>>>MYDEBUG conv time nhwc',x.shape,self.padding, self.dilation, self.kernel_size, self.stride,start.elapsed_time(end),flush=True)

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

class ConvTranspose2d_NHWC(conv.ConvTranspose2d_NHWC):
    def forward(self, x):
        if x.numel() > 0:
            before_pad = x.shape[0]
            if before_pad>8:
                #print('>>>MYDEBUG convtranspose in',before_pad,x.size(),flush=True)
                new_bs=32*((before_pad+32-1)//32)
                x = torch.nn.functional.pad(x, (0,0,0,0,0,0,0,new_bs-before_pad))
                #print('>>>MYDEBUG convtranspose pad in',before_pad,x.size(),flush=True)
            out=super(ConvTranspose2d_NHWC, self).forward(x)
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
'''
class ConvTranspose2d_NHWC(torch.nn.ConvTranspose2d):
    def forward(self, x):
        if x.numel() > 0:
            before_pad = x.shape[0]
            if before_pad>8:
                #print('>>>MYDEBUG convtranspose in',before_pad,x.size(),flush=True)
                new_bs=32*((before_pad+32-1)//32)
                x = torch.nn.functional.pad(x, (0,0,0,0,0,0,0,new_bs-before_pad))
                #print('>>>MYDEBUG convtranspose pad in',before_pad,x.size(),flush=True)
            x=x.permute(0,3,1,2).contiguous()
            out = super(ConvTranspose2d_NHWC, self).forward(x).permute(0,2,3,1).contiguous()
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
'''
@torch.jit.script
def nhwc_to_nchw_transform(x):
    #print('>>>MYDEBUG in NHtoNC',x.shape,x.stride())
    if x.numel() == 0:
        return x
    #op = transforms.NHWCToNCHW()
    #y=op(x)
    y=x.permute(0,3,1,2).contiguous()
    #print('>>>MYDEBUG out NHtoNC',y.shape,y.stride())
    return y

@torch.jit.script
def nchw_to_nhwc_transform(x):
    #print('>>>MYDEBUG in NCtoNH',x.shape,x.stride())
    if x.numel() == 0:
        return x
    #op = transforms.NCHWToNHWC()
    #y=op(x)
    y=x.permute(0,2,3,1).contiguous()
    #print('>>>MYDEBUG out NCtoNH',y.shape,y.stride())
    return y

def interpolate_nhwc(input, size=None, scale_factor=None, mode='nearest', align_corners=None):

    def _check_size_scale_factor(dim):
        if size is None and scale_factor is None:
            raise ValueError('either size or scale_factor should be defined')
        if size is not None and scale_factor is not None:
            raise ValueError('only one of size or scale_factor should be defined')
        if scale_factor is not None and isinstance(scale_factor, tuple)\
                and len(scale_factor) != dim:
            raise ValueError('scale_factor shape must match input shape. '
                             'Input is {}D, scale_factor size is {}'.format(dim, len(scale_factor)))
    def _output_size(dim):
        _check_size_scale_factor(dim)
        if size is not None:
            return size
        scale_factors = _ntuple(dim)(scale_factor)
        # math.floor might return float in py2.7

        # make scale_factor a tensor in tracing so constant doesn't get baked in
        if torch._C._get_tracing_state():
            return [(torch.floor((input.size(i + 1).float() * torch.tensor(scale_factors[i],
                     dtype=torch.float32)).float())) for i in range(dim)]
        else:
            return [int(math.floor(float(input.size(i + 1)) * scale_factors[i])) for i in range(dim)]

    if mode == 'nearest' and input.dim() == 4 and align_corners is None:
        return UpSampleNearest2d.upsample_nearest2d(input, _output_size(2))
    else:
        x = nhwc_to_nchw_transform(input)
        x = F.interpolate(x, size, scale_factor, mode, align_corners)
        result = nchw_to_nhwc_transform(x)
        return result
