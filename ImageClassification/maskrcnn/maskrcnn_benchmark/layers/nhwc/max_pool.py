# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

import torch
from torch.nn.modules.pooling import MaxPool2d
from apex import amp

try:
    from maskrcnn_benchmark import NHWC
except ImportError as error:
    print('No NHWC support!!')

class max_pool_NHWC_impl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, kernel_size, stride, padding, dilation):
        ctx.kernel_size = kernel_size
        ctx.stride = stride if stride is not None else 0
        ctx.padding = padding
        ctx.dilation = dilation

        y, indices = NHWC.max_pool_fwd_nhwc(x, kernel_size, stride, padding, dilation)

        # Need to save y as well
        ctx.save_for_backward(x, y, indices)
        return y

    @staticmethod
    def backward(ctx, y_grad):
        x, y, indices = ctx.saved_variables

        kernel = ctx.kernel_size
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation

        return NHWC.max_pool_bwd_nhwc(x,
                                   y,
                                   y_grad,
                                   indices,
                                   kernel,
                                   stride,
                                   padding,
                                   dilation), None, None, None, None

class MaxPool2d_NHWC(MaxPool2d):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(MaxPool2d_NHWC, self).__init__(kernel_size, stride=stride, padding=padding)
    def forward(self, x):
        x=x.permute(0,3,1,2).contiguous()
        return MaxPool2d.forward(self,x).permute(0,2,3,1).contiguous()
        #return max_pool_NHWC_impl.apply(x,
        #                                self.kernel_size,
        #                                self.stride,
        #                                self.padding,
        #                                self.dilation)

