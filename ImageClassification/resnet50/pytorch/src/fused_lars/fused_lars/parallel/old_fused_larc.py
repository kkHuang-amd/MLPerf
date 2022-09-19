import torch
from torch import nn
from torch.nn.parameter import Parameter
from apex.multi_tensor_apply import multi_tensor_applier

class FusedLARC(object):

    def __init__(self, optimizer, trust_coefficient=0.02, clip=True, eps=1e-8):
        self.optim = optimizer
        self.trust_coefficient = trust_coefficient
        self.eps = eps
        self.clip = clip

        if multi_tensor_applier.available:
            import amp_C
            self.multi_tensor_l2norm = amp_C.multi_tensor_l2norm
            self.multi_tensor_larc = amp_C.multi_tensor_larc
            self._dummy_overflow_buf = torch.cuda.IntTensor(1).zero_()

    def __getstate__(self):
        return self.optim.__getstate__()

    def __setstate__(self, state):
        self.optim.__setstate__(state)

    @property
    def state(self):
        return self.optim.state

    def __repr__(self):
        return self.optim.__repr__()

    @property
    def param_groups(self):
        return self.optim.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.optim.param_groups = value

    def state_dict(self):
        return self.optim.state_dict()

    def load_state_dict(self, state_dict):
        self.optim.load_state_dict(state_dict)

    def zero_grad(self):
        self.optim.zero_grad()

    def add_param_group(self, param_group):
        self.optim.add_param_group( param_group)

    def step(self):
        with torch.no_grad():
            weight_decays = []
            lrs = []
            for group in self.optim.param_groups:
                # absorb weight decay control from optimizer
                weight_decay = group['weight_decay'] if 'weight_decay' in group else 0
                weight_decays.append(weight_decay)
                group['weight_decay'] = 0

                lr = group['lr']
                lrs.append(lr)
                group['lr'] = lr

                fused_larc_params = []
                fused_larc_grads = []
                for p in group['params']:
                    if p.grad is None:
                        continue
                    fused_larc_grads.append(p.grad.data)
                    fused_larc_params.append(p.data)

                n = len(fused_larc_grads)

                # Compute L2 norms
                norms = multi_tensor_applier(
                        self.multi_tensor_l2norm,
                        self._dummy_overflow_buf,
                        [fused_larc_grads + fused_larc_params],
                        True)[1]

                # Compute new grads
                multi_tensor_applier(
                        self.multi_tensor_larc,
                        self._dummy_overflow_buf,
                        [fused_larc_grads, fused_larc_params],
                        norms[:n],
                        norms[n:],
                        lr,
                        self.trust_coefficient,
                        self.eps,
                        weight_decay,
                        group['is_skipped'])

        self.optim.step()
        # return weight decay control to optimizer
        for i, group in enumerate(self.optim.param_groups):
            group['weight_decay'] = weight_decays[i]
            group['lr'] = lrs[i]
