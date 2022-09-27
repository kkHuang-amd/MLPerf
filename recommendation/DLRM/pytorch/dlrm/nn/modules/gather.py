"""Customized embedding gather"""
import copy

import torch
from torch.autograd import Function
from torch import nn

from apex import amp

from dlrm import cuda_ext

__all__ = ["EmbeddingGatherFunction", "JointSparseEmbedding", "embedding_gather"]

class EmbeddingGatherFunction(Function):
    """Customized embedding gather with fused plain SGD"""
    @staticmethod
    def forward(ctx, embedding, indices, to_fp16):
        output = cuda_ext.gather_gpu_fwd(embedding, indices, to_fp16)
        ctx.save_for_backward(indices)
        ctx.num_features = embedding.size(0)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        indices = ctx.saved_tensors[0]

        grad_embedding = cuda_ext.gather_gpu_bwd(grad_output, indices, ctx.num_features)

        return grad_embedding, None, None


class EmbeddingGatherSGDFunction(Function):
    @staticmethod
    def forward(ctx, embedding, indices, to_fp16, lr):
        output = cuda_ext.gather_gpu_fwd(embedding, indices, to_fp16)
        ctx.save_for_backward(indices)
        ctx.lr = lr
        # Does not need to go through save_for_backward because embedding is Parameter
        ctx.embedding = embedding
        return output

    @staticmethod
    def backward(ctx, grad_output):
        indices = ctx.saved_tensors[0]
        cuda_ext.dense_sparse_add(grad_output.squeeze(), ctx.embedding, indices, -ctx.lr)
        return None, None, None, None


class JointSparseEmbedding(nn.Module):
    """Joint multiple one hot embedding together

    Multiple one hot embedding can be done as one embedding (indexing).

    Args:
        categorical_feature_sizes (list): A list of integer indicating number of features of each embedding table
        embedding_dim (int): the size of each embedding vector
        device (torch.device): where to create the embedding. Default "cuda"
    """
    def __init__(self, categorical_feature_sizes, embedding_dim, device="cuda", compress_embedding=False,
                 to_fp16=False, fuse_sgd=False):
        super(JointSparseEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.categorical_feature_sizes = copy.copy(categorical_feature_sizes)
        self.compress_embedding = compress_embedding

        offsets = torch.tensor([0] + categorical_feature_sizes).cumsum(0).to(device)
        if compress_embedding:
            offsets = offsets.view(-1, 1)
        self.register_buffer("offsets", offsets)

        self.weights = torch.nn.Parameter(torch.rand((self.offsets[-1].item(), embedding_dim), device=device))
        self.to_fp16 = to_fp16
        self.fuse_sgd = fuse_sgd
        if fuse_sgd:
            # Use the same interface as optimizer
            self.param_groups = [{"lr": 1}]

    def forward(self, categorical_inputs):
        if not self.compress_embedding:
            # Check input has the right shape
            assert categorical_inputs.shape[1] == len(self.categorical_feature_sizes)

            if self.fuse_sgd:
                embedding_out = embedding_gather_sgd(self.weights, categorical_inputs + self.offsets[:-1],
                                                     self.to_fp16, self.param_groups[0]["lr"])
            else:
                embedding_out = embedding_gather(self.weights, categorical_inputs + self.offsets[:-1],
                                                 self.to_fp16)
        else:
            if self.fuse_sgd:
                embedding_out = embedding_gather_sgd(self.weights, categorical_inputs.to(torch.long),
                                                     self.to_fp16, self.param_groups[0]["lr"])
            else:
                embedding_out = embedding_gather(self.weights, categorical_inputs.to(torch.long),
                                                 self.to_fp16)
            embedding_out = embedding_out.squeeze()

        return embedding_out

    def extra_repr(self):
        s = F"categorical_feature_sizes={self.categorical_feature_sizes}\n"
        s += F"offsets={self.offsets.cpu().numpy()}"
        return s

embedding_gather = amp.float_function(EmbeddingGatherFunction.apply)
embedding_gather_sgd = EmbeddingGatherSGDFunction.apply
