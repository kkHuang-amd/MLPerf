"""Distributed version of DLRM model

In order to code the hybrid decomposition, the model code needs to be restructured. I don't know a clean enough
way to do it in the serialized model.py. So even a lot of codes will be duplicated between the 2 files, I still
believe it is easier and cleaner to just implement a distributed version from scratch instead of reuse the same
file.

The model is broken into 2 parts:
    - Bottom model: embeddings and bottom MLP
    - Top model: interaction and top MLP
The all to all communication will happen between bottom and top model

"""

import copy
import math

from absl import logging
from typing import MutableSequence, Any, Sequence, List

import torch
from torch import nn

import dlrm.nn
from dlrm.utils import distributed as dist
from dlrm import cuda_ext

try:
    from apex import mlp
except ImportError:
    logging.warning("APEX MLP is not availaible!")
    _USE_APEX_MLP = False
else:
    _USE_APEX_MLP = True


def distribute_to_buckets(
        elements: MutableSequence[Any],
        buckets: Sequence[List[Any]],
        start_bucket: int = 0):
    current_bucket = start_bucket % len(buckets)
    while elements:
        element = elements.pop()
        buckets[current_bucket].append(element)
        current_bucket = (current_bucket + 1) % len(buckets)
    return current_bucket


def compute_criteo_device_mapping(num_gpus, num_embeddings, heavy_components):
    """Compute device mappings for hybrid parallelism

    Bottom MLP running on device 0. 26 embeddings will be distributed across among all the devices. 0, 9, 19, 20, 21
    are the large ones, 20GB each.

    Args:
        num_gpus (int): Default 4.
        num_embeddings (int):
        heavy_components (tuple):

    Returns:
        device_mapping (dict):
    """
    bottom_mlp_index = -1
    heavy_components = list(heavy_components)
    regular_components = [x for x in range(num_embeddings) if x not in heavy_components]

    gpu_buckets = [[] for _ in range(num_gpus)]
    gpu_buckets[0].append(bottom_mlp_index)

    next_bucket = distribute_to_buckets(heavy_components, gpu_buckets, start_bucket=1)
    distribute_to_buckets(regular_components, gpu_buckets, start_bucket=next_bucket)

    vectors_per_gpu = [len(bucket) for bucket in gpu_buckets]

    gpu_buckets[0].pop(0)  # pop bottom mlp

    return {
        'bottom_mlp': 0,
        'embedding': gpu_buckets,
        'vectors_per_gpu': vectors_per_gpu,
    }


def get_criteo_device_mapping(
        num_gpus=4,
        num_embeddings=26,
        heavy_components=(0, 9, 19, 21, 20),
        preset_mapping=True):
    """Get device mappings for hybrid parallelism

    Bottom MLP running on device 0. 26 embeddings will be distributed across among all the devices. 0, 9, 19, 20, 21
    are the large ones, 20GB each.

    Args:
        num_gpus (int): Default 4.

    Returns:
        device_mapping (dict):
    """
    if preset_mapping:
        device_mapping = {'bottom_mlp': 0}  # bottom_mlp must be on the first GPU for now.
        if num_gpus == 4:
            device_mapping.update({
                'embedding' : [
                    [1, 5, 6, 8, 12, 17, 19, 23],
                    [0, 2, 3, 7, 11, 18, 22, 24],
                    [4, 10, 13, 14, 15, 16, 25, 21],
                    [9, 20]],
                'vectors_per_gpu' : [9, 8, 8, 2]})
        elif num_gpus == 8:
            device_mapping.update({
                'embedding' : [
                    [],
                    [0, 1, 2, 3],
                    [4, 5, 6, 7],
                    [8, 9, 10, 11],
                    [12, 13, 14, 15],
                    [16, 17, 18, 19],
                    [20, 22, 23, 24],
                    [21, 25]],
                'vectors_per_gpu' : [1, 4, 4, 4, 4, 4, 4, 2]})
        elif num_gpus == 16:
            device_mapping.update({
                'embedding' : [
                    [],
                    [0, 1],
                    [2, 3],
                    [4, 5],
                    [6, 7],
                    [8, 9],
                    [10, 11],
                    [12, 13],
                    [14, 15],
                    [16, 17],
                    [18, 19],
                    [20],
                    [21, 22],
                    [23],
                    [24],
                    [25]],
                'vectors_per_gpu' : [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 1, 1]})
        else:
            raise NotImplementedError

        # Additional information
        vectors_per_gpu = device_mapping['vectors_per_gpu']
        emb_vectors_per_gpu = torch.tensor(vectors_per_gpu[1:], dtype=torch.int32, device="cuda")
        device_mapping.update({'emb_vectors_per_gpu': emb_vectors_per_gpu, 'num_features': sum(vectors_per_gpu)})
        return device_mapping
    else:
        return compute_criteo_device_mapping(num_gpus, num_embeddings, heavy_components)


class DlrmBottom(nn.Module):
    """Bottom model of DLRM

    Embeddings and bottom MLP of DLRM. Only joint embedding is supported in this version.

    Args:
        num_numerical_features (int): Number of dense features fed into bottom MLP
        categorical_feature_sizes (list): A list of integer indicating number of features of each embedding table
        bottom_mlp_sizes (list): A list of integer indicating bottom MLP layer sizes. Last bottom MLP layer
            must be embedding_dim. Default None, not create bottom embedding on current device.
        embedding_dim (int): Length of embedding vectors. Default 128
        hash_indices (bool): If True, hashed_index = index % categorical_feature_size. Default False
        device (torch.device): where to create the embedding. Default "cuda"
        use_embedding_ext (bool): If True, use embedding extension.
        compress_embedding (bool): If True, use multi-table embedding.
    """
    def __init__(self, num_numerical_features, categorical_feature_sizes, fp16, bottom_mlp_sizes=None, embedding_dim=128,
                 hash_indices=False, device="cuda", use_embedding_ext=True, use_apex_mlp=True,
                 compress_embedding=False, use_fused_emb_sgd=False):
        super(DlrmBottom, self).__init__()
        if bottom_mlp_sizes is not None and embedding_dim != bottom_mlp_sizes[-1]:
            raise TypeError("The last bottom MLP layer must have same size as embedding.")

        self._embedding_dim = embedding_dim
        self._hash_indices = hash_indices
        self._categorical_feature_sizes = copy.copy(categorical_feature_sizes)
        self._model_dtype = torch.float16 if fp16 else torch.float
        self._compress_embedding = compress_embedding

        # Create bottom MLP
        if bottom_mlp_sizes is not None:
            if _USE_APEX_MLP and use_apex_mlp:
                self.bottom_mlp = mlp.MLP([num_numerical_features] + bottom_mlp_sizes).to(device)
            else:
                bottom_mlp_layers = []
                input_dims = num_numerical_features
                for output_dims in bottom_mlp_sizes:
                    bottom_mlp_layers.append(
                        nn.Linear(input_dims, output_dims))
                    bottom_mlp_layers.append(nn.ReLU(inplace=True))
                    input_dims = output_dims
                self.bottom_mlp = nn.Sequential(*bottom_mlp_layers).to(device)
                for module in self.bottom_mlp.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.normal_(
                            module.weight.data, 0., math.sqrt(2. / (module.in_features + module.out_features)))
                        nn.init.normal_(module.bias.data, 0., math.sqrt(1. /  module.out_features))
        else:
            # An empty list with Module property makes other code eaiser. For example, can call parameters()
            # and return empty iterator intead of having a contidion to skip it.
            self.bottom_mlp = torch.nn.ModuleList()

        self.embedding_offsets = None

        # Create joint embedding
        if categorical_feature_sizes:
            logging.warning("Combined all categorical features to single embedding table.")
            if not use_embedding_ext:
                self.joint_embedding = dlrm.nn.BuckleEmbedding(categorical_feature_sizes, embedding_dim, device)
                for cat, size in enumerate(categorical_feature_sizes):
                    module = self.joint_embedding
                    nn.init.uniform_(
                        module.embedding.weight.data[module.offsets[cat]:module.offsets[cat + 1]],
                        -math.sqrt(1. / size),
                        math.sqrt(1. / size))
            else:
                self.joint_embedding = dlrm.nn.JointSparseEmbedding(
                    categorical_feature_sizes, embedding_dim, device, self._compress_embedding,
                    to_fp16=fp16, fuse_sgd=use_fused_emb_sgd)
                for cat, size in enumerate(categorical_feature_sizes):
                    module = self.joint_embedding
                    nn.init.uniform_(
                        module.weights.data[module.offsets[cat]:module.offsets[cat + 1]],
                        -math.sqrt(1. / size),
                        math.sqrt(1. / size))
                if self._compress_embedding:
                    self.embedding_offsets = self.joint_embedding.offsets[:-1, :]
        else:
            self.joint_embedding = torch.nn.ModuleList()

    def forward(self, numerical_input, categorical_inputs):
        """

        Args:
            numerical_input (Tensor): with shape [batch_size, num_numerical_features]
            categorical_inputs (Tensor): with shape [num_categorical_features, batch_size]

        Returns:
            Tensor: Concatenated bottom mlp and embedding output in shape [batch, 1 + #embedding, embeddding_dim]
        """
        bottom_output = []

        if self._compress_embedding and self.bottom_mlp and self.num_categorical_features > 0:
            raise RuntimeError("Bottom MLP and embedding cannot run on the same GPU "
                               "when --compress_embedding is used.")

        # Reshape bottom mlp to concatenate with embeddings
        if self.bottom_mlp:
            batch_size = numerical_input.size()[0]
            bottom_output.append(self.bottom_mlp(numerical_input).view(batch_size, 1, -1))

        if self._hash_indices:
            for cat, size in enumerate(self._categorical_feature_sizes):
                categorical_inputs[:, cat] %= size
                logging.log_first_n(
                    logging.WARNING, F"Hashed indices out of range.", 1)

        # NOTE: It doesn't transpose input
        if self.num_categorical_features > 0:
            bottom_output.append(self.joint_embedding(categorical_inputs))

        if self._compress_embedding:
            cat_bottom_out = bottom_output
        elif len(bottom_output) == 1:
            cat_bottom_out = bottom_output[0]
        else:
            cat_bottom_out = torch.cat(bottom_output, dim=1)
        return cat_bottom_out

    # pylint:disable=missing-docstring
    @property
    def num_categorical_features(self):
        return len(self._categorical_feature_sizes)

    def extra_repr(self):
        s = F"hash_indices={self._hash_indices}"
        return s
    # pylint:enable=missing-docstring


class DlrmTop(nn.Module):
    """Top model of DLRM

    Interaction and top MLP of DLRM.

    Args:
        top_mlp_sizes (list): A list of integers indicating top MLP layer sizes.
        num_interaction_inputs (int): Number of input vectors to interaction, equals to #embeddings + 1 (
            bottom mlp)
        embedding_dim (int): Length of embedding vectors. Default 128
        interaction_op (string): Type of interactions. Default "dot"
    """
    def __init__(self, top_mlp_sizes, num_interaction_inputs, embedding_dim=128, interaction_op="dot", use_apex_mlp=True,
                 compress_embedding=False, use_wmma_interaction=False):
        super(DlrmTop, self).__init__()
        self._interaction_op = interaction_op

        if interaction_op in ["dot", "pytorch_dot"]:
            num_interactions = (num_interaction_inputs * (num_interaction_inputs - 1)) // 2 + embedding_dim
        elif interaction_op == "cat":
            num_interactions = num_interaction_inputs * embedding_dim
        else:
            raise TypeError(F"Unknown interaction {interaction_op}.")

        if interaction_op == "dot":
            if use_wmma_interaction:
                from dlrm.nn.functional import DotBasedInteract
                self.dot_base_interaction = DotBasedInteract.apply
            else:
                from dlrm.nn.functional import DotBaseInteractNoWmma
                self.dot_base_interaction = DotBaseInteractNoWmma.apply

        # Create Top MLP
        top_mlp_layers = []
        input_dims = num_interactions + 1  # pad 1 to be multiple of 8
        if _USE_APEX_MLP and use_apex_mlp:
            top_mlp_layers.append(mlp.MLP([input_dims] + top_mlp_sizes[:-1]))
            top_mlp_layers.append(nn.Linear(top_mlp_sizes[-2], top_mlp_sizes[-1]))
        else:
            for output_dims in top_mlp_sizes[:-1]:
                top_mlp_layers.append(nn.Linear(input_dims, output_dims))
                top_mlp_layers.append(nn.ReLU(inplace=True))
                input_dims = output_dims
            top_mlp_layers.append(nn.Linear(input_dims, top_mlp_sizes[-1]))

        self.top_mlp = nn.Sequential(*top_mlp_layers)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight.data, 0., math.sqrt(2. / (module.in_features + module.out_features)))
                nn.init.normal_(module.bias.data, 0., math.sqrt(1. /  module.out_features))

        # Set corresponding weight of padding to 0
        if not (_USE_APEX_MLP and use_apex_mlp):
            nn.init.zeros_(self.top_mlp[0].weight[:, -1].data)
        else:
            nn.init.zeros_(self.top_mlp[0].weights[0][:, -1].data)

        if compress_embedding:
            self.forward = self._forward_compress_embedding
        else:
            self.forward = self._forward

    # pylint:disable=missing-docstring
    def extra_repr(self):
        s = F"interaction_op={self._interaction_op}"
        return s
    # pylint:enable=missing-docstring

    def _forward_compress_embedding(self, bottom_output):
        return self.top_mlp(bottom_output)

    def _forward(self, bottom_output):
        """

        Args:
            numerical_input (Tensor): with shape [batch_size, num_numerical_features]
            categorical_inputs (Tensor): with shape [num_categorical_features, batch_size]
        """
        # The first vector in bottom_output is from bottom mlp
        bottom_mlp_output = bottom_output.narrow(1, 0, 1).squeeze()
        if self._interaction_op == "dot":
            interaction_output = self.dot_base_interaction(bottom_output, bottom_mlp_output)
        elif self._interaction_op == "pytorch_dot":
            interaction = torch.bmm(bottom_output, torch.transpose(bottom_output, 1, 2))
            tril_indices_row, tril_indices_col = torch.tril_indices(
                interaction.shape[1], interaction.shape[2], offset=-1)
            interaction_flat = interaction[:, tril_indices_row, tril_indices_col]

            # concatenate dense features and interactions
            zero_padding = torch.zeros(
                bottom_output.shape[0], 1, dtype=bottom_output.dtype, device=bottom_output.device)
            interaction_output = torch.cat((bottom_mlp_output, interaction_flat, zero_padding), dim=1)
        elif self._interaction_op == "cat":
            interaction_output = bottom_output
        else:
            raise NotImplementedError

        top_mlp_output = self.top_mlp(interaction_output)

        return top_mlp_output


class BottomToTop(torch.autograd.Function):
    """Switch from model parallel to data parallel

    Wrap the communication of doing from bottom model in model parallel fashion to top model in data parallel

    TODO (haow): Current implementation assumes all the gpu gets same number of vectors from bottom model. May need
        to change it to a more generalized solution.
    """

    @staticmethod
    def forward(ctx, local_bottom_outputs, batch_size_per_gpu, vector_dim, vectors_per_gpu,
                use_alltoall_base):
        """
        Args:
            ctx : Pytorch convention
            local_bottom_outputs (Tensor): Concatenated output of bottom model
            batch_size_per_gpu (int):
            vector_dim (int):
            vectors_per_gpu (int): Note, bottom MLP is considered as 1 vector

        Returns:
            slice_embedding_outputs (Tensor): Patial output from bottom model to feed into data parallel top model
        """
        ctx.world_size = torch.distributed.get_world_size()
        ctx.batch_size_per_gpu = batch_size_per_gpu
        ctx.vector_dim = vector_dim
        ctx.vectors_per_gpu = vectors_per_gpu
        ctx.use_alltoall_base = use_alltoall_base

        if not ctx.use_alltoall_base:
            # Buffer shouldn't need to be zero out. If not zero out buffer affecting accuracy, there must be a bug.
            bottom_output_buffer = [torch.empty(
                batch_size_per_gpu, n * vector_dim,
                device=local_bottom_outputs.device, dtype=local_bottom_outputs.dtype) for n in vectors_per_gpu]

            torch.distributed.all_to_all(bottom_output_buffer, list(local_bottom_outputs.split(batch_size_per_gpu, dim=0)))
            return torch.cat(bottom_output_buffer, dim=1).view(batch_size_per_gpu, -1, vector_dim)
        else:
            bottom_output_buffer = torch.empty(
                    sum(vectors_per_gpu) * batch_size_per_gpu * vector_dim,
                    device=local_bottom_outputs.device,
                    dtype=local_bottom_outputs.dtype)
            output_split_sizes = [n * batch_size_per_gpu * vector_dim \
                                  for n in vectors_per_gpu]

            torch.distributed.all_to_all_single(
                    bottom_output_buffer,
                    local_bottom_outputs,
                    output_split_sizes=output_split_sizes)
            split_output = [v.view(batch_size_per_gpu, -1) \
                            for v in bottom_output_buffer.split(output_split_sizes, dim=0)]
            return torch.cat(split_output, dim=1).view(batch_size_per_gpu, -1, vector_dim)

    @staticmethod
    def backward(ctx, grad_slice_bottom_outputs):
        rank = dist.get_rank()

        if not ctx.use_alltoall_base:
            grad_local_bottom_outputs = torch.empty(
                ctx.batch_size_per_gpu * ctx.world_size, ctx.vectors_per_gpu[rank] * ctx.vector_dim,
                device=grad_slice_bottom_outputs.device,
                dtype=grad_slice_bottom_outputs.dtype)
            # All to all only takes list while split() returns tuple
            grad_local_bottom_outputs_split = list(grad_local_bottom_outputs.split(ctx.batch_size_per_gpu, dim=0))

            split_grads = [t.contiguous() for t in (grad_slice_bottom_outputs.view(ctx.batch_size_per_gpu, -1).split(
                [ctx.vector_dim * n for n in ctx.vectors_per_gpu], dim=1))]

            torch.distributed.all_to_all(grad_local_bottom_outputs_split, split_grads)
            return grad_local_bottom_outputs.view(grad_local_bottom_outputs.shape[0], -1, ctx.vector_dim), \
                    None, None, None, None
        else:
            grad_local_bottom_outputs = torch.empty(
                ctx.batch_size_per_gpu * ctx.world_size * ctx.vectors_per_gpu[rank] * ctx.vector_dim,
                device=grad_slice_bottom_outputs.device,
                dtype=grad_slice_bottom_outputs.dtype)
            input_split_sizes = [ctx.vector_dim * n for n in ctx.vectors_per_gpu]
            split_grads = grad_slice_bottom_outputs.view(
                    ctx.batch_size_per_gpu, -1).split(input_split_sizes, dim=1)
            split_grads = torch.cat([t.contiguous().view(-1, ctx.batch_size_per_gpu) \
                    for t in split_grads]).contiguous()

            torch.distributed.all_to_all_single(
                    grad_local_bottom_outputs,
                    split_grads,
                    input_split_sizes=input_split_sizes)
            return grad_local_bottom_outputs.view(-1, ctx.vectors_per_gpu[rank], ctx.vector_dim), \
                    None, None, None, None


def unique(input, batch_size_per_gpu, num_gpus, num_embeddings):
    output = cuda_ext.unique_transpose(
            input.view(-1, batch_size_per_gpu), num_embeddings, num_gpus)
    return output


def send_compression_metadata(categorical_features, categorical_feature_offsets,
        batch_size_per_gpu, vector_dim, num_vectors, vectors_per_gpu, bottom_mlp_rank,
        device):
    global dev_num_unique_indices
    rank = dist.get_rank()
    num_gpus = dist.get_world_size()

    # Find unique indices, inverse indices, and number of unique indices
    unique_indices = []
    inverse_indices = []
    if categorical_features is not None:
        batch_size = categorical_features[0].shape[0]
        cat_tensor = torch.cat(categorical_features, dim=0).view(-1, batch_size)
        cat_tensor += categorical_feature_offsets
        unique_indices, inverse_indices, dev_num_unique_indices = unique(
                cat_tensor, batch_size_per_gpu, num_gpus, vectors_per_gpu[rank])
        dev_num_unique_indices = dev_num_unique_indices.to(torch.int)
    else:
        inverse_indices = torch.empty(batch_size_per_gpu * num_gpus, device=device, dtype=torch.int)
        dev_num_unique_indices = torch.tensor([batch_size_per_gpu] * num_gpus, device=device, dtype=torch.int)

    # Send number of unique indices
    num_unique_indices = torch.empty(num_gpus, dtype=torch.int, device=device)
    num_unique_indices_handles = torch.distributed.all_to_all_single(
            num_unique_indices, dev_num_unique_indices, async_op=True)

    # Send inverse indices
    inv_indices = torch.empty(batch_size_per_gpu * num_vectors, dtype=torch.int, device=device)
    inv_indices_handle = torch.distributed.all_to_all_single(
            inv_indices, inverse_indices,
            output_split_sizes=[n * batch_size_per_gpu \
                                for n in vectors_per_gpu], async_op=True)

    # Prepare the unique indices count for CompressedBottomToTop
    if categorical_features is not None:
        dev_num_unique_indices = dev_num_unique_indices.clone().cpu().tolist()
    else:
        dev_num_unique_indices = [batch_size_per_gpu] * num_gpus

    return unique_indices, num_unique_indices, num_unique_indices_handles, \
            inv_indices, inv_indices_handle


class CompressedBottomToTop(torch.autograd.Function):
    @staticmethod
    def forward(ctx, batch_size_per_gpu, vector_dim, device_mapping,
                use_alltoall_base, num_unique_indices, num_unique_indices_handles,
                inverse_indices, inverse_indices_handle, *local_bottom_outputs):
        global dev_num_unique_indices
        ctx.bottom_mlp_rank = device_mapping['bottom_mlp']
        ctx.batch_size_per_gpu = batch_size_per_gpu
        ctx.vector_dim = vector_dim
        ctx.num_features = device_mapping['num_features']
        emb_vectors_per_gpu = device_mapping['emb_vectors_per_gpu']
        rank = dist.get_rank()
        num_gpus = dist.get_world_size()

        local_bottom = local_bottom_outputs[0]
        device = local_bottom.device
        model_dtype = local_bottom.dtype

        # Wait for metadata
        num_unique_indices_handles.wait()

        # Compute output tensor's size and allocate output tensor

        indices_offset = num_unique_indices.clone().cpu()
        ctx.num_unique_indices = indices_offset.clone().tolist()

        total_num_uniques = sum(ctx.num_unique_indices)
        bottom = torch.empty(total_num_uniques, vector_dim, device=device, dtype=model_dtype)

        # Scatter embedding output
        torch.distributed.all_to_all_single(
                bottom,
                local_bottom,
                input_split_sizes=dev_num_unique_indices,
                output_split_sizes=ctx.num_unique_indices)

        # Wait for inverse indices
        inverse_indices_handle.wait()
        inverse_indices = inverse_indices[batch_size_per_gpu:]

        bottom_mlp_output, emb_output = bottom.split([batch_size_per_gpu, bottom.shape[0] - batch_size_per_gpu])

        # Only require index offsets of embedding
        indices_offset[0] = 0
        indices_offset = indices_offset.cumsum(0, dtype=torch.int32).cuda()

        # Reshape the output
        bottom = cuda_ext.fused_index_select_dot(
                emb_output,
                bottom_mlp_output,
                inverse_indices,
                emb_vectors_per_gpu,
                indices_offset,
                batch_size_per_gpu,
                ctx.num_features,
                vector_dim,
                1)

        # Store the inverese indices for backward
        ctx.save_for_backward(
                emb_output,
                bottom_mlp_output,
                inverse_indices,
                emb_vectors_per_gpu,
                indices_offset)
        return bottom

    @staticmethod
    def backward(ctx, grads):
        global dev_num_unique_indices
        rank = dist.get_rank()
        device = grads.device
        model_dtype = grads.dtype
        num_gpus = dist.get_world_size()
        emb_output, bottom_mlp_output, inverse_indices, emb_vectors_per_gpu, indices_offset = ctx.saved_tensors

        local_bottom = cuda_ext.fused_dot_dedup(
                emb_output,
                bottom_mlp_output,
                inverse_indices,
                emb_vectors_per_gpu,
                indices_offset,
                grads,
                ctx.batch_size_per_gpu,
                ctx.num_features,
                ctx.vector_dim,
                1)

        # Compute output tensor's size and allocate output tensor
        total_unique_indices = sum(dev_num_unique_indices)
        bottom = torch.empty(total_unique_indices, ctx.vector_dim, dtype=model_dtype, device=device)

        # Gather embedding gradient
        torch.distributed.all_to_all_single(
                bottom,
                local_bottom,
                output_split_sizes=dev_num_unique_indices,
                input_split_sizes=ctx.num_unique_indices)

        ret = [None] * 8
        if ctx.bottom_mlp_rank != rank:
            ret.append(bottom)
        else:
            ret.append(bottom.unsqueeze(1))
        return tuple(ret)


# Default bottom-top interface
bottom_to_top = BottomToTop.apply


def set_bottom_to_top(compress_embedding):
    global bottom_to_top
    bottom_to_top = CompressedBottomToTop.apply \
            if compress_embedding else BottomToTop.apply


class DistDlrm():
    """Wrapper of top and bottom model

    To make interface simpler, this wrapper class is created to have bottom and top model in the same class.
    """

    def __init__(self, num_numerical_features, categorical_feature_sizes, bottom_mlp_sizes, top_mlp_sizes,
                 embedding_dim, world_num_categorical_features, fp16, interaction_op="dot",
                 hash_indices=False, device="cuda", use_embedding_ext=True, use_apex_mlp=True,
                 compress_embedding=False, use_fused_emb_sgd=False, use_wmma_interaction=False):
        super(DistDlrm, self).__init__()
        self.embedding_dim = embedding_dim
        self.bottom_model = DlrmBottom(
            num_numerical_features, categorical_feature_sizes, fp16, bottom_mlp_sizes, embedding_dim,
            hash_indices=hash_indices, device=device, use_embedding_ext=use_embedding_ext,
            use_apex_mlp=use_apex_mlp, compress_embedding=compress_embedding,
            use_fused_emb_sgd=use_fused_emb_sgd)

        num_interaction_inputs = world_num_categorical_features + 1
        self.top_model = DlrmTop(top_mlp_sizes, num_interaction_inputs,
                                 embedding_dim=embedding_dim, interaction_op=interaction_op,
                                 use_apex_mlp=use_apex_mlp, compress_embedding=compress_embedding,
                                 use_wmma_interaction=use_wmma_interaction).to(device)

        # Set bottom-top interface
        set_bottom_to_top(compress_embedding)

    def __call__(self, numerical_input, categorical_inputs):
        """Single GPU forward"""
        assert dist.get_world_size() == 1  # DONOT run this in distributed mode
        bottom_out = self.bottom_model(numerical_input, categorical_inputs)
        top_out = self.top_model(bottom_out)

        return top_out

    def __repr__(self):
        s = F"{self.__class__.__name__}{{\n"
        s += repr(self.bottom_model)
        s += "\n"
        s += repr(self.top_model)
        s += "\n}\n"
        return s

    def to(self, *args, **kwargs):
        self.bottom_model.to(*args, **kwargs)
        self.top_model.to(*args, **kwargs)
