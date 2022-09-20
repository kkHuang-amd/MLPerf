# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
The policy and value networks share a majority of their architecture.
This helps the intermediate layers extract concepts that are relevant to both
move prediction and score estimation.
"""

from absl import flags
import logging
import os.path
import struct
import tempfile
import time
import numpy as np
import random
import tf2onnx

import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense
import tensorflow.keras as keras

import features as features_lib
import go
import symmetries
import minigo_model
import horovod.tensorflow as hvd


flags.DEFINE_integer('train_batch_size', 256,
                     'Batch size to use for train/eval evaluation. For GPU '
                     'this is batch size as expected. If \"use_tpu\" is set,'
                     'final batch size will be = train_batch_size * num_tpu_cores')

flags.DEFINE_integer('conv_width', 256 if go.N == 19 else 32,
                     'The width of each conv layer in the shared trunk.')

flags.DEFINE_integer('policy_conv_width', 2,
                     'The width of the policy conv layer.')

flags.DEFINE_integer('value_conv_width', 1,
                     'The width of the value conv layer.')

flags.DEFINE_integer('fc_width', 256 if go.N == 19 else 64,
                     'The width of the fully connected layer in value head.')

flags.DEFINE_integer('trunk_layers', go.N,
                     'The number of resnet layers in the shared trunk.')

flags.DEFINE_multi_integer('lr_boundaries', [400000, 600000],
                           'The number of steps at which the learning rate will decay')

flags.DEFINE_multi_float('lr_rates', [0.01, 0.001, 0.0001],
                         'The different learning rates')

flags.DEFINE_integer('training_seed', 0,
                     'Random seed to use for training and validation')

flags.register_multi_flags_validator(
    ['lr_boundaries', 'lr_rates'],
    lambda flags: len(flags['lr_boundaries']) == len(flags['lr_rates']) - 1,
    'Number of learning rates must be exactly one greater than the number of boundaries')

flags.DEFINE_float('l2_strength', 1e-4,
                   'The L2 regularization parameter applied to weights.')

flags.DEFINE_float('value_cost_weight', 1.0,
                   'Scalar for value_cost, AGZ paper suggests 1/100 for '
                   'supervised learning')

flags.DEFINE_float('sgd_momentum', 0.9,
                   'Momentum parameter for learning rate.')

flags.DEFINE_string('work_dir', None,
                    'The Estimator working directory. Used to dump: '
                    'checkpoints, tensorboard logs, etc..')

flags.DEFINE_bool('use_tpu', False, 'Whether to use TPU for training.')

flags.DEFINE_string(
    'tpu_name', None,
    'The Cloud TPU to use for training. This should be either the name used'
    'when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.')

flags.DEFINE_integer(
    'num_tpu_cores', default=8,
    help=('Number of TPU cores. For a single TPU device, this is 8 because each'
          ' TPU has 4 chips each with 2 cores.'))

flags.DEFINE_string('gpu_device_list', None,
                    'Comma-separated list of GPU device IDs to use.')

flags.DEFINE_bool('quantize', False,
                  'Whether create a quantized model. When loading a model for '
                  'inference, this must match how the model was trained.')

flags.DEFINE_integer('quant_delay', 700 * 1024,
                     'Number of training steps after which weights and '
                     'activations are quantized.')

flags.DEFINE_integer(
    'iterations_per_loop', 128,
    help=('Number of steps to run on TPU before outfeeding metrics to the CPU.'
          ' If the number of iterations in the loop would exceed the number of'
          ' train steps, the loop will exit before reaching'
          ' --iterations_per_loop. The larger this value is, the higher the'
          ' utilization on the TPU.'))

flags.DEFINE_integer(
    'summary_steps', default=256,
    help='Number of steps between logging summary scalars.')

flags.DEFINE_integer(
    'keep_checkpoint_max', default=5, help='Number of checkpoints to keep.')

flags.DEFINE_bool(
    'use_random_symmetry', True,
    help='If true random symmetries be used when doing inference.')

flags.DEFINE_bool(
    'use_SE', False,
    help='Use Squeeze and Excitation.')

flags.DEFINE_bool(
    'use_SE_bias', False,
    help='Use Squeeze and Excitation with bias.')

flags.DEFINE_integer(
    'SE_ratio', 2,
    help='Squeeze and Excitation ratio.')

flags.DEFINE_bool(
    'use_swish', False,
    help=('Use Swish activation function inplace of ReLu. '
          'https://arxiv.org/pdf/1710.05941.pdf'))

flags.DEFINE_bool(
    'bool_features', False,
    help='Use bool input features instead of float')

flags.DEFINE_string(
    'input_features', 'agz',
    help='Type of input features: "agz" or "mlperf07"')

flags.DEFINE_string(
    'input_layout', 'nhwc',
    help='Layout of input features: "nhwc" or "nchw"')


# TODO(seth): Verify if this is still required.
flags.register_multi_flags_validator(
    ['use_tpu', 'iterations_per_loop', 'summary_steps'],
    lambda flags: (not flags['use_tpu'] or
                   flags['summary_steps'] % flags['iterations_per_loop'] == 0),
    'If use_tpu, summary_steps must be a multiple of iterations_per_loop')

FLAGS = flags.FLAGS


def get_features_planes():
    if FLAGS.input_features == 'agz':
        return features_lib.AGZ_FEATURES_PLANES
    elif FLAGS.input_features == 'mlperf07':
        return features_lib.MLPERF07_FEATURES_PLANES
    else:
        raise ValueError('unrecognized input features "%s"' %
                         FLAGS.input_features)


def get_features():
    if FLAGS.input_features == 'agz':
        return features_lib.AGZ_FEATURES
    elif FLAGS.input_features == 'mlperf07':
        return features_lib.MLPERF07_FEATURES
    else:
        raise ValueError('unrecognized input features "%s"' %
                         FLAGS.input_features)

def mg_activation(inputs):
    if FLAGS.use_swish:
        return tf.nn.swish(inputs)

    return tf.nn.relu(inputs)


#TODO (aayujain): remove params and explicitly pass arguments (filters)
def residual_block(inputs, params, data_format, bn_axis, name):
    x = Conv2D(params['conv_width'], 3, padding='same', use_bias=False,
               data_format=data_format)(inputs)
    x = BatchNormalization(axis=bn_axis, momentum=.95, epsilon=1e-5,
                           center=True, scale=True)(x)
    x = mg_activation(x)
    x = Conv2D(params['conv_width'], 3, padding='same', use_bias=False,
               data_format=data_format)(x)
    x = BatchNormalization(axis=bn_axis, momentum=.95, epsilon=1e-5,
                           center=True, scale=True)(x)
    output = mg_activation(inputs + x)

    return output

def policy_head(inputs, params, data_format, bn_axis, name):
    x =  Conv2D(params['policy_conv_width'], 1, padding='same', use_bias=False,
                data_format=data_format)(inputs)
    x = BatchNormalization(axis=bn_axis, momentum=.95, epsilon=1e-5,
                           center=False, scale=False)(x)
    x = mg_activation(x)
    #TODO (aayujain): Name the layer?
    logits = Dense(go.N * go.N + 1)(
        tf.reshape(x, [-1, params['policy_conv_width'] * go.N * go.N]))

    policy_output = tf.nn.softmax(logits, name='policy_output')

    return policy_output, logits

def value_head(inputs, params, data_format, bn_axis, name):
    x =  Conv2D(params['value_conv_width'], 1, padding='same', use_bias=False,
                data_format=data_format)(inputs)
    x = BatchNormalization(axis=bn_axis, momentum=.95, epsilon=1e-5,
                           center=False, scale=False)(x)
    x = mg_activation(x)

    #TODO (aayujain): Name the layer?
    value_fc_hidden = mg_activation(Dense(params['fc_width'])(
        tf.reshape(x, [-1, params['value_conv_width'] * go.N * go.N])))
    value_output = tf.nn.tanh(
        tf.reshape(tf.keras.layers.Dense(1)(value_fc_hidden), [-1]),
        name='value_output')

    return value_output


class DualNetwork():
    def __init__(self, save_file=None, training=True):
        self.save_file = save_file
        if FLAGS.input_layout == 'nhwc':
            self.bn_axis = -1
            self.data_format = 'channels_last'
            self.input_shape = (go.N, go.N, get_features_planes())
        else:
            self.bn_axis = 1
            self.data_format = 'channels_first'
            self.input_shape = (get_features_planes(), go.N, go.N)

        self.params = FLAGS.flag_values_dict()
        self.initialize_model(training)

        self.lr_scheduler = keras.optimizers.schedules.PiecewiseConstantDecay(
            self.params['lr_boundaries'], self.params['lr_rates'])

        self.optimizer = keras.optimizers.SGD(
            learning_rate=self.lr_scheduler,
            momentum=self.params['sgd_momentum'])

        self.checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        self.manager = tf.train.CheckpointManager(self.checkpoint, FLAGS.export_path, max_to_keep=100)
        if self.save_file:
            self.initialize_weights(self.save_file)

    def initialize_model(self, training):
        self.features = keras.Input(shape=self.input_shape, dtype=tf.bool)
        if self.params['bool_features']:
            casted_features = tf.cast(self.features, dtype=tf.float32)
        else:
            casted_features = self.features

        # Initial Block
        initial_block = Conv2D(self.params['conv_width'], 3, padding='same',
                               use_bias=False, data_format=self.data_format,
                               )(casted_features)
        initial_block = BatchNormalization(axis=self.bn_axis, momentum=.95,
                                           epsilon=1e-5, center=True,
                                           scale=True)(initial_block)
        initial_block = mg_activation(initial_block)

        shared_output = initial_block
        for i in range(self.params['trunk_layers']):
            shared_output = residual_block(shared_output, self.params,
                                           self.data_format, self.bn_axis,
                                           f"res_{i}")

        policy_output, logits = policy_head(shared_output, self.params,
                                            self.data_format, self.bn_axis,
                                            "policy")
        value_output = value_head(shared_output, self.params, self.data_format,
                                  self.bn_axis, "value")

        if training:
            self.model = keras.Model(inputs=self.features,
                                     outputs=[policy_output, value_output, logits])
        else:
            self.model = keras.Model(inputs=self.features,
                                     outputs=[policy_output, value_output])

    def initialize_weights(self, save_file):
        """Initialize the weights from the given save_file."""
        self.checkpoint.restore(os.path.abspath(save_file))

    def loss(self, labels, policy_output, value_output, logits):
        policy_cost = tf.reduce_mean(
            input_tensor=tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.stop_gradient(labels['pi_tensor']), logits=logits))

        value_cost = self.params['value_cost_weight'] * tf.reduce_mean(
            input_tensor=tf.square(value_output - labels['value_tensor']))

        reg_vars = [v for v in self.model.trainable_weights
                    if 'bias' not in v.name and 'beta' not in v.name]
        l2_cost = self.params['l2_strength'] * \
            tf.add_n([tf.nn.l2_loss(v) for v in reg_vars])

        return policy_cost + value_cost + l2_cost

    @tf.function
    def train_step(self, features, labels, first_batch=False):
        with tf.GradientTape() as tape:
            policy, value, logits = self.model(features, training=True)
            loss = self.loss(labels, policy, value, logits)

        tape = hvd.DistributedGradientTape(tape)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        #TODO (aayujain): still not sure what this does.
        # https://horovod.readthedocs.io/en/latest/tensorflow.html
        if first_batch:
            hvd.broadcast_variables(self.model.variables, root_rank=0)
            hvd.broadcast_variables(self.optimizer.variables(), root_rank=0)

        return loss

    def save_checkpoint(self, iteration_number):
        ckpt_path = self.manager.save(checkpoint_number=iteration_number)
        return ckpt_path

    def restore_checkpoint(self):
        """Restores the latest checkpoint."""
        #TODO(aayujain): Check if we can restore specific checkpoint by providing (state's) iteration number.
        self.checkpoint.restore(self.manager.latest_checkpoint)


class MigraphxBuffer:
    def __init__(self, model_buffer):
        self.graph = model_buffer

    def SerializeToString(self):
        return str(''.join(self.graph))


def freeze_graph(model_path, use_migraphx=True, migraphx_max_batch_size=128, inf_precision='fp16'):
    dual_net = DualNetwork(model_path, training=False)
    spec = tf.TensorSpec(dual_net.model.inputs[0].shape,
                         dual_net.model.inputs[0].dtype, name="pos_tensor")

    model_fn = tf.function(lambda x: dual_net.model(x))
    model_fn = model_fn.get_concrete_function(spec)
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    frozen_fn = convert_variables_to_constants_v2(model_fn)
    out_graph = frozen_fn.graph.as_graph_def()
    # hack to ensure compatibility with tf_dual_net (for eval)
    for node in out_graph.node:
        if node.name == "Identity":
            node.name = "policy_output"
        elif node.name == "Identity_1":
            node.name = "value_output"

    # eval is always fp32, so let's store a eval copy before we trt.
    metadata = make_model_metadata({
        'engine': 'tf',
        'use_trt': False,
    })
    output_path = model_path.replace("ckpt-", "0000")
    minigo_model.write_graph_def(out_graph, metadata, output_path + '.evalfp32minigo')

    if use_migraphx:
        path_breakdown = output_path.split('/')
        tf.io.write_graph(graph_or_graph_def=frozen_fn.graph,
                          logdir="/".join(path_breakdown[:-1]),
                          name=path_breakdown[-1]+".pb",
                          as_text=False)

        import migraphx
        #TODO(aayujain): add fn parse_tf_buffer to migraphx.
        model = migraphx.parse_tf(output_path+".pb", is_nhwc=False, batch_size=migraphx_max_batch_size)
        if inf_precision == 'fp16':
            migraphx.quantize_fp16(model)
        model.compile(migraphx.get_target("gpu"))
        out_graph = MigraphxBuffer(migraphx.save_buffer(model, 'json'))

        metadata = make_model_metadata({
            'engine': 'migraphx',
            'use_trt': bool(use_migraphx),
        })

    # double buffer model write
    minigo_model.write_graph_def(out_graph, metadata, output_path + '.stagedmodel')
    minigo_model.write_graph_def(out_graph, metadata, output_path + '.minigo')


def make_model_metadata(metadata):
    for f in ['conv_width', 'fc_width', 'trunk_layers', 'use_SE', 'use_SE_bias',
              'use_swish', 'input_features', 'input_layout']:
        metadata[f] = getattr(FLAGS, f)
    metadata['input_type'] = 'bool' if FLAGS.bool_features else 'float'
    metadata['board_size'] = go.N
    return metadata
