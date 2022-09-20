import tf2onnx

import glob

import sys
sys.path.insert(0, '.')  # nopep8

import logging
import os

import ml_perf.train_loop as train_loop

import tensorflow as tf
import dual_net
import preprocessing
import horovod.tensorflow as hvd
from tensorflow.python.tools import freeze_graph

from absl import app, flags
from mpi4py import MPI
import socket

#import minigo_python
import shutil


flags.DEFINE_string('model', '/data/mlperf07/work_dir/model.ckpt-5672',
                    'path to model checkpoint')
flags.DEFINE_string('save_dir', '/opt/reinforcement/saver/',
                    'directory for saving (frozen) graphs')

FLAGS = flags.FLAGS

def main(unused_argv):
    #model_loc = '/data/mlperf07/work_dir/model.ckpt-5672'
    #save_loc = '/opt/reinforcement/saver/'
    model_loc = FLAGS.model
    save_loc = FLAGS.save_dir
    model_binary=True

    cleanup(save_loc)
    init()

    n = dual_net.DualNetwork(model_loc)
    logging.info('MODEL {}'.format(n))

    graph_file = save_graph(model_loc, save_loc, model_binary)
    logging.info("SAVED GRAPH: {}".format(os.path.join(save_loc, graph_file)))
    frozen_graph_path = freeze(model_loc, save_loc, graph_file, model_binary)
    logging.info("FROZEN MODEL: {}".format(frozen_graph_path))
    run_frozen_graph(frozen_graph_path)
    #logging.info("MIGRAPHX - parsing frozen model")
    #from migraphx import parse_tf
    #parse_tf(frozen_graph_path, False)

    onnx_file = save_onnx(frozen_graph_path, model_loc, save_loc)
    logging.info("Saving onnx file: {}".format(onnx_file))



def init():
    train_loop.init_logger()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    logging.info('MPI rank {} size {}'.format(rank, size))

    train_ranks = [0]
    tcomm = train_loop.get_group_comm(comm, train_ranks)

    FLAGS.num_nodes = 1
    FLAGS.num_train_nodes = 1
    FLAGS.num_selfplay_nodes = 1

    train_loop.init_flags(rank)

    hvd.init(tcomm)


def cleanup(save_loc):
    if os.path.exists(save_loc) and os.path.isdir(save_loc):
        shutil.rmtree(save_loc)


def save_graph(model_loc, save_loc, save_binary=True):
    '''Returns the name of the graph file'''
    n = dual_net.DualNetwork(model_loc)

    model_name = model_loc.split('/')[-1]
    ext = '.pb' if save_binary else '.pbtxt'
    output_file = model_name + ext

    with n.sess.graph.as_default() as graph:
        # Option 1: pb, pbtxt
        #tf.train.write_graph(graph.as_graph_def(), save_loc, output_file, as_text=not save_binary)

        # Option 2: pb
        #os.mkdir(save_loc)
        #with tf.gfile.GFile(os.path.join(save_loc, output_file), 'wb') as f:
        #    f.write(graph.as_graph_def().SerializeToString())

        # Option 3: pb, pbtxt
        tf.io.write_graph(graph.as_graph_def(), save_loc, output_file, as_text=not save_binary)

    return output_file


def freeze(model_loc, save_loc, graph_file, input_binary=True):
    model_name = model_loc.split('/')[-1]
    input_graph_path = os.path.join(save_loc, graph_file)
    input_saver_def_path = ""
    output_node_names = "policy_output,value_output"
    output_graph_path = os.path.join(save_loc, model_name + "-frozen.pb")
    clear_devices = True

    freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                input_binary, model_loc,
                output_node_names, "", "",
                output_graph_path, clear_devices, "")

    return output_graph_path


def save_onnx(frozen_model, model_loc, save_loc):

    n = dual_net.DualNetwork(model_loc)
    model_name =  model_loc.split('/')[-1]
    onnx_file = save_loc + model_name + ".onnx"

    # WORKS
    graph_def, inputs, outputs = tf2onnx.tf_loader.from_graphdef(frozen_model, ["pos_tensor:0"], ["policy_output:0", "value_output:0"])
    with tf.Graph().as_default() as tf_graph:
        tf.import_graph_def(graph_def, name="")
    with tf2onnx.tf_loader.tf_session(graph=tf_graph):
        onnx_graph = tf2onnx.tfonnx.process_tf_graph(tf_graph, input_names=inputs, output_names=outputs)
        onnx_proto = onnx_graph.make_model("minigo-selfplay")
        with open(onnx_file, 'wb') as f:
            f.write(onnx_proto.SerializeToString())

    return onnx_file
    '''
    output_names = ['policy_output', 'value_output']
    out_graph = tf.graph_util.convert_variables_to_constants(
            n.sess, n.sess.graph.as_graph_def(), output_names)

    inputs = ["pos_tensor:0"]
    outputs = ["policy_output:0", "value_output:0"]
    with tf.Graph().as_default() as tf_graph:
        tf.import_graph_def(out_graph, name="")
    with tf2onnx.tf_loader.tf_session(graph=tf_graph):
        onnx_graph = tf2onnx.tfonnx.process_tf_graph(tf_graph, input_names=inputs, output_names=outputs)

    print("save_onnx::new approach")
    return onnx_file

    #with tf.Graph().as_default() as tf_graph:
    #    tf.import_graph_def(out_graph, "")
    #with tf.Session(graph=n.sess.graph) as sess:
    with n.sess as sess:
        #for node in sess.graph.as_graph_def().node:
        #    print(node.op, node.name)
        onnx_graph = tf2onnx.tfonnx.process_tf_graph(
                sess.graph,
                input_names=["pos_tensor:0"],
                output_names=["policy_output:0", "value_output:0"])

        model_proto = onnx_graph.make_model("selfplay")
        with open(onnx_file, 'wb') as f:
            f.write(model_proto.SerializeToString())

    return onnx_file
    '''


def manual_freeze(model_path, save_loc):
    model_name = model_path.split('/')[-1]
    frozen_file_name = os.path.join(save_loc, model_name + '-frozen_alt.pb')

    output_names = ['policy_output', 'value_output']

    n = dual_net.DualNetwork(model_path)
    out_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
        n.sess, n.sess.graph.as_graph_def(), output_names)

    with open(frozen_file_name, 'wb') as f:
        f.write(out_graph.SerializeToString())
    #tf.io.write_graph(out_graph, save_loc, frozen_file_name, as_text=False)

    return frozen_file_name


'''
def save_model(model_loc):
    n = dual_net.DualNetwork(model_loc)
    logging.info('MODEL {}'.format(n))

    save_loc = '/opt/reinforcement/savedmodel/1/'
    model_name = 'model.ckpt-5672'

    graph = n.sess.graph
    with graph.as_default():
    #with dual_net._get_session() as sess:
        #(input_ph, _) = dual_net.get_inference_input()
        input_tensor = graph.get_tensor_by_name("pos_tensor:0")
        policy_tensor = graph.get_tensor_by_name("policy_output:0")
        value_tensor = graph.get_tensor_by_name("value_output:0")
        logging.info("INPUT Op: {}\nPOLICY Op: {}\nVALUE Op: {}".format(input_tensor, policy_tensor, value_tensor))
        tf.saved_model.simple_save(n.sess, save_loc, inputs={"pos_tensor": input_tensor}, outputs={"policy_output": policy_tensor, "value_output": value_tensor})
        logging.info("*** SAVED MODEL")
'''


def run_frozen_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    #placeholder_nodes = [n for n in graph_def.node if n.op == 'Placeholder']

    #(input_placeholder,_) = dual_net.get_inference_input()
    #logging.info("Input Placeholder: {}".format(input_placeholder))

    with tf.Graph().as_default() as graph:
        #tf.import_graph_def(graph_def, input_map={'pos_tensor:0': input_placeholder}, return_elements=['policy_output:0','value_output:0'], name="")
        tf.import_graph_def(graph_def, name="")


    tf_records_ph = tf.placeholder(tf.string)
    data_iter = preprocessing.get_input_tensors(FLAGS.train_batch_size,
                        FLAGS.input_layout,
                        tf_records_ph,
                        filter_amount=FLAGS.filter_amount,
                        shuffle_examples=FLAGS.shuffle_examples,
                        shuffle_buffer_size=FLAGS.shuffle_buffer_size,
                        random_rotation=True)

    features, labels = data_iter.get_next()

    tf_records = glob.glob('/data/mlperf07/data/selfplay/**/*.tfrecord.zz', recursive=True)

    with dual_net._get_session() as dual_net_sess:
        dual_net_sess.run(data_iter.initializer, {tf_records_ph: tf_records})
        features_out = dual_net_sess.run(features)
        logging.info("LOAD FEATURES: {}, {}".format(features_out.shape, features_out.dtype))

    with tf.Session(graph=graph) as sess:
        input_tensor = graph.get_tensor_by_name("pos_tensor:0")
        policy_tensor = graph.get_tensor_by_name("policy_output:0")
        value_tensor = graph.get_tensor_by_name("value_output:0")
        #logging.info("INPUT Tensor: {}\nPOLICY Tensor: {}\nVALUE Tensor: {}".format(input_tensor, policy_tensor, value_tensor))
        results = sess.run([policy_tensor, value_tensor], feed_dict={input_tensor: features_out})
        logging.info("*** FROZEN RUN SUCCESSFUL ***")
        logging.info("Frozen graph output - policy: {};\t value: {}".format(policy_tensor.get_shape(), value_tensor.get_shape()))
        print(results)


def run_inference_graph():
    tf_records_ph = tf.placeholder(tf.string)
    data_iter = preprocessing.get_input_tensors(FLAGS.train_batch_size,
                        FLAGS.input_layout,
                        tf_records_ph,
                        filter_amount=FLAGS.filter_amount,
                        shuffle_examples=FLAGS.shuffle_examples,
                        shuffle_buffer_size=FLAGS.shuffle_buffer_size,
                        random_rotation=True)

    features, labels = data_iter.get_next()

    tf_records = glob.glob('/data/mlperf07/data/selfplay/**/*.tfrecord.zz', recursive=True)
    #with tf.Session() as sess:
    #   sess.run(data_iter.initializer, {tf_records_ph: tf_records})
    #logging.info("EXPORT INF GRAPH - TF Records: {}".format(tf_records_ph.shape))
    #n.sess.run(n.inference_output, feed_dict={n.inference_input: features})
    #logging.info("Outputs: {}".format(n.inference_output))

    #train_op = dual_net.model_fn(features, labels, tf.estimator.ModeKeys.TRAIN, FLAGS.flag_values_dict(), True)
    train_op = dual_net.model_fn(features, labels, tf.estimator.ModeKeys.EVAL, FLAGS.flag_values_dict(), False)
    sess = dual_net._get_session()
    tf.train.Saver().restore(sess, os.path.join('/opt/reinforcement/saver', 'model.ckpt-5672'))

    sess.run(data_iter.initializer, {tf_records_ph: tf_records})
    logging.info("TF Records: {}".format(tf_records_ph.shape))
    with sess:
        #result = sess.run(train_op)
        result = sess.run(train_op.predictions)
    logging.info("*** RAN SUCCESSFULLY ***\n{}".format(result))
    if 'policy_output' in result.keys():
        logging.info("Policy Output Shape: {}".format(result['policy_output'].shape))
    if 'value_output' in result.keys():
        logging.info("Value Output Shape: {}".format(result['value_output'].shape))


if __name__ == '__main__':
    app.run(main)
