import argparse
import functools
import glob
import migraphx
import numpy as np
import preprocessing
#import tensorflow as tf
#from tensorflow import keras
#import dual_net
#import horovod.tensorflow as hvd

#from tensorflow.python.util import deprecation
#deprecation._PRINT_DEPRECATION_WARNINGS = False

import pdb


def get_input_for_resnet(args, train_dir="/data/imnet_small/train"):
    """Returns input batch to run Resnet with MIGraphx"""
    '''
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
                train_dir,  # Source directory for the training images
                target_size=(224, 224),
                batch_size=args.batchsize,
                # Since we use binary_crossentropy loss, we need binary labels
                class_mode='binary')

    batch = train_generator.next()
    '''
    batch = np.random.rand(args.batchsize, 3, 224, 224)
    batch = np.ascontiguousarray(batch.astype(np.float32))
    return batch

def get_input_for_minigo(args, train_files="/data/mlperf07/data/selfplay/**/*.tfrecord.zz"):
    # Data loading
    #hvd.init()
    #tf_records_ph = tf.placeholder(tf.string)
    '''
    dataset = preprocessing.read_tf_records(
                args.batchsize,
                tf_records_ph,
                num_repeats=1,
                shuffle_records=True,
                shuffle_examples=False,
                shuffle_buffer_size=0,
                filter_amount=0.3,
                interleave=False)
    dataset = dataset.filter(lambda t: tf.equal(tf.shape(t)[0], args.batchsize))
    dataset = dataset.map(functools.partial(preprocessing.batch_parse_tf_example, args.batchsize, 'mlperf07'), num_parallel_calls=8)
    # Ignoring random_rotation for now: minigo/preprocessing.py::200
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.apply(tf.data.experimental.prefetch_to_device('/gpu:0'))
    data_iter = dataset.make_initializable_iterator()
    '''
    '''
    data_iter = preprocessing.get_input_tensors(args.batchsize,
                        'nhwc' if args.is_nhwc else 'nchw',
                        tf_records_ph,
                        filter_amount=0.3,
                        shuffle_examples=False,
                        shuffle_buffer_size=0,
                        random_rotation=True)

    features, labels = data_iter.get_next()

    tf_records = glob.glob('/data/mlperf07/data/selfplay/**/*.tfrecord.zz', recursive=True)

    with dual_net._get_session() as dual_net_sess:
        dual_net_sess.run(data_iter.initializer, {tf_records_ph: tf_records})
        features_out = dual_net_sess.run(features)
        print(features_out.shape, features_out.dtype, type(features_out))
    '''

    features_out = np.random.rand(args.batchsize, 13, 19, 19) > 0.5
    return features_out


def main(args):
    print(type(args.is_nhwc), args.is_nhwc)
    if args.onnx:
        model = migraphx.parse_onnx(args.model, default_dim_value=args.batchsize)
    else:
        model = migraphx.parse_tf(args.model, args.is_nhwc, args.batchsize)

    migraphx.quantize_fp16(model)
    model.compile(migraphx.get_target("gpu"))
    # allocate space on GPU for model parameters
    #params = {}
    #for key, value in model.get_parameter_shapes().items():
    #    params[key] = migraphx.allocate_gpu(value)

    if "resnet" in args.model:
        print("Reading input for RESNET")
        inp = get_input_for_resnet(args)
    else:
        print("Reading input for MINIGO")
        inp = get_input_for_minigo(args)
    print("Input shape: ", inp.shape, inp.dtype)

    #params['0'] = migraphx.to_gpu(migraphx.argument(inp))
    print("Running with MIGraphx")
    #pdb.set_trace()
    #model.run(params)
    #print("RAN SUCCESSFULLY")
    #result = migraphx.from_gpu(model.run(params), copy=False)
    print(model.get_parameter_names())
    print(model.get_parameter_shapes())
    print(model.get_output_shapes())
    if args.onnx:
        key = 'input' if "resnet" in args.model else 'pos_tensor:0'
    else:
        key = 'input' if "resnet" in args.model else 'pos_tensor'
    migraphx_result = model.run({key:migraphx.argument(inp)})
    #result = np.array(migraphx_result, copy=False)
    #print("Result:", migraphx_result)
    print("len(result):", len(migraphx_result))
    if args.onnx:
        policy_output = np.array(migraphx_result[0])
        value_output = np.array(migraphx_result[1])
        print(policy_output.shape)
        print(value_output.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True, help='path to frozen model')
    parser.add_argument('-b', '--batchsize', type=int, default=256, help='batch size')
    parser.add_argument('--is_nhwc', default=False, action='store_true', help='is_nhwc?')
    parser.add_argument('--onnx', default=False, action='store_true', help='ONNX file?')
    args = parser.parse_args()
    args.onnx = args.onnx or ("onnx" in args.model)
    main(args)
