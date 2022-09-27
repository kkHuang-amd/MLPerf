import argparse
import functools
import glob
import migraphx
import numpy as np
import os
import time
from mpi4py import MPI
import csv
#import tensorflow as tf
#from tensorflow import keras
#import dual_net
#import horovod.tensorflow as hvd

#from tensorflow.python.util import deprecation
#deprecation._PRINT_DEPRECATION_WARNINGS = False


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


def get_minigo_input(batchsize, is_nhwc):
    return np.random.rand(batchsize,13,19,19) > np.random.normal(0,1,1)


def get_resnet_input(batchsize, is_nhwc):
    if is_nhwc:
        return np.ascontiguousarray(np.random.rand(batchsize, 224, 224, 3).astype(np.float32))
    else:
        return np.ascontiguousarray(np.random.rand(batchsize,3,224,224).astype(np.float32))


def timed_run(model, key, input_fn, batchsize, is_nhwc, iterations, runs=1):
    """Return iteration time, averaged over `runs` runs."""
    elapsed_times = []
    inp = input_fn(batchsize, is_nhwc)
    for _ in range(runs+1):
        start = time.time()
        for i in range(iterations):
            #inp = get_input_for_minigo(args)
            #inp = np.random.rand(batchsize, 13, 19, 19) > np.random.normal(0,1,1)
            #inp = input_fn(batchsize, is_nhwc)
            #print("input:", inp.shape, inp.dtype)
            #time.sleep(0.02)
            time.sleep((10 + np.random.randint(11))/1000)
            migraphx_result = model.run({key:migraphx.argument(inp)})
            #print("output:", len(migraphx_result))
        end = time.time()
        elapsed_times.append(end-start)

    # first run takes disproportionately more time.
    return elapsed_times[1:]


def get_model(args, batchsize, fp16=True):
    if args.onnx:
        model = migraphx.parse_onnx(args.model, default_dim_value=batchsize)
    else:
        model = migraphx.parse_tf(args.model, args.is_nhwc, batchsize)

    if fp16:
        migraphx.quantize_fp16(model)

    model.compile(migraphx.get_target("gpu"))
    
    return model


def get_time(comm):
    comm.barrier()
    return time.time()


def write_logs(logs, header, filename="perf.csv"):
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in logs:
            writer.writerow(row)


def main(args):
    # Ensure all procs run on the same gpu
    os.environ['HIP_VISIBLE_DEVICES'] = "0"
    print("print env", os.environ.get('AMD_SERIALIZE_KERNEL'))

    # MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print("MPI:: Size: {}, Rank: {}".format(size, rank))
    #local_batchsize = args.batchsize // size

    key = 'pos_tensor:0' if args.onnx else 'pos_tensor'
    #key = "input:0"

    print("Running for {} iterations - averaged over 10 runs.".format(args.iterations))
    logs = []
    #enable_fp16 = [True, False]
    enable_fp16 = [True]
    for fp16_flag in enable_fp16:
        bs = args.batchsize
        #bs = 1
        while bs <= args.batchsize:
            if rank==0: print("{} - Per Process Batch Size: {}".format("fp16" if fp16_flag else "fp32", bs))
            model = get_model(args, bs, fp16_flag)
            wall_start = get_time(comm)
            iter_times = timed_run(model, key, get_minigo_input, bs, args.is_nhwc, args.iterations)
            #iter_times = timed_run(model, key, get_resnet_input, bs, args.is_nhwc, args.iterations)
            wall_stop = get_time(comm)

            send_buf = np.array(iter_times)
            recv_buf = None
            if rank == 0:
                recv_buf = np.empty([size]+list(send_buf.shape))
            comm.Gather(send_buf, recv_buf, root=0)

            if rank == 0:
                wall_time = wall_stop - wall_start
                print("wall time: {} seconds".format(wall_time))
                avg_iter_time = np.mean(recv_buf)
                print("Iteration Time: {} seconds".format(avg_iter_time))
                print("Batch Time: {} s".format(avg_iter_time/args.iterations))
                print("IPS: {} imgs/s".format((bs * args.iterations) / avg_iter_time))
                for rank_i, times in enumerate(recv_buf):
                    print(rank_i, np.mean(times))
                    for time in times:
                        logs.append(('fp16' if fp16_flag else 'fp32', bs, size, rank_i, time, time/args.iterations, (bs*args.iterations)/time))

            bs *= 2

    #header = ("precision", "batchsize", "size", "rank", "iter_time(s)", "batch_time(s)", "throughput(ips)")
    #write_logs(logs, header, "minigo7-z53-migraphx-inf-perf-peak-proc{}.log".format(size))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True, help='path to frozen model')
    parser.add_argument('-b', '--batchsize', type=int, default=256, help='global batch size')
    parser.add_argument('-n', '--iterations', type=int, default=100, help='iterations')
    parser.add_argument('--is_nhwc', default=False, action='store_true', help='is_nhwc?')
    parser.add_argument('--onnx', default=False, action='store_true', help='ONNX file?')
    args = parser.parse_args()
    main(args)
