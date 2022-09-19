import sys
import subprocess
import os
import socket
from argparse import ArgumentParser, REMAINDER
from .topo_detect import get_numactl_args, get_gpu_arch

def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(description="A modified PyTorch distributed training launch "
                                        "helper utilty that will spawn up "
                                        "multiple distributed processes")

    # Optional arguments for the launch helper
    parser.add_argument("--nnodes", type=int, default=1,
                        help="The number of nodes to use for distributed "
                             "training")
    parser.add_argument("--node_rank", type=int, default=0,
                        help="The rank of the node for multi-node distributed "
                             "training")
    parser.add_argument("--nproc_per_node", type=int, default=1,
                        help="The number of processes to launch on each node, "
                             "for GPU training, this is recommended to be set "
                             "to the number of GPUs in your system so that "
                             "each process can be bound to a single GPU.")
    parser.add_argument("--master_addr", default="127.0.0.1", type=str,
                        help="Master node (rank 0)'s address, should be either "
                             "the IP address or the hostname of node 0, for "
                             "single node multi-proc training, the "
                             "--master_addr can simply be 127.0.0.1")
    parser.add_argument("--master_port", default=29500, type=int,
                        help="Master node (rank 0)'s free port that needs to "
                             "be used for communciation during distributed "
                             "training")
    parser.add_argument('--no_hyperthreads', action='store_true',
                        help='Flag to disable binding to hyperthreads')
    parser.add_argument('--no_cpubind', action='store_true',
                        help='Flag to disable cpu binding')
    parser.add_argument('--no_membind', action='store_true',
                        help='Flag to disable memory binding')
    parser.add_argument('--auto_binding', action='store_true',
                        help='Flag to enable automatic binding using information from '
                             'NCCL_TOPO_DUMP_FILE and hwloc')
    parser.add_argument('--cpubind_type', type=str, default='bind_to_numas',
                        choices=['bind_to_numas', 'bind_to_sockets', 'bind_to_cores'],
                        help='Flag to set the CPU binding type')
    parser.add_argument('--membind_type', type=str, default='bind_to_numas',
                        choices=['bind_to_numas', 'bind_to_sockets'],
                        help='Flag to set the mem binding type')

    # non-optional arguments for binding
    parser.add_argument("--nsockets_per_node", type=int, default=None,
                        help="Number of CPU sockets on a node")
    parser.add_argument("--ncores_per_socket", type=int, default=None,
                        help="Number of CPU cores per socket")
    parser.add_argument("--nnuma_nodes", type=int, default=None,
                        help="Number of NUMA nodes")

    # Debug options
    parser.add_argument("--dump_to_stdout", default=False, action="store_true",
                        help="Dump all output to stdout")
    parser.add_argument("--dump_to_separate_files", default=False, action="store_true",
                        help="Dump outputs from stdout and stderr in separate files")
    parser.add_argument("--profiling_cmd", default="", type=str,
                        help="Profiling command (comma separated)")

    # positional
    parser.add_argument("training_script", type=str,
                        help="The full path to the single GPU training "
                             "program/script to be launched in parallel, "
                             "followed by all the arguments for the "
                             "training script")

    # rest from the training program
    parser.add_argument('training_script_args', nargs=REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()

    # world size in terms of number of processes
    dist_world_size = args.nproc_per_node * args.nnodes

    # set PyTorch distributed related environmental variables
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)
    os.environ["WORLD_SIZE"] = str(dist_world_size)
    os.environ["NUM_NODES"] = str(args.nnodes)

    if args.auto_binding:
        rank_numactlargs= get_numactl_args(args,
                                           "rccl_topo.xml",
                                           keep_topo_file=False)
    else:
        assert args.nsockets_per_node is not None
        assert args.ncores_per_socket is not None
        assert args.nnuma_nodes is not None

        # variables for numactrl binding
        NSOCKETS = args.nsockets_per_node
        NGPUS_PER_SOCKET = args.nproc_per_node // args.nsockets_per_node
        NCORES_PER_GPU = args.ncores_per_socket // NGPUS_PER_SOCKET
        NGPUS_PER_NUMA_NODE = args.nproc_per_node // args.nnuma_nodes

    os.environ["MLPERF_SUBMISSION_SYSTEM"] = get_gpu_arch()

    current_env = os.environ.copy()

    processes = []

    for local_rank in range(0, args.nproc_per_node):
        # each process's rank
        dist_rank = args.nproc_per_node * args.node_rank + local_rank
        current_env["RANK"] = str(dist_rank)
        current_env["LOCAL_RANK"] = str(local_rank)

        numactlargs = []

        if args.auto_binding:
            numactlargs = rank_numactlargs[local_rank]
        else:
            if not args.no_cpubind:
                # Instead of binding to a set of cores which this task has exclusive access to,
                #   bind to all cores on the local NUMA node (may share them with other ranks)
                if args.cpubind_type == 'bind_to_numas':
                    local_node = local_rank // NGPUS_PER_NUMA_NODE
                    numactlargs = ["--cpunodebind={}".format(local_node)]
                elif args.cpubind_type == 'bind_to_sockets':
                    local_node = local_rank // NGPUS_PER_SOCKET
                    numactlargs = ["--cpunodebind={}".format(local_node)]
                else:
                    ## form numactrl binding command
                    cpu_ranges = [local_rank * NCORES_PER_GPU,
                                 (local_rank + 1) * NCORES_PER_GPU - 1,
                                 local_rank * NCORES_PER_GPU + (NCORES_PER_GPU * NGPUS_PER_SOCKET * NSOCKETS),
                                 (local_rank + 1) * NCORES_PER_GPU + (NCORES_PER_GPU * NGPUS_PER_SOCKET * NSOCKETS) - 1]

                    if args.no_hyperthreads:
                        numactlargs += [ "--physcpubind={}-{}".format(*cpu_ranges[0:2]) ]
                    else:
                        numactlargs += [ "--physcpubind={}-{},{}-{}".format(*cpu_ranges) ]

            if not args.no_membind:
                if args.membind_type == 'bind_to_numas':
                    memnode = local_rank // NGPUS_PER_NUMA_NODE
                else:
                    memnode = local_rank // NGPUS_PER_SOCKET
                numactlargs += [ "--membind={}".format(memnode) ]

        cmd = []
        if len(numactlargs) > 0:
            cmd = [ "/usr/bin/numactl" ] \
                + numactlargs \

        if args.profiling_cmd != "":
            cmd += args.profiling_cmd.format(rank=dist_rank).split(',')

        cmd += [ sys.executable,
                "-u",
                args.training_script
              ] \
            + args.training_script_args

        print(cmd)

        if args.dump_to_stdout:
            stdout = None
            stderr = None
        elif args.dump_to_separate_files:
            stdout = open('gpu-%d.out' % local_rank, 'w')
            stderr = open('gpu-%d.err' % local_rank, 'w')
        else:
            stdout = None if local_rank == 0 else open(os.devnull, 'w')
            stderr = None if local_rank == 0 else open(os.devnull, 'w')

        # spawn the processes
        process = subprocess.Popen(cmd, env=current_env, stdout=stdout, stderr=stderr)
        processes.append(process)

    for process in processes:
        process.wait()


if __name__ == "__main__":
    main()
