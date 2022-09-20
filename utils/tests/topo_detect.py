from argparse import ArgumentParser
import mlperf_utils.topo_detect as topo

def parse_args():
    parser = ArgumentParser(description="A test program for auto topology detection")

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

    return parser.parse_args()

def main():
    args = parse_args()
    rccl_topo_file = "rccl_topo.xml"
    gpu_numa_map, num_gpus_per_numa, numa_cpu_map, cpu_offsets, numactlargs \
            = topo.get_numactl_args(args,
                                    rccl_topo_file,
                                    keep_topo_file=True,
                                    return_all_configs=True)

    print("GPU-NUMA map (gpu_numa_map): {}".format(gpu_numa_map))

    print("\nGPU count per NUMA (num_gpus_per_numa): {}".format(num_gpus_per_numa))

    print("\nNUMA-CPU map (numa_cpu_map):")
    for numa_id in numa_cpu_map:
        print("`- NUMA {}".format(numa_id))
        for hwt, cpus in enumerate(numa_cpu_map[numa_id]):
            print("   `- {}: {}".format(hwt, cpus))

    print("\nCPU offsets (cpu_offsets): {}".format(cpu_offsets))

    print("\nRCCL topo file: {}".format(rccl_topo_file))
    print("\nnumactl args:")

    for rank in range(args.nproc_per_node):
        print("`- rank {}: {}".format(rank, numactlargs[rank]))

if __name__ == "__main__":
    main()
