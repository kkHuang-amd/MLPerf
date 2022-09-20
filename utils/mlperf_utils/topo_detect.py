import torch
import torch.distributed as dist
import xml.etree.ElementTree as ET
import shutil
import os
from multiprocessing import Process
import subprocess
import re

def run_dist(rank, world_size, local_rank):
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl",
                            world_size=world_size,
                            rank=rank,
                            init_method="env://")
    dist.barrier()

    tensor = torch.tensor([rank], dtype=torch.float32, device="cuda")

    dist.all_reduce(tensor)

    dist.barrier()
    torch.cuda.synchronize()

def find_gpu_in_rccl_xml(rccl_topo_file):
    root = ET.parse(rccl_topo_file).getroot()
    gpu_numa_map = {}
    num_gpus_per_numa = {}

    def _find_gpu_in_rccl_xml(xml, numa_id):
        if xml.tag == "gpu":
            gpu_id = int(xml.attrib["rank"])
            gpu_numa_map[gpu_id] = numa_id
            if numa_id not in num_gpus_per_numa:
                num_gpus_per_numa[numa_id] = 1
            else:
                num_gpus_per_numa[numa_id] += 1
        else:
            if xml.tag == "cpu":
                numa_id = int(xml.attrib["numaid"])
            for child in xml:
                _find_gpu_in_rccl_xml(child, numa_id)

    _find_gpu_in_rccl_xml(root, None)

    return gpu_numa_map, num_gpus_per_numa

def find_cpu_in_hwloc_xml():
    hwloc_bin = shutil.which("hwloc-ls")
    assert hwloc_bin is not None, "hwloc-ls not found.  Please install hwloc."

    numa_cpu_map = {}

    def is_type(xml, xml_type):
        return xml.tag == "object" and xml.attrib["type"] == xml_type

    def _find_cpu_in_hwloc_xml(xml, numa_id):
        if is_type(xml, "Core"):
            # Handle the case that there is only one NUMA node
            if numa_id is None:
                numa_id = 0

            # Each child is a hardware thread
            c_idx = 0
            for child in xml:
                if is_type(child, "PU"):
                    pu_id = int(child.attrib["os_index"])

                    if numa_id not in numa_cpu_map:
                        numa_cpu_map[numa_id] = [[pu_id]]
                    elif len(numa_cpu_map[numa_id]) == c_idx:
                        numa_cpu_map[numa_id].append([pu_id])
                    else:
                        numa_cpu_map[numa_id][c_idx].append(pu_id)
                    c_idx += 1
        else:
            # In hwloc v1 the NUMANode is some ancestor of the Core.
            # In hwloc v2 the NUMANode is a child of some ancestor of the Core.
            # Handle either case here.
            if is_type(xml, "NUMANode"):
                numa_id = int(xml.attrib["os_index"])
            for child in xml:
                if is_type(child, "NUMANode"):
                    numa_id = int(child.attrib["os_index"])
                _find_cpu_in_hwloc_xml(child, numa_id)

    hwloc_info = os.popen("{} --output-format xml".format(hwloc_bin)).read()
    root = ET.ElementTree(ET.fromstring(hwloc_info)).getroot()

    _find_cpu_in_hwloc_xml(root, None)

    return numa_cpu_map

def get_cpu_offsets(num_gpus_per_numa, numa_cpu_map):
    offsets = {}
    for numa_id in range(len(numa_cpu_map)):
        if numa_id in num_gpus_per_numa:
            num_gpus = num_gpus_per_numa[numa_id]
            num_cpus = len(numa_cpu_map[numa_id][0])
            num_cores_per_gpu = num_cpus // num_gpus
            remainder = num_cpus % num_gpus

            offsets[numa_id] = [0]
            for g in range(num_gpus):
                ncpus = num_cores_per_gpu + (1 if g < remainder else 0)
                offsets[numa_id].append(offsets[numa_id][-1] + ncpus)
        else:
            print("Warning: CPUs in NUMA {} are not used".format(numa_id))
    return offsets

def get_gpu_numa_map(args, rccl_topo_file, keep_topo_file):
    dist_world_size = args.nproc_per_node * args.nnodes

    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)
    os.environ["WORLD_SIZE"] = str(dist_world_size)
    os.environ["NCCL_TOPO_DUMP_FILE"] = rccl_topo_file

    processes = []
    for local_rank in range(args.nproc_per_node):
        dist_rank = args.nproc_per_node * args.node_rank + local_rank
        p = Process(target=run_dist, args=(dist_rank, dist_world_size, local_rank))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    gpu_numa_output = find_gpu_in_rccl_xml(rccl_topo_file)

    del os.environ["NCCL_TOPO_DUMP_FILE"]

    if not keep_topo_file:
        os.remove(rccl_topo_file)

    return gpu_numa_output

def get_numactl_args(args, rccl_topo_file, keep_topo_file=False, return_all_configs=False):
    gpu_numa_map, num_gpus_per_numa = get_gpu_numa_map(args, rccl_topo_file, keep_topo_file)

    numa_cpu_map = find_cpu_in_hwloc_xml()

    cpu_offsets = get_cpu_offsets(num_gpus_per_numa, numa_cpu_map)

    gpu_count_per_numa = [0 for _ in range(len(numa_cpu_map))]

    rank_numactlargs = []

    for local_rank in range(0, args.nproc_per_node):
        numa_id = gpu_numa_map[local_rank]
        gpu_id = gpu_count_per_numa[numa_id]
        gpu_count_per_numa[numa_id] += 1
        offsets = cpu_offsets[numa_id]
        cores = None
        for cpus in numa_cpu_map[numa_id]:
            cores_str = "{}-{}".format(cpus[offsets[gpu_id]],
                                       cpus[offsets[gpu_id + 1] - 1])
            if cores is None:
                cores = cores_str
            else:
                cores += "," + cores_str

        rank_numactlargs.append(["--physcpubind={}".format(cores),
                                 "--membind={}".format(numa_id)])

    if return_all_configs:
        return gpu_numa_map, num_gpus_per_numa, numa_cpu_map, cpu_offsets, rank_numactlargs
    else:
        return rank_numactlargs

def get_gpu_arch():
    gpu_arch_name = {"gfx908": "MI100 system",
                     "gfx90a": "MI200 system",
                     "gfx1030": "NV21 system"}

    gpus = subprocess.check_output("/opt/rocm/bin/rocminfo").decode('UTF-8').split('\n')
    gpus = [re.search('(gfx.*[0-9a-z])', g).group(0) for g in gpus if 'gfx' in g]

    if len(gpus) > 1:
        print("Warning: There is more than one GPU architecture on the system")

    if gpus[0] in gpu_arch_name:
        return gpu_arch_name[gpus[0]]
    return "AMD system"
