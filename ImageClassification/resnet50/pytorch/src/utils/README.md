# Utils

Helper modules for MLPerf workloads

## Installation

```bash
python3 setup.py install
```

## Usage

### Resource binding launcher (``bind_launch``)

Requires ``RCCL`` and ``hwloc``

Use automatic resource binding.

```bash
python3 -m mlperf_utils.bind_launch \
    --nproc_per_node <number of GPU per node> \
    --nnodes <number of nodes> \
    --auto_binding
```

See full usage

```bash
python3 -u -m mlperf_utils.bind_launch -h
```
