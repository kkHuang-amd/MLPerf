import os
from setuptools import setup

if os.system("apt-get install -y hwloc"):
    print ("ERROR: Failed to install hwloc.")
    sys.exit(1)
if os.system("apt-get install -y numactl"):
    print ("ERROR: Failed to install numactl.")
    sys.exit(1)

setup(
    name="mlperf_utils",
    version="0.0.1",
    author="AMD MLPerf",
    description=("A program for launching a Python program with resource "
                 "binding."),
    packages=["mlperf_utils"],
)
