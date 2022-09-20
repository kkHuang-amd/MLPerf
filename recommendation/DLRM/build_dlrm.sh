#!/bin/bash

# Install Mlcommons logging for MLPerf
pip install --no-cache-dir git+https://github.com/mlcommons/logging.git@1.1.0-rc2

# Install required packages
sudo apt-get update && apt-get install -y libnuma-dev numactl hwloc && apt-get clean

pip install ninja

pip -v install --no-cache-dir -e .
