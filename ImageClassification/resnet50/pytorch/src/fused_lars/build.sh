#!/bin/bash

export MAX_JOBS=64
python3 -m pip uninstall -y fused_lars
python3 -m pip uninstall -y fused_lars
rm -rf build/
python3 setup.py clean --all
python3 setup.py install --cuda_ext --cpp_ext --user 2>&1 | tee build.log
