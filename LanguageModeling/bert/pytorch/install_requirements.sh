#!/bin/bash
pip install ninja
pip install --no-cache-dir -r requirements.txt
cd mhalib
python setup.py build && cp build/lib*/mhalib* ../
cd ../
#cd apexcontrib
#python setup.py install --distributed_lamb --deprecated_fused_adam
#cd ../
