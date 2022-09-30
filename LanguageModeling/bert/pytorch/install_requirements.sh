#!/bin/bash
pip3 install ninja
pip3 install --no-cache-dir -r requirements.txt
cd mhalib
python3 setup.py build && cp build/lib*/mhalib* ../
cd ../

apex_path=`pip show apex | grep Location | awk -F ":" '{print $2}'`

cp optim/distributed_fused_lamb.py ${apex_path}/apex/contrib/optimizers/
#cd apexcontrib
#python setup.py install --distributed_lamb --deprecated_fused_adam
#cd ../
