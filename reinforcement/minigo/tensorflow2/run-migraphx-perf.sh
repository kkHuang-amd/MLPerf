python3 migraphx-perf.py -m /opt/reinforcement/saver/model.ckpt-5672.onnx -b 512 -n 2000 --onnx
mpirun --allow-run-as-root -np 2 python3 migraphx-perf.py -m /opt/reinforcement/saver/model.ckpt-5672.onnx -b 512 -n 2000 --onnx
mpirun --allow-run-as-root -np 3 python3 migraphx-perf.py -m /opt/reinforcement/saver/model.ckpt-5672.onnx -b 512 -n 2000 --onnx

