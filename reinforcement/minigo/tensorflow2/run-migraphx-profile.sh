folder='results-bs256-hip-stream-varsleep'
rocprof --hip-trace -d "${folder}1" --timestamp on python3 migraphx-profile.py -m /opt/reinforcement/saver/model.ckpt-5672.onnx -b 256 -n 200 --onnx
rocprof --hip-trace -d "${folder}2" --timestamp on mpirun --allow-run-as-root -np 2 python3 migraphx-profile.py -m /opt/reinforcement/saver/model.ckpt-5672.onnx -b 256 -n 200 --onnx
rocprof --hip-trace -d "${folder}3" --timestamp on mpirun --allow-run-as-root -np 3 python3 migraphx-profile.py -m /opt/reinforcement/saver/model.ckpt-5672.onnx -b 256 -n 200 --onnx

