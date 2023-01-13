import argparse
import ast

def parse_arguments(add_help=True):
    parser = argparse.ArgumentParser(description="Check RetinaNet Throughput", add_help=add_help)
    parser.add_argument("-f", "--file", help="RetinaNet log file")
    args = parser.parse_args()
    return args

mlperf_log = 'MLLOG'
key = 'key'
value = 'value'
epoch_stop = 'epoch_stop'
target = 'throughput'

if __name__ == '__main__':
    args = parse_arguments()
    print(f'RetinaNet log file: {args.file}')
    val, last_val = float(), float()
    idx = 0
    is_stop = False
    with open(args.file, 'r') as file_desc:
        for line in file_desc.readlines():
            if mlperf_log in line and 'null' not in line:
                start = line.index('{')
                curr = ast.literal_eval(line[start:])
                if key in curr and curr[key] == epoch_stop:
                    is_stop = True
                    idx = curr[value]
                elif is_stop and value in curr and target in curr[value]:
                    print(f'[Epoch {idx}] Throughput: {curr[value][target]:.3f} samples/second')
                    val += curr[value][target]
                    last_val = curr[value][target]
                    is_stop = False
    val = (val - last_val) / idx   # ignore the last incomplete epoch
    print('==================================================')
    print(f'Average Throughput: {val:.3f} samples/second')
