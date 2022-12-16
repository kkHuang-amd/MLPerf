import sys
import os
import subprocess
import re
import datetime

storage_file = "/tmp/miopen"
cmd_line = "MIOpenDriver convfp16"

d_cmds = {}

def find_in_memory(ln):
    if d_cmds.__contains__(ln):
        return 1
    else:
        return 0

def increase_in_memory(ln):
    d_cmds[ln][0] = d_cmds[ln][0] + 1

def add_in_memory(ln):
    d_cmds[ln] = list()
    d_cmds[ln].append(0)

def get_time_in_storage(f_in):
    with open(f_in) as f:
        f.read(16)
        return 0

def set_time_in_memory(ln, duration):
    d_cmds[ln].append(duration)

def get_duration(s_in):
    s = s_in.split('Elapsed: ', 1)[1]
    #print("debug substr" + s)
    idx = s.find(' ms')
    #print("debug duration: " + s[:idx-1])
    return float(s[:idx])

def write_file(f_out, cmd):
    with open(f_out, "a") as f:
        ln = cmd + ',' + str(d_cmds[cmd][0]) + ',' + str(d_cmds[cmd][1]) + '\n'
        f.write(ln)

def main():
    with open(sys.argv[1]) as f_o:
        for ln_o in f_o:
            #print(ln_o)
            idx = ln_o.find(cmd_line)
            if idx != -1:
                line = ln_o[idx:]
                line = line.replace('\n', '')
                ret = find_in_memory(line)
                if ret == 1:
                    increase_in_memory(line)
                    continue

                add_in_memory(line)

        n = 0
        time = datetime.datetime.now()
        time_rocprof = time.strftime("%2m%2d")
        time_output = time.now().strftime("%2m%2d%H%M")
        output_file = storage_file + '_' + time_output + '.csv'
        for cmd in d_cmds:
            #print(cmd, d_cmds[cmd])
            cmd = cmd.replace('\r\n', '')
            rpl_file = 'output_' + time_rocprof  + '_' + str('{0:04d}'.format(n)) + '.csv'
            rpl_cmd = 'rocprof --stats -o ' + rpl_file + ' ' + cmd
            print(rpl_cmd)
            #os.system(rpl_cmd)
            output = subprocess.getoutput(rpl_cmd)
            #print("debug result: " + result)
            duration = get_duration(output)
            #print("debug duration: " + str(duration))
            set_time_in_memory(cmd, duration)
            #print("debug list: " + str(d_cmds[cmd]))
            write_file(output_file, cmd)
            n = n + 1
            #if n >= 10:
            #    break

if __name__ == "__main__":
    main()
