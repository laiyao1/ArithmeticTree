import os
import time
import subprocess
import selectors
import io
import sys
import argparse

parser = argparse.ArgumentParser(description='multiplier design')
parser.add_argument('--input_bit', type = int, default=16)
parser.add_argument('--area_w', type = float, default=0.01)
args = parser.parse_args()


def capture_subprocess_output(subprocess_args):
    process = subprocess.Popen(subprocess_args,
                               bufsize=1,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               universal_newlines=True)
    buf = io.StringIO()
    def handle_output(stream, mask):
        line = stream.readline()
        buf.write(line)
        sys.stdout.write(line)
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, handle_output)
    while process.poll() is None:
        events = selector.select()
        for key, mask in events:
            callback = key.data
            callback(key.fileobj, mask)
    return_code = process.wait()
    selector.close()
    success = (return_code == 0)
    output = buf.getvalue()
    buf.close()
    return (success, output)


def get_best_file_from_ppo(strftime, input_bit):
    file_name = "mult_logs/mult_{}b_{}.log".format(input_bit, strftime)
    print("file_name", file_name)
    assert os.path.exists(file_name)
    fopen = open(file_name, "r")
    verilog_file_name = None
    best_score = 1e8
    for line in fopen.readlines():
        raw_line = line.strip()
        line = line.strip().split("\t")
        score = float(line[1]) + float(line[2]) * args.area_w + float(line[3]) + float(line[4]) * args.area_w
        if score < best_score:
            best_score = score
            verilog_file_name = line[0]
            data_line = raw_line
    return verilog_file_name, data_line


def save_mult_file(verilog_file_name, strftime):
    template_mult_name = "mult_template_{}.v".format(strftime)
    fwrite = open(os.path.join("multiplier_template", template_mult_name), "w")
    fopen = open(os.path.join("run_verilog_mult_mid",verilog_file_name), "r")
    for line in fopen.readlines():
        if not line.startswith("module adder(a,b,s);"):
            fwrite.write(line)
        else:
            break
    fopen.close()
    fwrite.close()
    return template_mult_name


def get_best_file_from_mcts(strftime, input_bit):
    file_name = "mcts_mult_adder/mcts_mult_adder_{}b_openroad_{}.log".format(input_bit, strftime)
    assert os.path.exists(file_name)
    fopen = open(file_name, "r")
    verilog_file_name = None
    best_score = 1e8
    for line in fopen.readlines():
        raw_line = line.strip()
        line = line.strip().split("\t")
        score = float(line[1]) + float(line[2]) * args.area_w + float(line[3]) + float(line[4]) * args.area_w
        if score < best_score:
            best_score = score
            verilog_file_name = line[0]
            data_line = raw_line
    return verilog_file_name, data_line


def save_adder_file(verilog_file_name, strftime, input_bit):
    template_adder_name = "adder_template_{}.v".format(strftime)
    fwrite = open(os.path.join("adder_template", template_adder_name), "w")
    fopen = open(os.path.join("run_verilog_mult_add_mid",verilog_file_name), "r")
    cnt = 0 
    find_adder = False
    for line in fopen.readlines():
        if not line.startswith("//"):
            if line.startswith("module adder(a,b,s);"):
                find_adder = True
            if find_adder:
                fwrite.write(line)
            else:
                continue
        else:
            if "2" in line:
                continue
            else:
                if cnt < input_bit:
                    fwrite.write(line)
                    cnt += 1
    fwrite.close()
    fopen.close()
    return template_adder_name
                

def main():
    input_bit = args.input_bit
    each_iter_ppo = 900
    each_iter_mcts = 100
    total_times = 3
    strftime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    if not os.path.exists("back_and_forth"):
        os.mkdir("back_and_forth")
    flog = open("back_and_forth/bandf_{}b_{}_{:.2f}.log".format(input_bit, strftime, args.area_w), "w")
    start_time = time.time()
    for i in range(total_times):
        flog.write("time {}\n".format(time.time()-start_time))
        flog.flush()
        if i == 0:
            template_adder_name = ""
            capture_subprocess_output(["python", "PPO2_mult.py", "--input_bit={}".format(input_bit),
                "--max_iter={}".format(each_iter_ppo), "--strftime={}".format(strftime+"-{}".format(i))])
        else:
            capture_subprocess_output(["python", "PPO2_mult.py", "--input_bit={}".format(input_bit),
                "--max_iter={}".format(each_iter_ppo), "--template={}".format(template_adder_name), "--strftime={}".format(strftime+"-{}".format(i))])
        verilog_file_name, data_line = get_best_file_from_ppo(strftime+ "-{}".format(i), input_bit)
        flog.write("PPO:\t"+data_line+"\n")
        flog.flush()
        template_mult_name = save_mult_file(verilog_file_name, strftime + "-{}".format(i))
        if i == 0:
            capture_subprocess_output(["python", "MCTS_mult.py", "--input_bit={}".format(input_bit * 2),
                "--max_iter={}".format(each_iter_mcts), 
                "--template={}".format(template_mult_name), "--strftime={}".format(strftime+"-{}".format(i)),
                "--area_w={:.2f}".format(args.area_w)])
        else:
            capture_subprocess_output(["python", "MCTS_mult.py", "--input_bit={}".format(input_bit * 2),
                "--max_iter={}".format(each_iter_mcts), "--template={}".format(template_mult_name), 
                "--init_state", "--strftime={}".format(strftime+"-{}".format(i)),
                "--area_w={:.2f}".format(args.area_w)])        
        verilog_file_name, data_line = get_best_file_from_mcts(strftime+"-{}".format(i), input_bit * 2)
        flog.write("MCTS\t"+data_line+"\n")
        flog.flush()
        template_adder_name = save_adder_file(verilog_file_name, strftime + "-{}".format(i), input_bit * 2)
        i += 1
    flog.write("time {}\n".format(time.time()-start_time))
    flog.flush()


if __name__ == "__main__":
    main()