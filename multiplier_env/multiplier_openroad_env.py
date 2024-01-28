import gym
import numpy as np
import math
import re
import bisect
import subprocess
import os
import hashlib
import shutil
import copy
import time

def comp_fa_delay(a, b, cin):
    sum_delay = max(max(a,b)+1, cin) + 1
    cout_delay = max(max(a,b)+1, cin) + 2
    return sum_delay, cout_delay
 

def comp_ha_delay(a, b):
    sum_delay = max(a, b) + 1
    cout_delay = max(a, b) + 1
    return sum_delay, cout_delay

HA_ADDER = '''module HA(a,b,c,s);
input a,b;
output c,s;
xor x1(s,a,b);
and a1(c,a,b);
endmodule'''

FA_ADDER = '''module FA(a,b,c,cy,sm);
input a,b,c;
output cy,sm;
wire x,y,z;
HA h1(a,b,x,z);
HA h2(z,c,y,sm);
or o1(cy,x,y);
endmodule
'''

DEFAULT_ADDER = '''module adder(a,b,s);
input [{0}:0] a,b;
output [{0}:0] s;
assign s = a+b;
endmodule
'''

yosys_script_format = \
'''read -sv {}
synth -top main
flatten
opt
abc -fast -liberty NangateOpenCellLibrary_typical.lib -constr abc_constr -D 5000
write_verilog {}
'''

yosys_script_wo_flatten_format = \
'''read -sv {}
synth -top main
flatten
opt
abc -fast -liberty NangateOpenCellLibrary_typical.lib -constr abc_constr -D 50
write_verilog {}
'''

openroad_tcl = \
'''read_lef "Nangate45/Nangate45_tech.lef"
read_lef "Nangate45/Nangate45_stdcell.lef"
read_lib "Nangate45/Nangate45_typ.lib"
read_verilog {}
link_design "main"
set_max_delay -from [all_inputs] 0
set critical_path [lindex [find_timing_paths -sort_by_slack] 0]
set path_delay [sta::format_time [[$critical_path path] arrival] 4]
puts "wns $path_delay"
report_design_area
exit
'''

result_cache = {}
cache_hit = 0


class MultiplierEnv(gym.Env):

    def __init__(self, input_bit, template = None):
        self.input_bit = input_bit
        self.output_bit = 2 * input_bit
        self.input_bit_log = input_bit * int(math.log2(input_bit))
        self.dot_num = np.zeros((self.output_bit), dtype = np.int32)
        self.MAX_DELAY = 1e5
        self.fa_cost = 9
        self.ha_cost = 4
        self.action_seq = []
        self.tmp_verilog_str = ""
        self.reset()
        self.fopen = open("log_dot.log", 'w')
        self.verilog_file = open("multiplier.v", 'w')
        if template is not None:
            self.template_adder = open(template, 'r').read()
        else:
            self.template_adder = None

    def extract_results(self, stats):
        print("stats", stats)
        raw_line = stats.decode("utf-8").split('\n')[-40:]
        for i in range(len(raw_line)):
            if raw_line[i].startswith("ABC: WireLoad = \"none\"  Gates ="):
                line = raw_line[i].strip()
        ob = re.search(r'Delay *= *[1-9]+.?[0-9]*', line)
        delay = float(ob.group().split('=')[1].strip())
        ob = re.search(r'Area *= *[1-9]+.?[0-9]*', line)
        area = float(ob.group().split('=')[1].strip())
        return delay, area

    def get_reward(self):
        self.output_verilog()
        self.run_yosys()
        d, a, dw, aw = self.run_openroad()
        print("OPENROAD: delay = {:.2f}, area = {:.2f}".format(d, a))
        reward = -(d + dw)/1000.0
        return reward, d, a, dw, aw

    def reset(self):
        print("====in RESET====")
        self.verilog_file = open("multiplier.v", 'w')
        self.max_delay = 1
        self.all_area = self.input_bit ** 2 # 0
        self.fa = 0
        self.ha = 0
        self.and_gate = 0
        self.pin_num = 0
        self.error = False
        self.wo_error = False
        self.dot_num = np.zeros((self.output_bit), dtype = np.int32)
        self.this_digit_ha = np.zeros((self.output_bit), dtype = np.int32)
        self.action_seq.clear()
        self.tmp_verilog_str = ""
        self.adder_list_for_easymac = []

        for i in range(self.input_bit):
            self.dot_num[i] = i+1 
            self.dot_num[self.output_bit-2-i] = i+1 

        self.delay_map = []
        for i in range(2*self.input_bit):
            self.delay_map.append([])
        
        for i in range(self.input_bit):
            for j in range(self.input_bit):
                self.delay_map[i+j].append((1, "ip_{}_{}".format(i, j)))
        print("self.delay_map", self.delay_map)
        self.now_digit = 2
        self.max_delay = 1
        next_digit = 2
        assert self.dot_num[next_digit] == 3
        mask = [0, 0]
        self.next_delay_map = np.zeros(3)
        if self.dot_num[next_digit] >= 3:
            self.next_delay_map[0], self.next_delay_map[1], self.next_delay_map[2] = \
                self.delay_map[next_digit][0][0], self.delay_map[next_digit][1][0], self.delay_map[next_digit][2][0]
        elif self.dot_num[next_digit] == 2:
            self.next_delay_map[0], self.next_delay_map[1] = \
                self.delay_map[next_digit][0][0], self.delay_map[next_digit][1][0]
        elif self.dot_num[next_digit] == 1:
            self.next_delay_map[0] = self.delay_map[next_digit][0][0]
        else:
            pass

        self.state = np.concatenate((np.array([next_digit/self.input_bit, 
            self.max_delay / self.input_bit_log,
            self.this_digit_ha[next_digit]/self.input_bit]),
            mask,
            self.next_delay_map / self.input_bit_log
            ))

        return self.state

    def step(self, action, action_digit = None):
        if action_digit is None:
            action_digit = self.now_digit
        
        if action == 0:
            self.adder_list_for_easymac.append((action_digit, 1))
        elif action == 1:
            self.adder_list_for_easymac.append((action_digit, 0))
        
        action_type = action
        old_delay = self.max_delay
        old_area = self.all_area
        self.action_seq.append(action)

        if action_type == 0:
            assert self.dot_num[action_digit] >= 3
        elif action_type == 1:
            assert self.dot_num[action_digit] >= 2
        
        if action_type == 0 or action_type == 1:
            new_pin_name_c = "p{}".format(self.pin_num)
            self.pin_num += 1
            
            new_pin_name_s = "p{}".format(self.pin_num)
            self.pin_num += 1

        if action_type == 0:
            a = self.delay_map[action_digit][0][0]
            b = self.delay_map[action_digit][1][0]
            cin = self.delay_map[action_digit][2][0]
            sum_delay, cout_delay = comp_fa_delay(a, b, cin)
            
            self.dot_num[action_digit] -= 2

            self.tmp_verilog_str += "FA fa{}({},{},{},{},{});\n".format(
                self.fa,
                self.delay_map[action_digit][0][1],
                self.delay_map[action_digit][1][1], self.delay_map[action_digit][2][1],
                new_pin_name_c, new_pin_name_s
            )
            self.fa += 1
            self.delay_map[action_digit].pop(0)
            self.delay_map[action_digit].pop(0)
            self.delay_map[action_digit].pop(0)
            
        elif action_type == 1:
            a = self.delay_map[action_digit][0][0]
            b = self.delay_map[action_digit][1][0]
            sum_delay, cout_delay = comp_ha_delay(a, b)
            self.dot_num[action_digit] -= 1
            self.this_digit_ha[action_digit] += 1
            self.tmp_verilog_str += "HA ha{}({},{},{},{});\n".format(
                self.ha,
                self.delay_map[action_digit][0][1],
                self.delay_map[action_digit][1][1], 
                new_pin_name_c, new_pin_name_s
            )
            self.ha += 1
            self.delay_map[action_digit].pop(0)
            self.delay_map[action_digit].pop(0)
        else:
            assert False

        if action_type == 0 or action_type == 1:
            x = (sum_delay, new_pin_name_s)
            bisect.insort_left(self.delay_map[action_digit], x)
            x = (cout_delay, new_pin_name_c)
            bisect.insort_left(self.delay_map[action_digit+1], x)
            if action_digit+1 < self.output_bit:
                self.dot_num[action_digit+1] += 1

        if action_type == 0 or action_type == 1:
            self.max_delay = max(self.max_delay, max(sum_delay, cout_delay))
            
        if action_type == 0:
            self.all_area += self.fa_cost
        elif action_type == 1:
            self.all_area += self.ha_cost

        if self.dot_num[action_digit] <= 2: # 1:
            next_digit = action_digit + 1
        else:
            next_digit = action_digit
        
        self.now_digit = next_digit
        if next_digit >= self.output_bit-1:
            done = True
            print("dot_num")
            print(self.dot_num)
        else:
            done = False
        
        reward = -0.1 if action == 1 else 0
        expected_delay = 0
        if not done:
            if self.dot_num[next_digit] >= 3:
                mask = np.array([0, 0])
            else:
                mask = np.array([1, 0])
            
            delay = 0
            area = 0
            delay_wo = 0
            area_wo = 0
        else:
            
            for i in range(self.output_bit):
                expected_delay = max(expected_delay, self.delay_map[i][0][0])

            reward, delay, area, delay_wo, area_wo = self.get_reward()
            mask = np.array([1, 1])

        self.next_delay_map = np.zeros(3)
        if not done:
            if self.dot_num[next_digit] >= 3:
                self.next_delay_map[0], self.next_delay_map[1], self.next_delay_map[2] = \
                    self.delay_map[next_digit][0][0], self.delay_map[next_digit][1][0], self.delay_map[next_digit][2][0]
            elif self.dot_num[next_digit] == 2:
                self.next_delay_map[0], self.next_delay_map[1] = \
                    self.delay_map[next_digit][0][0], self.delay_map[next_digit][1][0]
            elif self.dot_num[next_digit] == 1:
                self.next_delay_map[0] = self.delay_map[next_digit][0][0]
            else:
                assert False

        self.state = np.concatenate((np.array([next_digit/self.input_bit, 
            self.max_delay / self.input_bit_log,
            self.this_digit_ha[next_digit]/self.input_bit]),
            mask,
            self.next_delay_map / self.input_bit_log
            ))

        return self.state, done, reward, {"next_digit": next_digit, 
            "delay": delay, "area": area, "expected_delay": expected_delay,
            "delay_wo": delay_wo, "area_wo": area_wo}
    
    def get_represent_int(self):
        rep_int = 0
        print("self.action_seq")
        print(self.action_seq)
        for i in self.action_seq:
            rep_int = rep_int * 2 + i
        self.rep_int = rep_int
        return rep_int
    
    def output_verilog(self, file_name = None):
        verilog_start_time = time.time()
        self.tmp_verilog_str += "wire [{}:0] a,b;\n".format(self.input_bit * 2 - 1)
        self.tmp_verilog_str += "wire [{}:0] s;\n".format(self.input_bit *2 - 1)

        for i in range(1, self.input_bit * 2):
            if len(self.delay_map[i]) >= 2:
                self.tmp_verilog_str += "assign a[{}] = {};\n".format(i, self.delay_map[i][0][1])
                self.tmp_verilog_str += "assign b[{}] = {};\n".format(i, self.delay_map[i][1][1])
            elif len(self.delay_map[i]) == 1:
                self.tmp_verilog_str += "assign a[{}] = {};\n".format(i, self.delay_map[i][0][1])
                self.tmp_verilog_str += "assign b[{}] = 1'b0;\n".format(i)
            else:
                self.tmp_verilog_str += "assign a[{}] = 1'b0;\n".format(i)
                self.tmp_verilog_str += "assign b[{}] = 1'b0;\n".format(i)
        self.tmp_verilog_str += "assign a[0] = ip_0_0;\n"
        self.tmp_verilog_str += "assign b[0] = 1'b0;\n"

        test_verilog_str = copy.deepcopy(self.tmp_verilog_str)
        if len(self.delay_map[self.input_bit * 2 - 1]) > 2:
            self.tmp_verilog_str += "assign o[{}] = s[{}]".format(self.input_bit * 2 - 1, 
                self.input_bit * 2 - 1)
            for i in range(2, len(self.delay_map[self.input_bit * 2 - 1])):
                self.tmp_verilog_str += " & {}".format(self.delay_map[self.input_bit * 2 -1][i][1])
            self.tmp_verilog_str += ";\n"
        else:
            self.tmp_verilog_str += "assign o[{}] = s[{}];\n".format(self.input_bit * 2 - 1,
                self.input_bit * 2 - 1)
        
        for i in range(self.input_bit * 2 -1):
            self.tmp_verilog_str += "assign o[{}] = s[{}];\n".format(i, i)
        
        for i in range(self.input_bit * 2 -1):
            test_verilog_str += "assign o[{}] = a[{}] & b[{}];\n".format(i, i, i)
        self.tmp_verilog_str += "adder add(a,b,s);\n"
        if not os.path.exists("run_verilog_mult_mid"):
            os.mkdir("run_verilog_mult_mid")
        
        rep_int = self.get_represent_int()
        dot_str = ""
        for i in range(self.dot_num.shape[0]):
            dot_str += str(self.dot_num[i])
        hash_value = hashlib.md5((str(rep_int)+dot_str).encode()).hexdigest()
        if file_name is None:
            file_name = "run_verilog_mult_mid/multiplier_{}b_{}.v".format(self.input_bit, 
                hash_value)
        
        self.verilog_file_name = file_name.split("/")[-1]
        verilog_file = open(file_name, "w")
        head_str = "module main(x,y,o);\n"
        head_str += "input [{}:0] x,y;\noutput [{}:0] o;\n".format(self.input_bit-1, self.output_bit-1)
        head_str += "wire "

        for i in range(self.input_bit):
            for j in range(self.input_bit):
                if i < self.input_bit - 1 or j < self.input_bit - 1:
                    head_str += "ip_{}_{},".format(i, j)
                else:
                    head_str += "ip_{}_{};\n".format(i, j)        
        head_str += "wire "
        for i in range(self.pin_num):
            if i < self.pin_num-1:
                head_str += "p{},".format(i)
            else:
                head_str += "p{};\n".format(i)
        
        cnt = 0
        for i in range(self.input_bit):
            for j in range(self.input_bit):
                head_str += "and and{}(ip_{}_{},x[{}],y[{}]);\n".format(cnt,i,j,i,j)
                cnt += 1

        self.tmp_verilog_str = head_str + self.tmp_verilog_str
        self.tmp_verilog_str += "\n"
        self.tmp_verilog_str += "endmodule\n\n"
        self.tmp_verilog_str += HA_ADDER
        self.tmp_verilog_str += '\n'
        self.tmp_verilog_str += FA_ADDER
        self.tmp_verilog_str += '\n'

        verilog_file.write("// ")
        for i in range(self.input_bit * 2):
            verilog_file.write("{} ".format(self.dot_num[i]))
        verilog_file.write("\n\n")
        verilog_file.write(self.tmp_verilog_str)

        if self.template_adder is None:
            verilog_file.write(DEFAULT_ADDER.format(self.input_bit * 2 - 1))
        else:
            verilog_file.write(self.template_adder)
        verilog_file.close()

        self.verilog_time = time.time() - verilog_start_time
    
    def run_yosys(self):
        if not os.path.exists("run_yosys_mult_mid"):
            os.mkdir("run_yosys_mult_mid")
        dst_file_name = os.path.join("run_yosys_mult_mid", self.verilog_file_name.split(".")[0] + "_yosys.v")
        dst_file_name_wo_flatten = os.path.join("run_yosys_mult_mid", 
            self.verilog_file_name.split(".")[0] + "_wo_flatten_yosys.v")
        file_name_prefix = self.verilog_file_name.split(".")[0] + "_yosys"
        src_file_path = os.path.join("run_verilog_mult_mid", self.verilog_file_name)

        if not os.path.exists("run_yosys_mult_script"):
            os.mkdir("run_yosys_mult_script")
        yosys_script_file_name = os.path.join("run_yosys_mult_script", 
            "ly_{}.ys".format(file_name_prefix))
        fopen = open(yosys_script_file_name, "w")
        fopen.write(yosys_script_format.format(src_file_path, dst_file_name))
        fopen.close()
        
        try:
            _ = subprocess.check_output(["yosys {}".format(yosys_script_file_name)], shell= True)
        except:
            self.error = True

        fopen = open(yosys_script_file_name, "w")
        fopen.write(yosys_script_wo_flatten_format.format(src_file_path, dst_file_name_wo_flatten))
        print(yosys_script_wo_flatten_format.format(src_file_path, dst_file_name_wo_flatten))
        fopen.close()
        
        if not os.path.exists("run_yosys_mult_script/abc_constr"):
            fopen = open("run_yosys_mult_script/abc_constr", "w")
            fopen.write("set_driving_cell BUF_X1\n")
            fopen.write("set_load 10.0 [all_outputs]\n")
            fopen.close()
        try:
            yosys_start_time = time.time()
            _ = subprocess.check_output(["yosys {}".format(yosys_script_file_name)], shell= True)
            self.yosys_time = time.time() - yosys_start_time
        except:
            self.wo_error = True
            self.yosys_time = 0.0

    def run_openroad(self):
        global result_cache
        global cache_hit
        def substract_results(p):
            lines = p.split("\n")[-5:]
            for line in lines:
                print("line", line)
                if "wns" in line:
                    delay = float(line.strip().split()[-1])
                elif "area" in line:
                    area = float(line.strip().split()[2])
            return area, delay, note

        file_name_prefix = self.verilog_file_name.split(".")[0]
        hash_idx = file_name_prefix.split("_")[-1]
        if hash_idx in result_cache:
            delay = result_cache[hash_idx]["delay"]
            area = result_cache[hash_idx]["area"]
            delay_wo = result_cache[hash_idx]["delay_wo"]
            area_wo = result_cache[hash_idx]["area_wo"]
            cache_hit += 1
            self.delay = delay
            self.area = area
            return delay, area, delay_wo, area_wo
        verilog_file_path = "~/OpenROAD/test/mult_tmp_{}.v".format(file_name_prefix)
        
        if self.error == False:
            yosys_file_name = os.path.join("run_yosys_mult_mid", self.verilog_file_name.split(".")[0] + "_yosys.v")
            shutil.copyfile(yosys_file_name, verilog_file_path)
            fopen_tcl = open("~/OpenROAD/test/nangate45_mult_{}.tcl".format(file_name_prefix), "w")
            fopen_tcl.write(openroad_tcl.format("mult_tmp_{}.v".format(file_name_prefix)))
            print("file_name_prefix", file_name_prefix)
            fopen_tcl.close()
            
            output = subprocess.check_output(['openroad',
                "~/OpenROAD/test/nangate45_mult_{}.tcl".format(file_name_prefix)], 
                cwd="~/OpenROAD/test").decode('utf-8')
            note = None
            retry = 0
            area, delay, note = substract_results(output)
            while note is None and retry < 3:
                output = subprocess.check_output(['openroad',
                    "~/OpenROAD/test/nangate45_{}.tcl".format(file_name_prefix)], 
                    shell=True, cwd="~/OpenROAD/test").decode('utf-8')
                area, delay, note = substract_results(output)
                retry += 1
            os.remove(verilog_file_path)
            os.remove(yosys_file_name)
        
        if self.wo_error == False:
            verilog_file_path = "~/OpenROAD/test/mult_tmp_{}_wo_flatten.v".format(file_name_prefix)
            yosys_file_name = os.path.join("run_yosys_mult_mid", self.verilog_file_name.split(".")[0] + "_wo_flatten_yosys.v")
            shutil.copyfile(yosys_file_name, verilog_file_path)
            fopen_tcl = open("~/OpenROAD/test/nangate45_mult_{}.tcl".format(file_name_prefix), "w")
            fopen_tcl.write(openroad_tcl.format("mult_tmp_{}_wo_flatten.v".format(file_name_prefix)))
            fopen_tcl.close()
            output_wo_flatten = subprocess.check_output(['openroad',
                "~/OpenROAD/test/nangate45_mult_{}.tcl".format(file_name_prefix)],
                cwd="~/OpenROAD/test").decode('utf-8')
            area_wo, delay_wo, note = substract_results(output_wo_flatten)
        else:
            area_wo = 1e5
            delay_wo = 1e2

        delay *= 1000
        delay_wo *= 1000
        self.delay = delay
        self.area = area
        result_cache[hash_idx] = {"delay": delay, "area": area, "delay_wo": delay_wo, "area_wo": area_wo}
        return delay, area, delay_wo, area_wo


if __name__ == "__main__":
    env = MultiplierEnv(input_bit = 4)
    