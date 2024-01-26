#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import math
import random
import numpy as np
import argparse
import copy
import time
import os
import hashlib
import subprocess
import shutil


BLACK_CELL = '''module BLACK(gik, pik, gkj, pkj, gij, pij);
input gik, pik, gkj, pkj;
output gij, pij;
assign pij = pik & pkj;
assign gij = gik | (pik & gkj);
endmodule
'''

GREY_CELL = '''module GREY(gik, pik, gkj, gij);
input gik, pik, gkj;
output gij;
assign gij = gik | (pik & gkj);
endmodule
'''

yosys_script_format = \
'''read -sv {}
synth -top main
flatten
opt
abc -constr ./abc_constr -fast -liberty NangateOpenCellLibrary_typical.lib -D 200000
write_verilog {}
'''

yosys_script_wo_flatten_format = \
'''read -sv {}
synth -top main
flatten
opt
abc -constr ./abc_constr -fast -liberty NangateOpenCellLibrary_typical.lib -D 50
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

best_result = {}
start_time = {}
update_time = {}
output_time = {}
result_cache = {}
global_step = 0
cache_hit = 0

parser = argparse.ArgumentParser(description='Adder MCTS')
parser.add_argument('--input_bit', type = int, default=8)
parser.add_argument('--template', type = str, default = None)
parser.add_argument('--initial_adder_type', type = int, default = 1)
parser.add_argument('--strftime', type = str, default = None)
parser.add_argument('--init_state', action = 'store_true', default = False)
parser.add_argument('--max_iter', type = int, default = 3000)
parser.add_argument('--area_w', type = float, default = 0.0)

args = parser.parse_args()
INPUT_BIT = args.input_bit
initial_adder_type = args.initial_adder_type

if args.strftime is not None:
    strftime = args.strftime
else:
    strftime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

if not os.path.exists("mcts_mult_adder"):
    os.mkdir("mcts_mult_adder")
if not os.path.exists("multiplier_template"):
    os.mkdir("multiplier_template")

flog = open("mcts_mult_adder/mcts_mult_adder_{}b_openroad_{}.log".format(INPUT_BIT, strftime), "w")

if args.template is not None:
    MULT_TEMPLATE = open(os.path.join("multiplier_template", args.template), "r").read()
else:
    pass

update = False
global_depth = 0
global_iter = 0
save_result = {}


class Adder(object):
    def __init__(self, cell_map, level_map):
        self.cell_map_str = ""
        for i in range(INPUT_BIT):
            for j in range(INPUT_BIT):
                self.cell_map_str += str(int(cell_map[i][j]))
        self.level = int(level_map.max())
        self.size = int(cell_map.sum() - INPUT_BIT)
    
    def __eq__(self, other):
        return self.cell_map_str == other.cell_map_str

    def __gt__(self, other):
        if self.level < other.level:
            return True
        elif (self.level == other.level) and (self.size < other.size):
            return True
        return False


class State(object):
    def __init__(self, level, size, cell_map, level_map, min_map,
            step_num, action, reward, level_bound_delta):
        self.current_value = 0.0
        self.current_round_index = 0
        self.input_bit = INPUT_BIT
        self.cumulative_choices = []
        self.level = level
        self.cell_map = cell_map
        self.level_map = level_map
        self.min_map = min_map
        self.reward = reward
        self.size = size
        self.delay = None
        self.area = None
        self.error = False
        self.wo_error = False
        self.level_bound_delta = level_bound_delta
        self.level_bound = int(math.log2(INPUT_BIT) + 1 + level_bound_delta)
        assert self.cell_map.sum() - self.input_bit == self.size

        up_tri_mask = np.triu(np.ones((self.input_bit, self.input_bit), dtype = np.int8),
            k = 1)
        self.prob = np.ones((2, self.input_bit, self.input_bit), dtype = np.int8)
        self.prob[0] = np.where(self.cell_map >= 1.0, 0, self.prob[0])
        self.prob[0] = np.where(up_tri_mask >= 1.0, 0, self.prob[0])
        self.prob[1] = np.where(self.min_map <= 0.0, 0, self.prob[1])
        self.prob[1] = np.where(up_tri_mask >= 1.0, 0, self.prob[1])

        self.available_choice = int(self.prob.sum())
        self.available_choice_list = [0] * self.available_choice # []
        cnt = 0

        for i in range(INPUT_BIT):
            for j in range(INPUT_BIT):
                if self.prob[1, i, j] == 1:
                    self.available_choice_list[cnt] = self.input_bit **2 +i* self.input_bit+j
                    cnt += 1
        for i in range(INPUT_BIT):
            for j in range(INPUT_BIT):
                if self.prob[0, i, j] == 1:
                    self.available_choice_list[cnt] = i* self.input_bit+j
                    cnt += 1
        self.action = action
        self.step_num = step_num
    
    def save_cell_map(self):
        strftime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        if not os.path.exists("cell_map_{}b_{}_{}".format(self.input_bit, int(self.level),
            int(self.size))):
            os.mkdir("cell_map_{}b_{}_{}".format(self.input_bit, int(self.level), int(self.size)))
        fdot_save = open("cell_map_{}b_{}_{}/cell_map_{}b_{}_{}_{}.log".format(self.input_bit, int(self.level),
            int(self.size), self.input_bit, int(self.level), int(self.size), strftime),'w')
        for i in range(self.input_bit):
            for j in range(self.input_bit):
                fdot_save.write("{} ".format(str(int(self.cell_map[i, j]))))
            fdot_save.write("\n")
        fdot_save.write("\n")
        for i in range(self.input_bit):
            for j in range(self.input_bit):
                fdot_save.write("{} ".format(str(int(self.level_map[i, j]))))
            fdot_save.write("\n")
        fdot_save.write("\n")
        fdot_save.close()

    def get_represent_int(self):
        rep_int = 0
        for i in range(1, self.input_bit):
            for j in range(i):
                if self.cell_map[i,j] == 1:
                    rep_int = rep_int * 2 + 1
                else:
                    rep_int *= 2
        self.rep_int = rep_int
        return rep_int

    def output_verilog(self, file_name = None):
        if not os.path.exists("run_verilog_mult_add_mid"):
            os.mkdir("run_verilog_mult_add_mid")
        rep_int = self.get_represent_int()

        dot_num = MULT_TEMPLATE.split("\n")[0].strip().split()[1:]
        dot_str = "".join(dot_num)
        for i in range(self.input_bit):
            dot_num[i] = int(dot_num[i])
        dot_num = np.array(dot_num)
        if dot_num.max() <= 1:
            dot_num = MULT_TEMPLATE.split("\n")[17].strip().split()[1:]
            dot_str = "".join(dot_num)
            for i in range(self.input_bit):
                dot_num[i] = int(dot_num[i])
            dot_num = np.array(dot_num)
        hash_value = hashlib.md5((str(rep_int) + dot_str).encode()).hexdigest()
        print("hash_value", hash_value)
        if file_name is None:
            file_name = "run_verilog_mult_add_mid/mult_{}b_{}_{}_{}.v".format(self.input_bit, 
                int(self.level_map.max()), int(self.cell_map.sum()-self.input_bit),
                hash_value)
        print("file_name", file_name)
        self.verilog_file_name = file_name.split("/")[-1]
        verilog_file = open(file_name, "w")

        for i in range(self.input_bit):
            verilog_file.write("// ")
            for j in range(self.input_bit):
                verilog_file.write("{:.0f} ".format(self.cell_map[i, j]))
            verilog_file.write("\n") 
        
        verilog_file.write("\n")

        verilog_file.write(MULT_TEMPLATE)
        verilog_file.write("\n\n")
        verilog_file.write("module adder(a,b,s);\n")
        verilog_file.write("input [{}:0] a,b;\n".format(self.input_bit-1))
        verilog_file.write("output [{}:0] s;\n".format(self.input_bit-1))
        wires = set()
        for i in range(self.input_bit):
            wires.add("c{}".format(i))
        
        for x in range(self.input_bit-1, 0, -1):
            last_y = x
            for y in range(x-1, -1, -1):
                if self.cell_map[x, y] == 1:
                    assert self.cell_map[last_y-1, y] == 1
                    if y==0:
                        wires.add("g{}_{}".format(x, last_y))
                        wires.add("p{}_{}".format(x, last_y))
                        wires.add("g{}_{}".format(last_y-1, y))
                    else:
                        wires.add("g{}_{}".format(x, last_y))
                        wires.add("p{}_{}".format(x, last_y))
                        wires.add("g{}_{}".format(last_y-1, y))
                        wires.add("p{}_{}".format(last_y-1, y))
                        wires.add("g{}_{}".format(x, y))
                        wires.add("p{}_{}".format(x, y))
                    last_y = y
        
        for i in range(self.input_bit):
            wires.add("p{}_{}".format(i, i))
            wires.add("g{}_{}".format(i, i))
            wires.add("c{}".format(x))
        assert 0 not in wires
        assert "0" not in wires
        verilog_file.write("wire ")
        
        for i, wire in enumerate(wires):
            if i < len(wires) - 1:
                    verilog_file.write("{},".format(wire))
            else:
                verilog_file.write("{};\n".format(wire))
        verilog_file.write("\n")
        
        for i in range(self.input_bit):
            verilog_file.write('assign p{}_{} = a[{}] ^ b[{}];\n'.format(i,i,i,i))
            verilog_file.write('assign g{}_{} = a[{}] & b[{}];\n'.format(i,i,i,i))
        
        for i in range(1, self.input_bit):
            verilog_file.write('assign g{}_0 = c{};\n'.format(i, i))
        
        for x in range(self.input_bit-1, 0, -1):
            last_y = x
            for y in range(x-1, -1, -1):
                if self.cell_map[x, y] == 1:
                    assert self.cell_map[last_y-1, y] == 1
                    if y == 0: # add grey module
                        verilog_file.write('GREY grey{}(g{}_{}, p{}_{}, g{}_{}, c{});\n'.format(
                            x, x, last_y, x, last_y, last_y-1, y, x
                        ))
                    else:
                        verilog_file.write('BLACK black{}_{}(g{}_{}, p{}_{}, g{}_{}, p{}_{}, g{}_{}, p{}_{});\n'.format(
                            x, y, x, last_y, x, last_y, last_y-1, y, last_y-1, y, x, y, x, y 
                        ))
                    last_y = y
        
        verilog_file.write('assign s[0] = a[0] ^ b[0];\n')
        verilog_file.write('assign c0 = g0_0;\n')
        for i in range(1, self.input_bit):
            verilog_file.write('assign s[{}] = p{}_{} ^ c{};\n'.format(i, i, i, i-1))
        verilog_file.write("endmodule")
        verilog_file.write("\n\n")

        verilog_file.write(GREY_CELL)
        verilog_file.write("\n")
        verilog_file.write(BLACK_CELL)
        verilog_file.write("\n")
        verilog_file.close()

    def run_yosys(self):
        if not os.path.exists("run_yosys_mult_add_mid"):
            os.mkdir("run_yosys_mult_add_mid")
        dst_file_name = os.path.join("run_yosys_mult_add_mid", self.verilog_file_name.split(".")[0] + "_yosys.v")
        dst_file_name_wo_flatten = os.path.join("run_yosys_mult_mid", 
            self.verilog_file_name.split(".")[0] + "_wo_flatten_yosys.v")
        
        file_name_prefix = self.verilog_file_name.split(".")[0] + "_yosys"
        src_file_path = os.path.join("run_verilog_mult_add_mid", self.verilog_file_name)

        if not os.path.exists("run_yosys_mult_add_script"):
            os.mkdir("run_yosys_mult_add_script")
        yosys_script_file_name = os.path.join("run_yosys_mult_add_script", 
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
        
        if not os.path.exists("run_yosys_mult_add_script/abc_constr"):
            fopen = open("run_yosys_mult_add_script/abc_constr", "w")
            fopen.write("set_driving_cell BUF_X1\n")
            fopen.write("set_load 10.0 [all_outputs]\n")
            fopen.close()
        try:
            _ = subprocess.check_output(["yosys {}".format(yosys_script_file_name)], shell= True)
        except:
            self.wo_error = True
    
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
            self.delay_wo = delay_wo
            self.area_wo = area_wo
            return delay, area, delay_wo, area_wo
        verilog_file_path = "~/OpenROAD/test/mult_adder_tmp_{}.v".format(file_name_prefix)
        
        if self.error == False:
            yosys_file_name = os.path.join("run_yosys_mult_add_mid", self.verilog_file_name.split(".")[0] + "_yosys.v")
            shutil.copyfile(yosys_file_name, verilog_file_path)
        fopen_tcl = open("~/OpenROAD/test/nangate45_mult_adder_{}.tcl".format(file_name_prefix), "w")
        fopen_tcl.write(openroad_tcl.format("mult_adder_tmp_{}.v".format(file_name_prefix)))
        fopen_tcl.close()
        
        output = subprocess.check_output(['openroad',
            "~/OpenROAD/test/nangate45_mult_adder_{}.tcl".format(file_name_prefix)], 
            cwd="~/OpenROAD/test").decode('utf-8')
        note = None
        retry = 0
        if self.error == False:
            area, delay, note = substract_results(output)
            while note is None and retry < 3:
                output = subprocess.check_output(['openroad',
                    "~/OpenROAD/test/nangate45_mult_adder_{}.tcl".format(file_name_prefix)], 
                    shell=True, cwd="~/OpenROAD/test").decode('utf-8')
                area, delay, note = substract_results(output)
                retry += 1
        else:
            area, delay = 1e5, 1e2

        verilog_file_path = "~/OpenROAD/test/mult_tmp_{}_wo_flatten.v".format(file_name_prefix)
        
        if self.wo_error == False:
            yosys_file_name = os.path.join("run_yosys_mult_mid", self.verilog_file_name.split(".")[0] + "_wo_flatten_yosys.v")
            shutil.copyfile(yosys_file_name, verilog_file_path)
        
            fopen_tcl = open("~/OpenROAD/test/nangate45_mult_{}.tcl".format(file_name_prefix), "w")
            fopen_tcl.write(openroad_tcl.format("mult_tmp_{}_wo_flatten.v".format(file_name_prefix)))
            fopen_tcl.close()
            output_wo_flatten = subprocess.check_output(['openroad',
                "~/OpenROAD/test/nangate45_mult_{}.tcl".format(file_name_prefix)],
                cwd="~/OpenROAD/test").decode('utf-8')
        
        if self.wo_error == False:
            area_wo, delay_wo, note = substract_results(output_wo_flatten)
        else:
            area_wo, delay_wo = 1e5, 1e2
        os.remove(verilog_file_path)
        os.remove(yosys_file_name)
        os.remove("~/OpenROAD/test/nangate45_mult_adder_{}.tcl".format(file_name_prefix))
        delay *= 1000
        delay_wo *= 1000
        self.delay = delay
        self.area = area
        self.delay_wo = delay_wo
        self.area_wo = area_wo
        result_cache[hash_idx] = {"delay": delay, "area": area, "delay_wo": delay_wo, "area_wo": area_wo}
        return delay, area, delay_wo, area_wo

    def update_available_choice(self):
        up_tri_mask = np.triu(np.ones((self.input_bit, self.input_bit), dtype = np.int8), 
            k = 1)
        self.prob = np.ones((2, self.input_bit, self.input_bit), dtype = np.int8)
        self.prob[0] = np.where(self.cell_map >= 1.0, 0, self.prob[0])
        self.prob[0] = np.where(up_tri_mask >= 1.0, 0, self.prob[0])
        self.prob[1] = np.where(self.min_map <= 0.0, 0, self.prob[1])
        self.prob[1] = np.where(up_tri_mask >= 1.0, 0, self.prob[1])

        self.available_choice = int(self.prob.sum())
        self.available_choice_list = [0] * self.available_choice # []
        cnt = 0

        for i in range(INPUT_BIT):
            for j in range(INPUT_BIT):
                if self.prob[1, i, j] == 1:
                    self.available_choice_list[cnt] = self.input_bit **2 + i * self.input_bit+j
                    cnt += 1

        for i in range(INPUT_BIT):
            for j in range(INPUT_BIT):
                if self.prob[0, i, j] == 1:
                    self.available_choice_list[cnt] = i* self.input_bit + j
                    cnt += 1

    def is_terminal(self):
        if self.available_choice == 0:
            return True
        return False

    def compute_reward(self):
        if self.area is None:
            self.output_verilog()
            self.run_yosys()
            self.run_openroad()
        return -(self.delay + self.delay_wo + self.area*args.area_w + self.area_wo*args.area_w) / 1000.0

    def legalize(self, cell_map, min_map):
        min_map = copy.deepcopy(cell_map)
        for i in range(self.input_bit):
            min_map[i, 0] = 0
            min_map[i, i] = 0
        for x in range(self.input_bit-1, 0, -1):
            last_y = x
            for y in range(x-1, -1, -1):
                if cell_map[x, y] == 1:
                    cell_map[last_y-1, y] = 1
                    min_map[last_y-1, y] = 0
                    last_y = y
        return cell_map, min_map

    def update_level_map(self, cell_map, level_map):
        level_map[1:].fill(0)
        level_map[0, 0] = 1
        for x in range(1, self.input_bit):
            level_map[x, x] = 1
            last_y = x
            for y in range(x-1, -1, -1):
                if cell_map[x, y] == 1:
                    level_map[x, y] = max(level_map[x, last_y], level_map[last_y-1, y])+ 1
                    last_y = y
        return level_map

    def get_next_state_with_random_choice(self, set_action = None):
        global global_step, global_iter
        try_step = 0
        min_score = 1e8
        while self.available_choice > 0 and (initial_adder_type == 1  or \
            (initial_adder_type == 0  and try_step < INPUT_BIT) ):
            sample_prob = np.ones((self.available_choice))
            choice_idx = np.random.choice(self.available_choice, size = 1, replace=False, 
                    p = sample_prob/sample_prob.sum())[0]
            random_choice = self.available_choice_list[choice_idx]
            action_type = random_choice // (self.input_bit ** 2)
            x = (random_choice % (self.input_bit ** 2)) // self.input_bit
            y = (random_choice % (self.input_bit ** 2)) % self.input_bit
            next_cell_map = copy.deepcopy(self.cell_map)
            next_min_map = np.zeros((INPUT_BIT, INPUT_BIT))
            next_level_map = np.zeros((INPUT_BIT, INPUT_BIT))

            if action_type == 0:
                next_cell_map[x, y] = 1
                next_cell_map, next_min_map = self.legalize(next_cell_map, next_min_map, start_bit = x)
            elif action_type == 1:
                assert self.min_map[x, y] == 1
                assert self.cell_map[x, y] == 1
                next_cell_map[x, y] = 0
                next_cell_map, next_min_map = self.legalize(next_cell_map, next_min_map, start_bit = x)
            next_level_map = self.update_level_map(next_cell_map, next_level_map, start_bit = x)
            next_level = next_level_map.max()
            next_size = next_cell_map.sum() - self.input_bit
            next_step_num = self.step_num + 1
            action = random_choice
            reward = 0
            next_state = State(next_level, next_size, next_cell_map,
                next_level_map, next_min_map, 
                next_step_num, action, reward, self.level_bound_delta)
            
            next_state.output_verilog()
            next_state.run_yosys()
            delay, area, delay_wo, area_wo = next_state.run_openroad()
            global_step += 1
            print("delay = {}, area = {}".format(delay, area))
            print("self.delay = {}, self.area = {}".format(self.delay, self.area))
            next_state.delay = delay
            next_state.area = area
            print("try_step = {}".format(try_step))
            try_step += 1
            flog.write("{}\t{:.2f}\t{:.0f}\t{:.2f}\t{:.0f}\t{}\t{}\t{}\t{}\n".format(next_state.verilog_file_name, 
                        next_state.delay, next_state.area, 
                        next_state.delay_wo, next_state.area_wo,
                        next_state.level, next_state.size,
                        global_step, cache_hit))
            flog.flush()
            global_iter += 1
            if global_iter >= args.max_iter:
                return
            if next_state.delay + next_state.delay_wo + next_state.area * args.area_w + next_state.area_wo * args.area_w <= min_score:
                best_next_state = copy.deepcopy(next_state)
                min_score = next_state.delay + next_state.delay_wo + next_state.area*args.area_w + next_state.area_wo * args.area_w
            find = True
            if initial_adder_type == 0:
                if next_state.delay + next_state.delay_wo + next_state.area*args.area_w + next_state.area_wo*args.area_w <= self.delay + self.delay_wo + self.area*args.area_w + self.area_wo*args.area_w:
                    pass
                else:
                    find = False
            elif initial_adder_type == 1:
                if next_state.delay + next_state.delay_wo + next_state.area*args.area_w + next_state.area_wo*args.area_w <= self.delay + self.delay_wo + self.area*args.area_w + self.area_wo*args.area_w:
                    pass
                else:
                    find = False
            if find is False:
                self.available_choice_list.remove(random_choice)
                self.available_choice -=1
                assert self.available_choice == len(self.available_choice_list)
                continue
            self.cumulative_choices.append(action)
            return next_state
        
        return best_next_state

    def __repr__(self):
        return "State: {}, level: {}, choices: {}".format(
            hash(self), self.level, 
            self.cumulative_choices)


class Node(object):
    def __init__(self):
        self.parent = None
        self.children = []
        self.visit_times = 0
        self.quality_value = 0.0
        self.best_reward = -sys.maxsize
        self.state = None

    def set_state(self, state):
        self.state = state

    def get_state(self):
        return self.state

    def get_parent(self):
        return self.parent

    def set_parent(self, parent):
        self.parent = parent

    def get_children(self):
        return self.children

    def get_visit_times(self):
        return self.visit_times

    def set_visit_times(self, times):
        self.visit_times = times

    def visit_times_add_one(self):
        self.visit_times += 1

    def get_quality_value(self):
        return self.quality_value

    def set_quality_value(self, value):
        self.quality_value = value

    def quality_value_add_n(self, n):
        self.quality_value += n
    
    def update_best_reward(self, n):
        self.best_reward = max(self.best_reward, n)
    
    def get_best_reward(self):
        return self.best_reward

    def is_all_expand(self):
        return len(self.children) == self.state.available_choice

    def add_child(self, sub_node):
        sub_node.set_parent(self)
        self.children.append(sub_node)

    def __repr__(self):
        return "Node: {}, Q/N: {}/{}, best: {}, state: {}".format(
            hash(self), self.quality_value, self.visit_times, self.best_reward, self.state)


def tree_policy(node):
    if global_iter >= args.max_iter:
        return None
    eps = 0.8
    while node.get_state().is_terminal() == False:
        if global_iter >= args.max_iter:
            return None
        if node.is_all_expand() or (random.random() > eps and len(node.get_children()) >= 1):
            print("IS ALL EXPAND")
            node = best_child(node, True)
        else:
            node = expand(node)
            break
    return node


def default_policy(node, level_bound_delta):
    global update
    current_state = node.get_state()
    best_state_reward = current_state.compute_reward()
    step = 0
    while current_state.is_terminal() == False and step < INPUT_BIT // 16:
        current_state = current_state.get_next_state_with_random_choice()
        if global_iter >= args.max_iter:
            return None
        if current_state is None:
            break 
        print("step = {}".format(step))
        step += 1
        best_state_reward = max(best_state_reward, current_state.compute_reward())
    print("default policy finished")
    return best_state_reward


def expand(node):
    tried_sub_node_states = [
        sub_node.get_state().action for sub_node in node.get_children()
    ]
    new_state = node.get_state().get_next_state_with_random_choice()
    while new_state.action in tried_sub_node_states:
        new_state = node.get_state().get_next_state_with_random_choice()
    sub_node = Node()
    sub_node.set_state(new_state)
    node.add_child(sub_node)
    return sub_node


def best_child(node, is_exploration):
    best_score = -sys.maxsize
    best_sub_node = None
    for sub_node in node.get_children():
        if is_exploration:
            C = 1 / math.sqrt(2.0)
        else:
            C = 0.0
        if node.get_visit_times() >= 1e-2 and sub_node.get_visit_times() >= 1e-2:
            left = sub_node.get_best_reward() * 0.99 + sub_node.get_quality_value() / sub_node.get_visit_times() * 0.01 
            right = math.log(node.get_visit_times()) / sub_node.get_visit_times()
            right = C * 10.0 * math.sqrt(right)
            print("left = {}, right = {}".format(left, right))
            score = left + right
        else:
            score = 1e9
        if score > best_score:
            best_sub_node = sub_node
            best_score = score
    return best_sub_node


def backup(node, reward):
    while node != None:
        node.visit_times_add_one()
        node.quality_value_add_n(reward)
        node.update_best_reward(reward)
        if node.parent is not None:
            node = node.parent
        else:
            break
    assert node is not None
    assert node.parent is None
    return node


def monte_carlo_tree_search(node, level_bound_delta):
    global update
    computation_budget = 5000000
    for i in range(computation_budget):
        node = tree_policy(node)
        if global_iter >= args.max_iter:
            return None
        reward = default_policy(node, level_bound_delta)
        
        if global_iter >= args.max_iter:
            return None
        node = backup(node, reward)
        if global_iter >= args.max_iter:
            return None
        assert node.parent is None
        print("=== best result ===")
        output_best_result = list(best_result.items())
        output_best_result.sort(key = lambda x: x[0])
        print(str(output_best_result))


def get_sklansky_init():
    cell_map = np.zeros((INPUT_BIT, INPUT_BIT))
    level_map = np.zeros((INPUT_BIT, INPUT_BIT))
    for i in range(INPUT_BIT):
        cell_map[i, i] = 1
        level_map[i, i] = 1
        t = i
        now = i
        x = 1
        level = 1
        while t > 0:
            if t % 2 ==1:
                last_now = now
                now -= x
                cell_map[i, now] = 1
                level_map[i, now] = max(level, level_map[last_now-1, now]) +1
                level += 1
            t = t // 2
            x *= 2
    min_map = copy.deepcopy(cell_map)
    for i in range(INPUT_BIT):
        min_map[i, i] = 0
        min_map[i, 0] = 0
    
    level = level_map.max()
    size = cell_map.sum() - INPUT_BIT
    state = State(level, size, cell_map, level_map, min_map,
            0, 0, 0, 0)
    state.cell_map, state.min_map = state.legalize(cell_map, min_map)
    state.update_available_choice()
    return state


def get_min_map_from_cell_map(cell_map):
    min_map = copy.deepcopy(cell_map)
    for i in range(INPUT_BIT):
        min_map[i, i] = 0
        min_map[i, 0] = 0
    return min_map

    
def read_init_state():
    cell_map = np.zeros((INPUT_BIT, INPUT_BIT))
    if args.template is None:
        fopen = open("multiplier_template/mult_template.v", "r")
    else:
        fopen = open("multiplier_template/{}".format(args.template), "r")
    print("file_name: {}".format("multiplier_template/{}".format(args.template)))
    i = 0
    for line in fopen.readlines():
        if i >= args.input_bit:
            break
        if not line.startswith("//"):
            continue
        line = line.strip().split(" ")[1:]
        for k in range(args.input_bit):
            line[k] = int(line[k])
        line = np.array(line)
        if line.max() > 1.5:
            continue
        assert len(line) == args.input_bit
        for j in range(args.input_bit):
            cell_map[i, j] = int(line[j])
        i += 1
    return cell_map


def get_normal_init():
    cell_map = np.zeros((INPUT_BIT, INPUT_BIT))
    level_map = np.zeros((INPUT_BIT, INPUT_BIT))
    for i in range(INPUT_BIT):
        cell_map[i, i] = 1
        cell_map[i, 0] = 1
        level_map[i, i] = 1
        level_map[i, 0] = i+1
    level = level_map.max()
    min_map = copy.deepcopy(cell_map)
    for i in range(INPUT_BIT):
        min_map[i, i] = 0
        min_map[i, 0] = 0
    size = cell_map.sum() - INPUT_BIT
    state = State(level, size, cell_map, level_map, min_map,
            0, 0, 0, 0)
    return state


def update_level_map(cell_map, level_map):
        level_map.fill(0)
        level_map[0, 0] = 1
        for x in range(1, INPUT_BIT):
            level_map[x, x] = 1
            last_y = x
            for y in range(x-1, -1, -1):
                if cell_map[x, y] == 1:
                    level_map[x, y] = max(level_map[x, last_y], level_map[last_y-1, y])+ 1
                    last_y = y
        return level_map


def get_brent_kung_init(level_bound_delta):
    cell_map = np.zeros((INPUT_BIT, INPUT_BIT))
    level_map = np.zeros((INPUT_BIT, INPUT_BIT))
    for i in range(INPUT_BIT):
        cell_map[i, i] = 1 
        cell_map[i, 0] = 1
    t = 2
    while t < INPUT_BIT:
        for i in range(t-1, INPUT_BIT, t):
            cell_map[i, i-t+1] = 1
        t *= 2
    level_map = update_level_map(cell_map, level_map)
    level = level_map.max()
    min_map = copy.deepcopy(cell_map)
    for i in range(INPUT_BIT):
        min_map[i, i] = 0
        min_map[i, 0] = 0
    size = cell_map.sum() - INPUT_BIT
    print("BK level ={}, size = {}".format(level_map.max(), cell_map.sum()-INPUT_BIT))
    state = State(level, size, cell_map, level_map, min_map,
            0, 0, 0, level_bound_delta)
    return state


def get_state_from_cell_map(cell_map):
    level_map = np.zeros((INPUT_BIT, INPUT_BIT))
    level_map = update_level_map(cell_map, level_map)
    level = level_map.max()
    min_map = copy.deepcopy(cell_map)
    for i in range(INPUT_BIT):
        min_map[i, i] = 0
        min_map[i, 0] = 0
    size = cell_map.sum() - INPUT_BIT
    state = State(level, size, cell_map, level_map, min_map,
            0, 0, 0, 0)
    return state


def search_best_adder(level_bound_delta):
    if level_bound_delta not in save_result:
        save_result[level_bound_delta] = {}
    if initial_adder_type == 0:
        init_state = get_sklansky_init()
    else:
        if args.init_state == False:
            init_state = get_brent_kung_init(level_bound_delta)
        else:
            init_cell_map = read_init_state()
            print("init_cell_map")
            print(init_cell_map)
            init_state = get_state_from_cell_map(init_cell_map)
    init_state.output_verilog()
    init_state.run_yosys()
    delay, area, delay_wo, area_wo = init_state.run_openroad()
    print("INITIAL delay = {}, area = {}, delay_wo = {}, area_wo = {}".format(delay, area,
        delay_wo, area_wo))
    flog.write("{}\t{:.2f}\t{:.0f}\t{:.2f}\t{:.0f}\t{}\t{}\t{}\t{}\n".format(init_state.verilog_file_name, 
        init_state.delay, init_state.area, 
        init_state.delay_wo, init_state.area_wo,
        init_state.level, init_state.size,
        -1, -1))
    init_node = Node()
    init_node.set_state(init_state)

    current_node = init_node
    monte_carlo_tree_search(current_node, level_bound_delta)


def recover_cell_map_from_cell_map_str(cell_map_str):
    assert len(cell_map_str) == INPUT_BIT ** 2
    cell_map = np.zeros((INPUT_BIT, INPUT_BIT))
    for i in range(INPUT_BIT):
        for j in range(INPUT_BIT):
            cell_map[i, j] = int(cell_map_str[i * INPUT_BIT + j])
    return cell_map
            

def output_data(level_bound_delta):
    if not os.path.exists("adder_{}b".format(INPUT_BIT)):
        os.mkdir("adder_{}b".format(INPUT_BIT))
    level = int(math.log2(INPUT_BIT) + 1 + level_bound_delta)
    if not os.path.exists("adder_{}b/adder_{}b_{}".format(INPUT_BIT, INPUT_BIT, level)):
        os.mkdir("adder_{}b/adder_{}b_{}".format(INPUT_BIT, INPUT_BIT, level))
    print("len save_result[{}] = {}".format(level_bound_delta, len(save_result[level_bound_delta])))
    for i in save_result[level_bound_delta]:
        size = i
        assert level == save_result[level_bound_delta][i].level
        cell_map = recover_cell_map_from_cell_map_str(save_result[level_bound_delta][i].cell_map_str)
        dirs = os.listdir("adder_{}b/adder_{}b_{}".format(
            INPUT_BIT, INPUT_BIT, level,
        ))
        num = 0
        for file_name in dirs:
            item = file_name.split(".")[0].split("_")
            item_size = int(item[-2])
            if size == item_size:
                num += 1
        if num >= 1:
            continue
        file_name = "adder_{}b/adder_{}b_{}/adder_{}b_{}_{}_{}.log".format(
            INPUT_BIT, INPUT_BIT, level, INPUT_BIT, level, int(size), num
        )
        fwrite = open(file_name, "w")
        for i in range(INPUT_BIT):
            for j in range(INPUT_BIT):
                fwrite.write("{} ".format(str(int(cell_map[i, j]))))
            fwrite.write("\n")
        fwrite.close()


def main():
    search_best_adder(0)
    

if __name__ == "__main__":
    main()