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
import sys


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
hierarchy -top main
flatten
proc; techmap; opt;
abc -fast -liberty NangateOpenCellLibrary_typical.lib
write_verilog {}
'''

sdc_format = \
'''create_clock [get_ports clk] -name core_clock -period 3.0
set_all_input_output_delays
'''

openroad_tcl = \
'''source "helpers.tcl"
source "flow_helpers.tcl"
source "Nangate45/Nangate45.vars"
set design "adder"
set top_module "main"
set synth_verilog "{}"
set sdc_file "{}"
set die_area {{0 0 80 80}}
set core_area {{0 0 80 80}}
source -echo "fast_flow.tcl"
'''

start_time = {}
update_time = {}
output_time = {}
result_cache = {}
global_step = 0
cache_hit = 0
record_num = 0

parser = argparse.ArgumentParser(description='Adder MCTS for optimizing practical metrics (delay and area).')
parser.add_argument('--input_bit', type = int, default=8)
parser.add_argument('--adder_type', type = int, default = 0)
parser.add_argument('--step', type = int, default = 1666)
parser.add_argument('--openroad_path', type = str, default = '/home')
parser.add_argument('--save_verilog', action = 'store_true', default = False)

args = parser.parse_args()

initial_adder_type = args.adder_type
INPUT_BIT = args.input_bit
strftime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
if not os.path.exists("adder_parc_log"):
    os.mkdir("adder_parc_log")
if not os.path.exists("adder_parc_log/adder_{}b".format(INPUT_BIT)):
    os.mkdir("adder_parc_log/adder_{}b".format(INPUT_BIT))
flog = open("adder_parc_log/adder_{}b/adder_{}b_openroad_type{}_{}.log".format(INPUT_BIT, 
    INPUT_BIT, args.adder_type, strftime), "w")

start_time = time.time()

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
        self.fanout_map = np.zeros((self.input_bit, self.input_bit), dtype = np.int8)
        self.min_map = min_map
        self.reward = reward
        self.size = size
        self.delay = None
        self.area = None
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

        self.available_choice_list = []
        cnt = 0
        for i in range(INPUT_BIT):
            for j in range(INPUT_BIT):
                if self.prob[1, i, j] == 1:
                    self.available_choice_list.append(self.input_bit **2 +i* self.input_bit+j)
                    cnt += 1
        for i in range(INPUT_BIT):
            for j in range(INPUT_BIT):
                if self.prob[0, i, j] == 1:
                    self.available_choice_list.append(i* self.input_bit+j)
                    cnt += 1
        self.available_choice = cnt
        self.action = action
        self.step_num = step_num

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
    
    def output_cell_map(self):
        if not os.path.exists("run_verilog_mid"):
            os.mkdir("run_verilog_mid")
        fdot_save = open("run_verilog_mid/adder_{}b_{}_{}_{}.log".format(self.input_bit, 
                int(self.level_map.max()), int(self.cell_map.sum()-self.input_bit),
                self.hash_value), 'w')
        for i in range(self.input_bit):
            for j in range(self.input_bit):
                fdot_save.write("{}".format(str(int(self.cell_map[i, j]))))
            fdot_save.write("\n")
        fdot_save.write("\n")
        fdot_save.close()

    def output_verilog(self, file_name = None):
        if not os.path.exists("run_verilog_mid"):
            os.mkdir("run_verilog_mid")
        rep_int = self.get_represent_int()
        self.hash_value = hashlib.md5(str(rep_int).encode()).hexdigest()
        self.output_cell_map()
        if file_name is None:
            file_name = "run_verilog_mid/adder_{}b_{}_{}_{}.v".format(self.input_bit, 
                int(self.level_map.max()), int(self.cell_map.sum()-self.input_bit),
                self.hash_value)
        self.verilog_file_name = file_name.split("/")[-1]

        verilog_file = open(file_name, "w")
        verilog_file.write("module main(a,b,s,cout);\n")
        verilog_file.write("input [{}:0] a,b;\n".format(self.input_bit-1))
        verilog_file.write("output [{}:0] s;\n".format(self.input_bit-1))
        verilog_file.write("output cout;\n")
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
        verilog_file.write('assign cout = c{};\n'.format(self.input_bit-1))
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
        if not os.path.exists("run_yosys_mid"):
            os.mkdir("run_yosys_mid")
        dst_file_name = os.path.join("run_yosys_mid", self.verilog_file_name.split(".")[0] + "_yosys.v")
        file_name_prefix = self.verilog_file_name.split(".")[0] + "_yosys"
        if os.path.exists(dst_file_name):
            return
        src_file_path = os.path.join("run_verilog_mid", self.verilog_file_name)

        if not os.path.exists("run_yosys_script"):
            os.mkdir("run_yosys_script")
        yosys_script_file_name = os.path.join("run_yosys_script", 
            "{}.ys".format(file_name_prefix))
        fopen = open(yosys_script_file_name, "w")
        fopen.write(yosys_script_format.format(src_file_path, dst_file_name))
        fopen.close()
        _ = subprocess.check_output(["yosys {}".format(yosys_script_file_name)], shell= True)
        if not args.save_verilog:
            os.remove(src_file_path)
    
    def run_openroad(self):
        global result_cache
        global cache_hit
        def substract_results(p):
            lines = p.split("\n")[-15:]
            area = -100.0
            wslack = -100.0
            power = 0.0
            note = None
            for line in lines:
                if not line.startswith("result:") and not line.startswith("Total"):
                    continue
                print("line", line)
                if "design_area" in line:
                    area = float(line.split(" = ")[-1])
                elif "worst_slack" in line:
                    wslack = float(line.split(" = ")[-1])
                    note = lines
                elif "Total" in line:
                    power = float(line.split()[-2])

            return area, wslack, power, note

        file_name_prefix = self.verilog_file_name.split(".")[0]
        hash_idx = file_name_prefix.split("_")[-1]
        if hash_idx in result_cache:
            delay = result_cache[hash_idx]["delay"]
            area = result_cache[hash_idx]["area"]
            power = result_cache[hash_idx]["power"]
            cache_hit += 1
            self.delay = delay
            self.area = area
            self.power = power
            return delay, area, power
        verilog_file_path = "{}/OpenROAD/test/adder_tmp_{}.v".format(args.openroad_path, file_name_prefix)
        yosys_file_name = os.path.join("run_yosys_mid", self.verilog_file_name.split(".")[0] + "_yosys.v")
        shutil.copyfile(yosys_file_name, verilog_file_path)
        
        sdc_file_path = "{}/OpenROAD/test/adder_nangate45_{}.sdc".format(args.openroad_path, file_name_prefix)
        fopen_sdc = open(sdc_file_path, "w")
        fopen_sdc.write(sdc_format)
        fopen_sdc.close()
        fopen_tcl = open("{}/OpenROAD/test/adder_nangate45_{}.tcl".format(args.openroad_path, file_name_prefix), "w")
        fopen_tcl.write(openroad_tcl.format("adder_tmp_{}.v".format(file_name_prefix), 
            "adder_nangate45_{}.sdc".format(file_name_prefix)))
        fopen_tcl.close()
        
        command = "openroad {}/OpenROAD/test/adder_nangate45_{}.tcl".format(args.openroad_path, file_name_prefix)
        # print("COMMAND: {}".format(command))
        output = subprocess.check_output(['openroad',
            "{}/OpenROAD/test/adder_nangate45_{}.tcl".format(args.openroad_path, file_name_prefix)], 
            cwd="{}/OpenROAD/test".format(args.openroad_path)).decode('utf-8')
        note = None
        retry = 0
        area, wslack, power, note = substract_results(output)
        while note is None and retry < 3:
            output = subprocess.check_output(['openroad',
                "{}/OpenROAD/test/adder_nangate45_{}.tcl".format(args.openroad_path, file_name_prefix)], 
                shell=True, cwd="{}/OpenROAD/test".format(args.openroad_path)).decode('utf-8')
            area, wslack, power, note = substract_results(output)
            retry += 1
        if os.path.exists(yosys_file_name):
            os.remove(yosys_file_name)
        if os.path.exists("{}/OpenROAD/test/adder_nangate45_{}.tcl".format(args.openroad_path, 
                file_name_prefix)):
            os.remove("{}/OpenROAD/test/adder_nangate45_{}.tcl".format(args.openroad_path, file_name_prefix))
        if os.path.exists("{}/OpenROAD/test/adder_nangate45_{}.sdc".format(args.openroad_path, 
                file_name_prefix)):
            os.remove("{}/OpenROAD/test/adder_nangate45_{}.sdc".format(args.openroad_path, file_name_prefix))
        if os.path.exists("{}/OpenROAD/test/adder_tmp_{}.v".format(args.openroad_path, 
                file_name_prefix)):
            os.remove("{}/OpenROAD/test/adder_tmp_{}.v".format(args.openroad_path,file_name_prefix))
        delay = 3.0 - wslack
        delay *= 1000
        self.delay = delay
        self.area = area
        self.power = power
        result_cache[hash_idx] = {"delay": delay, "area": area, "power": power}
        return delay, area, power

    def update_available_choice(self):
        up_tri_mask = np.triu(np.ones((self.input_bit, self.input_bit), dtype = np.int8), 
            k = 1)
        self.prob = np.ones((2, self.input_bit, self.input_bit), dtype = np.int8)
        self.prob[0] = np.where(self.cell_map >= 1.0, 0, self.prob[0])
        self.prob[0] = np.where(up_tri_mask >= 1.0, 0, self.prob[0])
        self.prob[1] = np.where(self.min_map <= 0.0, 0, self.prob[1])
        self.prob[1] = np.where(up_tri_mask >= 1.0, 0, self.prob[1])

        self.available_choice_list = []
        cnt = 0

        for i in range(INPUT_BIT):
            for j in range(INPUT_BIT):
                if self.prob[1, i, j] == 1:
                    self.available_choice_list.append(self.input_bit **2 + i * self.input_bit+j)
                    cnt += 1
        self.available_choice = cnt

    def is_terminal(self):
        if self.available_choice == 0:
            return True
        return False

    def compute_reward(self):
        if initial_adder_type == 0:
            return - (self.area) 
        else:
            return - (self.delay + self.area)

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

    def update_fanout_map(self):
        self.fanout_map.fill(0)
        self.fanout_map[0, 0] = 0
        for x in range(1, self.input_bit):
            self.fanout_map[x, x] = 0
            last_y = x
            for y in range(x-1, -1, -1):
                if self.cell_map[x, y] == 1:
                    self.fanout_map[last_y-1, y] += 1
                    self.fanout_map[x, last_y] += 1
                    last_y = y

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

    def get_next_state_with_random_choice(self):
        global global_step
        global record_num
        try_step = 0
        min_metric = 1e10
        while self.available_choice > 0 and \
            ((initial_adder_type != 0 and try_step < 4) or \
            (initial_adder_type == 0 and try_step < 4) ):
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
                assert next_cell_map[x, y] == 0
                next_cell_map[x, y] = 1
                next_cell_map, next_min_map = self.legalize(next_cell_map, next_min_map)
            elif action_type == 1:
                assert self.min_map[x, y] == 1
                assert self.cell_map[x, y] == 1
                next_cell_map[x, y] = 0
                next_cell_map, next_min_map = self.legalize(next_cell_map, next_min_map)
            next_level_map = self.update_level_map(next_cell_map, next_level_map)
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
            delay, area, power = next_state.run_openroad()
            global_step += 1
            print("delay = {}, area = {}".format(delay, area))
            # print("self.delay = {}, self.area = {}".format(self.delay, self.area))
            next_state.delay = delay
            next_state.area = area
            next_state.power = power
            next_state.update_fanout_map()
            fanout = next_state.fanout_map.max()
            print("try_step = {}".format(try_step))
            try_step += 1
            flog.write("{}\t{:.2f}\t{:.2f}\t{}\t{}\t{}\t{}\t{}\t{}\t{:d}\t{:.2f}\n".format(
                        next_state.verilog_file_name.split(".")[0], 
                        next_state.delay, next_state.area, next_state.power, 
                        int(next_state.level), int(next_state.size), fanout,
                        global_step, cache_hit,
                        0, time.time() - start_time))
            record_num += 1
            flog.flush()
            print("record_num : {}/{}".format(record_num, args.step))
            if record_num >= args.step:
                sys.exit()
            if initial_adder_type == 0: 
                if next_state.area < min_metric:
                    best_next_state = copy.deepcopy(next_state)
                    min_metric = next_state.area
            else:
                if next_state.area + next_state.delay <= min_metric:
                    best_next_state = copy.deepcopy(next_state)
                    min_metric = next_state.area + next_state.delay
            find = True
            if initial_adder_type == 0:
                if next_state.area <= self.area:
                    pass
                else:
                    find = False
            if initial_adder_type == 1 or initial_adder_type == 2:
                if next_state.area + next_state.delay <= self.area + self.delay:
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
    eps = 0.8
    while node.get_state().is_terminal() == False:
        if node.is_all_expand() or (random.random() > eps and len(node.get_children()) >= 1):
            print("IS ALL EXPAND")
            node = best_child(node, True)
        else:
            node = expand(node)
            break
    return node


def default_policy(node):
    current_state = node.get_state()
    best_state_reward = current_state.compute_reward()
    step = 0
    while current_state.is_terminal() == False and \
        ((step < 16 and initial_adder_type == 0) or
            (step < 16 and initial_adder_type != 0)):
        current_state = current_state.get_next_state_with_random_choice()
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
            right = C * 10 *  math.sqrt(right)
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


def monte_carlo_tree_search(node):
    computation_budget = int(1e6)

    for i in range(computation_budget):

        node = tree_policy(node)
        reward = default_policy(node)
        node = backup(node, reward)

        assert node.parent is None

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


def get_brent_kung_init():

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
            0, 0, 0, 0)
    return state


def search_best_adder():
    if initial_adder_type == 0:
        init_state = get_normal_init()
    elif initial_adder_type == 1:
        init_state = get_sklansky_init()
    else:
        init_state = get_brent_kung_init()
    init_state.output_verilog()
    init_state.run_yosys()
    delay, area, power = init_state.run_openroad()
    print("delay = {}, area = {}".format(delay, area))
    init_node = Node()
    init_node.set_state(init_state)
    current_node = init_node
    monte_carlo_tree_search(current_node)


def recover_cell_map_from_cell_map_str(cell_map_str):
    assert len(cell_map_str) == INPUT_BIT ** 2
    cell_map = np.zeros((INPUT_BIT, INPUT_BIT))
    for i in range(INPUT_BIT):
        for j in range(INPUT_BIT):
            cell_map[i, j] = int(cell_map_str[i * INPUT_BIT + j])
    return cell_map


def main():
    search_best_adder()


if __name__ == "__main__":
    main()
