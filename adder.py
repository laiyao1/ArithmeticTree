#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import math
import random
import numpy as np
import argparse
import copy
import torch
import time
import os


best_result = {}
parser = argparse.ArgumentParser(description='Adder MCTS')
parser.add_argument('--input_bit', type = int, default=8)
parser.add_argument('--level_bound_delta', type = int, default=0)
parser.add_argument('--max_save', type = int, default = 10)
parser.add_argument('--seed', type = int, default = 1)
parser.add_argument('--demo', action = 'store_true', default = False)
args = parser.parse_args()
INPUT_BIT = args.input_bit
random_seed = args.seed
random.seed(random_seed)
np.random.seed(random_seed)
update = False
global_depth = 0
min_size = 0
best_cell_map = None
LEVEL_BOUND_DELTA = args.level_bound_delta

save_result = []
global_step = 0
global_start_time = time.time()
time_log = open("time_log_{}b_{}_{}.log".format(args.input_bit, LEVEL_BOUND_DELTA, args.seed), "w")


class State(object):
    def __init__(self, level, size, cell_map, level_map, min_map,
            step_num, action, reward):

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
        self.level_bound = int(math.log2(INPUT_BIT) + 1 + LEVEL_BOUND_DELTA)

        assert self.cell_map.sum() - self.input_bit == self.size

        up_tri_mask = np.triu(torch.ones(self.input_bit, self.input_bit), 
            k = 1)
        self.prob = np.ones((2, self.input_bit, self.input_bit))
        self.prob[0] = np.where(self.cell_map >= 1.0, 0, self.prob[0])
        self.prob[0] = np.where(up_tri_mask >= 1.0, 0, self.prob[0])
        self.prob[1] = np.where(self.min_map <= 0.0, 0, self.prob[1])
        self.prob[1] = np.where(up_tri_mask >= 1.0, 0, self.prob[1])

        self.rep_int = 0 

        self.available_choice = int(self.prob[1].sum())
        self.available_choice_list = [0] * self.available_choice

        cnt = 0
        for i in range(INPUT_BIT):
            for j in range(INPUT_BIT):
                if self.prob[1, i, j] == 1:
                    self.available_choice_list[cnt] = self.input_bit **2 +i* self.input_bit+j
                    cnt += 1

        self.action = action
        self.step_num = step_num
    
    def init_idx(self):
        self.idx["cell_map"] = {}
        self.idx["min_map"] = {}
        self.idx["level_map"] = {}  
        for i in range(self.input_bit):
            self.idx["cell_map"][i] = []
            self.idx["min_map"][i] = []
            self.idx["level_map"][i] = []
        
        for i in range(self.input_bit):
            for j in range(self.input_bit):
                if self.cell_map[i,j] == 1:
                    self.idx["cell_map"][i].append(j)
                if self.min_map[i,j] == 1:
                    self.idx["min_map"][i].append(j)
                if self.level_map[i,j] > 0:
                    self.idx["level_map"][i].append((j, self.level_map[i,j]))

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

    def save_cell_map(self):

        if not os.path.exists("cell_map"):
            os.mkdir("cell_map")
        if not os.path.exists("cell_map/adder_{}b".format(self.input_bit)):
            os.mkdir("cell_map/adder_{}b".format(self.input_bit))
        if os.path.exists("cell_map/adder_{}b/adder_{}b_{}l_{}s_{}.log".format(self.input_bit,
            self.input_bit, int(self.level), int(self.size), int(args.max_save)-1)):
            return
        i = 0
        while os.path.exists("cell_map/adder_{}b/adder_{}b_{}l_{}s_{}.log".format(self.input_bit,
            self.input_bit, int(self.level), int(self.size), i)):
            i += 1
        fdot_save = open("cell_map/adder_{}b/adder_{}b_{}l_{}s_{}.log".format(self.input_bit, 
            self.input_bit, int(self.level), int(self.size), i), 'w')
        for i in range(self.input_bit):
            for j in range(self.input_bit):
                fdot_save.write("{} ".format(str(int(self.cell_map[i, j]))))
            fdot_save.write("\n")
        fdot_save.write("\n")
        fdot_save.close()

    def update_available_choice(self):
        up_tri_mask = np.triu(torch.ones(self.input_bit, self.input_bit), 
            k = 1)
        self.prob = np.ones((2, self.input_bit, self.input_bit))
        self.prob[0] = np.where(self.cell_map >= 1.0, 0, self.prob[0])
        self.prob[0] = np.where(up_tri_mask >= 1.0, 0, self.prob[0])
        self.prob[1] = np.where(self.min_map <= 0.0, 0, self.prob[1])
        self.prob[1] = np.where(up_tri_mask >= 1.0, 0, self.prob[1])
        self.available_choice = int(self.prob[1].sum())
        self.available_choice_list = [0] * self.available_choice
        cnt = 0
        for i in range(INPUT_BIT):
            for j in range(INPUT_BIT):
                if self.prob[1, i, j] == 1:
                    self.available_choice_list[cnt] = self.input_bit **2 +i* self.input_bit+j
                    cnt += 1

    def is_terminal(self):
        return self.available_choice == 0

    def compute_reward(self):
        return - self.size

    def legalize(self, cell_map, min_map, start_bit = 1):
        for i in range(self.input_bit):
            min_map[i, 0] = 0
            min_map[i, i] = 0
        x = start_bit
        activate_x_list = [start_bit]
        assert self.input_bit <= 256
        for x in range(self.input_bit-1, 0, -1):
            last_y = x
            for y in range(x-1, -1, -1):
                if cell_map[x, y] == 1:
                    if cell_map[last_y-1, y] == 0:
                        assert last_y - 1 <= start_bit
                        next_bit = last_y - 1
                        cell_map[last_y-1, y] = 1
                        activate_x_list.append(next_bit)
                    if min_map[last_y-1,y] == 1:
                        min_map[last_y-1, y] = 0
                    last_y = y
        return cell_map, min_map, activate_x_list
    
    def update_level_map(self, cell_map, level_map, start_bit = 1, activate_x_list = []):
        activate_x_list.reverse()
        min_x = min(activate_x_list)
        for x in range(min_x, self.input_bit):
            level_map[x, :] = 0
            level_map[x, x] = 1
            last_y = x
            for y in range(x-1, -1, -1):
                if cell_map[x, y] == 1:
                    level_map[x, y] = max(level_map[x, last_y], level_map[last_y-1, y]) + 1
                    last_y = y

        return level_map

    def get_next_state_with_random_choice(self, set_action = None, remove_action = None):
        while self.available_choice > 0:
            sample_prob = np.ones((self.available_choice))
            choice_idx = np.random.choice(self.available_choice, size = 1, replace=False, 
                    p = sample_prob/sample_prob.sum())[0]
            random_choice = self.available_choice_list[choice_idx]
            retry = 0
            while remove_action is not None and random_choice in remove_action and retry < 20:
                choice_idx = np.random.choice(self.available_choice, size = 1, replace=False, 
                    p = sample_prob/sample_prob.sum())[0]
                random_choice = self.available_choice_list[choice_idx]
                retry += 1
            action_type = random_choice // (self.input_bit ** 2)
            x = (random_choice % (self.input_bit ** 2)) // self.input_bit
            y = (random_choice % (self.input_bit ** 2)) % self.input_bit
            assert self.min_map[x, y] == 1
            next_cell_map = np.copy(self.cell_map)
            next_level_map = np.copy(self.level_map)
            next_min_map = np.copy(self.cell_map)

            assert action_type == 1
            assert self.min_map[x, y] == 1
            assert self.cell_map[x, y] == 1
            next_cell_map[x, y] = 0
            next_min_map[x, y] = 0
            next_cell_map, next_min_map, activate_x_list = self.legalize(next_cell_map, next_min_map, start_bit = x)
            next_level_map = self.update_level_map(next_cell_map, next_level_map, start_bit = x, activate_x_list = activate_x_list)
            next_level = next_level_map.max()
            next_size = next_cell_map.sum() - self.input_bit
            next_step_num = self.step_num + 1

            if (next_level <= self.level and next_size <= self.size) or \
                    (next_level < self.level and next_size <= self.size) or \
                     (next_level <= self.level_bound and next_size <= self.size):
                pass
            else:
                self.available_choice_list.remove(random_choice)
                self.available_choice -=1
                continue

            reward = -1 + INPUT_BIT * (next_level - self.level)
            action = random_choice
            next_state = State(next_level, next_size, next_cell_map,
                next_level_map, next_min_map, 
                next_step_num, action, reward)
            self.cumulative_choices.append(action)
            return next_state
        
        return None

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
        return len(self.children) == self.state.available_choice or self.state.available_choice == 0

    def add_child(self, sub_node):
        sub_node.set_parent(self)
        self.children.append(sub_node)

    def __repr__(self):
        return "Node: {}, Q/N: {}/{}, best: {}, state: {}".format(
            hash(self), self.quality_value, self.visit_times, self.best_reward, self.state)


def tree_policy(node):
    print("TREE_POLICY")
    while node.get_state().is_terminal() == False:
        if node.is_all_expand() or (random.random() > 0.8 and len(node.get_children()) >= 1):
            print("IS ALL EXPAND")
            node = best_child(node, True)
        else:
            print("EXPAND")
            node = expand(node)
            break
    return node


def default_policy(node):
    global update
    global min_size
    global global_step
    global best_cell_map
    print("DEFAULT_POLICY")
    current_state = node.get_state()
    global_step += 1
    if current_state.level not in best_result:
        update = True
        best_result[current_state.level] = current_state.size
    elif best_result[current_state.level] > current_state.size:
        update = True
        best_result[current_state.level] = current_state.size
    if min_size > current_state.size:
        best_cell_map = current_state.cell_map
        time_log.write("{}\t{}\t{:.2f}\t{}\n".format(current_state.level, 
            current_state.size, time.time()-global_start_time, global_step))
        time_log.flush()
    min_size = min(min_size, current_state.size)
    current_state.save_cell_map()

    best_state_reward = current_state.compute_reward()
    while current_state.is_terminal() == False:
        current_state = current_state.get_next_state_with_random_choice()
        if current_state is None:
            break 
        global_step += 1
        if min_size > current_state.size:
            best_cell_map = current_state.cell_map
            time_log.write("{}\t{}\t{:.2f}\t{}\n".format(current_state.level, 
                current_state.size, time.time()-global_start_time, global_step))
            time_log.flush()
        min_size = min(min_size, current_state.size)
        current_state.save_cell_map()
        if current_state.level not in best_result:
            update = True
            best_result[current_state.level] = current_state.size
        elif best_result[current_state.level] > current_state.size:
            update = True
            best_result[current_state.level] = current_state.size
        
        best_state_reward = max(best_state_reward, current_state.compute_reward())
    print("default policy finished")
    return best_state_reward


def expand(node):
    tried_sub_node_states = set(
        sub_node.get_state().action for sub_node in node.get_children()
    )
    print("available_choice = {}, untry_choice = {}".format(node.get_state().available_choice, node.get_state().available_choice - len(tried_sub_node_states)))
    
    if node.get_state().available_choice - len(tried_sub_node_states) <= 0:
        return node
    new_state = node.get_state().get_next_state_with_random_choice(remove_action = tried_sub_node_states)
    try_times = 0
    if new_state is None:
        return node
    while new_state.action in tried_sub_node_states and try_times < 20:
        new_state = node.get_state().get_next_state_with_random_choice(remove_action = tried_sub_node_states)
        try_times += 1
        if new_state is None:
            print("NO new state")
            return node

    sub_node = Node()
    sub_node.set_state(new_state)
    node.add_child(sub_node)

    return sub_node


def best_child(node, is_exploration):
    best_score = -sys.maxsize
    best_sub_node = None
    print("child len = {}".format(len(node.get_children())))
    for sub_node in node.get_children():
        if is_exploration:
            C = 1 / math.sqrt(2.0)
        else:
            C = 0.0
        if node.get_visit_times() >= 1e-2 and sub_node.get_visit_times() >= 1e-2:
            left = sub_node.get_best_reward() * 0.99 + sub_node.get_quality_value() / sub_node.get_visit_times() * 0.01
            right = math.log(node.get_visit_times()) / (sub_node.get_visit_times() + 1e-5)
            right = C * 10.0 * math.sqrt(right)
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


def best_size_child(node):
    best_size = node.state.size
    i = 0
    while len(node.children) >0: 
        print(" j = {}".format(i))
        i += 1
        print("node.state.size", node.state.size)
        best_next_node = None
        for sub_node in node.get_children():
            print("sub_node.state.size", sub_node.state.size)
            if best_size > sub_node.state.size:
                print("update")
                best_size = sub_node.state.size
                best_next_node = sub_node 
        if best_next_node is not None:
            node = best_next_node
            node.parent = None
        else:
            break
    return node


def monte_carlo_tree_search(node, computation_budget):
    global update
    global best_cell_map
    global min_size
    start_time = time.time()
    for i in range(computation_budget):
        print("i = {}/{}".format(i, computation_budget))
        start_time = time.time()
        node = tree_policy(node)
        reward = default_policy(node)
        node = backup(node, reward)
        assert node.parent is None
        print("best_size = {}, level = {}".format(min_size, int(math.log2(INPUT_BIT) + 1 + LEVEL_BOUND_DELTA)))

    print("node", node)
    best_next_node = best_child(node, False)
    return best_next_node


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
    
    for x in range(INPUT_BIT-1, 0, -1):
        last_y = x
        for y in range(x-1, -1, -1):
            if cell_map[x, y] == 1:
                min_map[last_y-1, y] = 0
                last_y = y
    
    level = level_map.max()
    size = cell_map.sum() - INPUT_BIT
    state = State(level, size, cell_map, level_map, min_map,
            0, 0, 0)
    state.update_available_choice()
    print("init_state cell_map sum", state.cell_map.sum())
    return state


def get_known_init(file_path, cell_map = None):

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
    
    
    if cell_map is None:
        fopen = open(file_path, "r")
        cell_map = np.zeros((INPUT_BIT, INPUT_BIT))
        i = 0
        for line in fopen.readlines():
            item_list = line.strip().split()
            print("item_list", item_list)
            for j, x in enumerate(item_list):
                cell_map[i, j] = int(x)
            i += 1
    
    level_map = np.zeros((INPUT_BIT, INPUT_BIT))
    level_map = update_level_map(cell_map, level_map)

    level = level_map.max()
    min_map = copy.deepcopy(cell_map)
    for i in range(INPUT_BIT):
        min_map[i, i] = 0
        min_map[i, 0] = 0
    
    for x in range(INPUT_BIT-1, 0, -1):
        last_y = x
        for y in range(x-1, -1, -1):
            if cell_map[x, y] == 1:
                min_map[last_y-1, y] = 0
                last_y = y
    
    for i in range(INPUT_BIT-1, -1, -1):
        last_y = i
        for j in range(i-1, -1, -1):
            if cell_map[i, j] == 1:
                assert cell_map[last_y-1, j] == 1
                last_y = j
    size = cell_map.sum() - INPUT_BIT
    print("READ level = {}, size = {}".format(level_map.max(), cell_map.sum()-INPUT_BIT))
    state = State(level, size, cell_map, level_map, min_map,
            0, 0, 0)
    print("cell_map sum = {}, min_map sum = {}".format(cell_map.sum(), min_map.sum()))
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
            0, 0, 0)
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
            0, 0, 0)
    return state


def search_best_adder():
    global best_cell_map
    file_path = None 
    if os.path.exists("cell_map/adder_{}b".format(INPUT_BIT)):
        dirs = os.listdir("cell_map/adder_{}b".format(INPUT_BIT))
        dirs.sort()
        dir_min_size = INPUT_BIT ** 2 // 2
        for d in dirs:
            size = int(d.split("_")[-2][:-1])
            level = int(d.split("_")[-3][:-1])
            if size <= dir_min_size and level == int(math.log2(INPUT_BIT) + 1 + LEVEL_BOUND_DELTA) - 1:
                dir_min_size = size
                file_path = os.path.join("cell_map/adder_{}b".format(INPUT_BIT), d)
    if args.demo and INPUT_BIT == 128 and LEVEL_BOUND_DELTA > 0 and LEVEL_BOUND_DELTA < 4:
        if LEVEL_BOUND_DELTA == 1:
            file_path = "demo_cell_map/adder_128b/adder_128b_8l_364s.log"
        elif LEVEL_BOUND_DELTA == 2:
            file_path = "demo_cell_map/adder_128b/adder_128b_10l_300s_0.log"
        elif LEVEL_BOUND_DELTA == 3:
            file_path = "demo_cell_map/adder_128b/adder_128b_10l_248s.log"
    
    if file_path is not None and LEVEL_BOUND_DELTA > 0:
        print("FROM_KNOWN_INIT_ADDER")
        print("file_path = {}".format(file_path))
        init_state = get_known_init(file_path)
    else:
        init_state = get_sklansky_init()
    global min_size
    min_size = init_state.size
    init_state.save_cell_map()
    if init_state.level not in best_result:
        best_result[init_state.level] = init_state.size
    if best_result[init_state.level] > init_state.size:
        best_result[init_state.level] = init_state.size

    init_node = Node()
    init_node.set_state(init_state)
    best_cell_map = init_state.cell_map
    current_node = init_node
    for i in range(10):
        print("global search = {}/10".format(i))
        current_node = monte_carlo_tree_search(current_node, 2000)
        current_node.parent = None


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