
import argparse
from collections import namedtuple

import os
import numpy as np

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import multiplier_env
import random
import time
from tqdm import tqdm
import shutil

# Parameters
parser = argparse.ArgumentParser(description='Solve the Pendulum-v0 with PPO')
parser.add_argument(
    '--gamma', type=float, default=0.8, metavar='G', help='discount factor (default: 0.9)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--input_bit', type = int, default=4)
parser.add_argument('--max_iter', type = int, default = 3000)
parser.add_argument('--strftime', type = str, default = None)
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    metavar='N',
    help='interval between training status logs (default: 10)')
parser.add_argument(
    '--template',
    type=str,
    default=None,
)
parser.add_argument('--use_easymac', action='store_true')
parser.add_argument('--lr', type=float, default=1e-3)
args = parser.parse_args()


if args.template is None:
    env = gym.make('multiplier-openroad-v0', input_bit = args.input_bit,
        template = None).unwrapped
else:
    env = gym.make('multiplier-openroad-v0', input_bit = args.input_bit,
        template = os.path.join("adder_template",args.template)).unwrapped

num_state = 3 + 2 + 3
torch.manual_seed(args.seed)
env.seed(args.seed)

# set device to cpu or cuda
device = torch.device('cpu')

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

Transition = namedtuple('Transition',['state', 'action', 'reward', 'a_log_prob', 'next_state', 'done'])
TrainRecord = namedtuple('TrainRecord',['episode', 'reward'])

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 64)
        self.fc2 = nn.Linear(64,16)
        self.fc3 = nn.Linear(16,2)
        self.softmax = nn.Softmax(dim=-1)
        self.temp = 2

    def forward(self, x):
        mask = x[:, 3: 5].reshape(-1, 2)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        x = x / self.temp
        x = torch.where(mask >= 1.0, -1.0e10, x.double())
        x = self.softmax(x)
        return x

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, 64)
        self.fc2 = nn.Linear(64, 8)
        self.state_value= nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.state_value(x)
        return value

class PPO():
    clip_param = 0.2
    max_grad_norm = 0.2
    ppo_epoch = 10
    buffer_capacity = args.input_bit * args.input_bit * 6
    batch_size = 64

    def __init__(self):
        super(PPO, self).__init__()
        self.actor_net = Actor().float().to(device)
        self.critic_net = Critic().float().to(device)
        self.buffer = []
        self.counter = 0
        self.training_step = 0

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), args.lr)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), args.lr)
        if not os.path.exists('../param'):
            os.makedirs('../param/net_param')
            os.makedirs('../param/img')

    def select_action(self, state, eps = 0.0):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action_probs = self.actor_net(state)
        dist = Categorical(action_probs)
        if random.random() <= eps and state[0, 3:5].sum() == 0:
            action = torch.randint(0, 2, (1,)).to(device)
        else:
            action = dist.sample()
        
        action_log_prob = dist.log_prob(action)

        return action.item(), action_log_prob.item()


    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def save_param(self):
        torch.save(self.actor_net.state_dict(), '../param/net_param/actor_net_{}.pkl'.format(str(time.time())[:10]))
        torch.save(self.critic_net.state_dict(), '../param/net_param/critic_net_{}.pkl'.format(str(time.time())[:10]))

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter+=1
        return self.counter % self.buffer_capacity == 0

    def update(self):
        self.training_step +=1

        state = torch.tensor(np.array([t.state for t in self.buffer]), dtype=torch.float).to(device)
        action = torch.tensor(np.array([t.action for t in self.buffer]), dtype=torch.float).view(-1, 1).to(device)
        reward = torch.tensor(np.array([t.reward for t in self.buffer]), dtype=torch.float).view(-1, 1).to(device)
        done = torch.tensor(np.array([t.done for t in self.buffer]), dtype=torch.bool).view(-1, 1).to(device)
        old_action_log_prob = torch.tensor(np.array([t.a_log_prob for t in self.buffer]), dtype=torch.float).view(-1, 1).to(device)
        target = []
        discounted_reward = 0
        for reward_t, is_terminal in zip(reversed([t.reward for t in self.buffer]), reversed([t.done for t in self.buffer])):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward_t + (args.gamma * discounted_reward)
            target.insert(0, discounted_reward)


        target = torch.tensor(target, dtype=torch.float32).to(device)
        target = (target - target.mean()) / (target.std() + 1e-7)

        action_loss = 100.0
        value_loss = 100.0
        for _ in range(self.ppo_epoch): # iteration ppo_epoch 
            for index in tqdm(BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, True), 
                desc = 'actor loss = {:.2f}, critic loss = {:.2f}'.format(action_loss, value_loss)):
                action_probs = self.actor_net(state[index].to(device))
                dist = Categorical(action_probs)
                action_log_prob = dist.log_prob(action[index].squeeze().to(device))

                ratio = torch.exp(action_log_prob - old_action_log_prob)
                critic_net_output = self.critic_net(state[index]).squeeze()
                target_idx = target[index]
                advantage = (target_idx - critic_net_output).detach()

                L1 = ratio * advantage
                L2 = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param) * advantage
                action_loss = -torch.min(L1, L2).mean() # MAX->MIN desent
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                value_loss = F.smooth_l1_loss(critic_net_output, target_idx)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()

        del self.buffer[:]


def main():
    agent = PPO()
    best_score = -100
    training_records = []
    running_reward = None 
    if args.strftime is None:
        strftime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    else:
        strftime = args.strftime
    flog = open("mult_logs/mult_{}b_{}.log".format(args.input_bit, strftime), "w")
    cnt = 0
    for i_epoch in range(100000):
        score = 0
        state = env.reset()
        done = False
        while done is False:
            action, action_log_prob = agent.select_action(state)
            next_state, done, reward, info = env.step(action)
            trans = Transition(state, action, reward, action_log_prob, next_state, done)
            if agent.store_transition(trans):
                agent.update()
            score += reward
            state = next_state

        if running_reward is None:
            running_reward = score
        else:
            running_reward = running_reward * 0.9 + score * 0.1
        
        if score >= best_score * 0.95:
            best_score = score
        training_records.append(TrainRecord(i_epoch, running_reward))
        if i_epoch % 10 ==0:
            print("Epoch {}, Moving average score is: {:.2f}, this score is: {:.2f}".format(i_epoch, running_reward, score))
        flog.write("{}\t{:.2f}\t{:.0f}\t{:.2f}\t{:.0f}\t{}\t{}\n".format(env.verilog_file_name, 
            info["delay"], info["area"],
            info["delay_wo"], info["area_wo"],
            env.fa, env.ha))
        cnt += 1
        if cnt >= args.max_iter:
            break
        flog.flush()

if __name__ == '__main__':
    main()