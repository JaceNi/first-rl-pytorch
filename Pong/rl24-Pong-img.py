# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Dueling DQN implementation
Tuning - version stable - last 100 episodes - no reward lost

@author: udemy
"""
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import random
import math
import time

import matplotlib.pyplot as plt

# if gpu is to be used
use_cuda = torch.cuda.is_available()

device = torch.device("cuda:0" if use_cuda else "cpu")
Tensor = torch.Tensor
LongTensor = torch.LongTensor

env = gym.make('Pong-v0')

seed_value = 23
env.seed(seed_value)
torch.manual_seed(seed_value)
random.seed(seed_value)

###### PARAMS ######
learning_rate = 0.0001
num_episodes = 800
gamma = 0.99

replay_mem_size = 100000
batch_size = 64

update_target_frequency = 5000

double_dqn = True
hidden_layer = 300   # int((number_of_inputs+number_of_outputs)/2)


egreedy = 1
egreedy_final = 0.01
egreedy_decay = 10000

report_interval = 10
score_to_solve = 700

clip_error = True
normalize_image = True

####################

number_of_inputs = env.observation_space.shape[0]
number_of_outputs = env.action_space.n

def calculate_epsilon(steps_done):
    epsilon = egreedy_final + (egreedy - egreedy_final) * \
              math.exp(-1. * steps_done / egreedy_decay )
    return epsilon

def preprocess_frame(frame):
    frame = frame.transpose((2,0,1))
    frame = torch.from_numpy(frame)
    frame = frame.to(device, dtype=torch.float32)
    frame = frame.unsqueeze(1)
    return frame

class ExperienceReplay(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
 
    def push(self, state, action, new_state, reward, done):
        transition = (state, action, new_state, reward, done)
        
        if self.position >= len(self.memory):
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition
        
        self.position = ( self.position + 1 ) % self.capacity
        
        
    def sample(self, batch_size):
        return zip(*random.sample(self.memory, batch_size))
        
        
    def __len__(self):
        return len(self.memory)
        

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # padding = (kernel_size-1)/2

        self.pool  = nn.MaxPool2d(kernel_size=2)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1,  padding = 2) 
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding = 2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding = 2)

        self.lstm = nn.LSTMCell(537600, hidden_layer)
        
        self.advantage1 = nn.Linear(537600,hidden_layer)
        self.advantage2 = nn.Linear(hidden_layer,number_of_outputs)

        self.value1 = nn.Linear(537600,hidden_layer)
        self.value2 = nn.Linear(hidden_layer,1)
        
        self.activation1 = nn.ReLU()
        self.activation2 = nn.Tanh()
        
        
    def forward(self, x):
        if normalize_image:
            x = x[191:] /225
        print('#########################################cnn####################')
        print(x.shape)
        output_conv = self.conv1(x)                # (1, 210, 160)
        print(output_conv.shape)
        output_conv = self.activation1(output_conv) # (1, 210, 160)
        print(output_conv.shape)
        output_conv = self.conv2(output_conv)      # (32, 210, 160)
        print(output_conv.shape)
        output_conv = self.pool(output_conv)       # (32, 105, 80)
        print(output_conv.shape)
        output_conv = self.activation1(output_conv) # (32, 105, 80)
        print(output_conv.shape)
        output_conv = self.conv3(output_conv)      # (64, 105, 80)
        print(output_conv.shape)
        output_conv = self.activation1(output_conv) # (64, 105, 80)
        print(output_conv.shape)
        # flatten the tensor
        output_conv = output_conv.view(output_conv.size(0), -1)
        print(output_conv.shape)
        
        output_advantage = self.advantage1(output_conv)
        print(output_advantage.shape)
        output_advantage = self.activation1(output_advantage)
        output_advantage = self.advantage2(output_advantage)
        print(output_advantage.shape)
        
        output_value     = self.value1(output_conv)
        print(output_value.shape)
        output_value     = self.activation1(output_value)
        output_value     = self.value2(output_value)
        print(output_value.shape)
        
        output_final = output_value + output_advantage - output_advantage.mean()
        print('output_final: ', output_final, output_final.shape)
    
        return output_final
    
class QNet_Agent(object):
    def __init__(self):
        self.nn = NeuralNetwork().to(device)
        self.target_nn = NeuralNetwork().to(device)

        self.loss_func = nn.MSELoss()
        #self.loss_func = nn.SmoothL1Loss()
        
        self.optimizer = optim.Adam(params=self.nn.parameters(), lr=learning_rate)
        #self.optimizer = optim.RMSprop(params=mynn.parameters(), lr=learning_rate)
        
        self.update_target_counter = 0
        
    def select_action(self,state,epsilon):
        
        random_for_egreedy = torch.rand(1)[0]
        
        if random_for_egreedy > epsilon:      
            
            with torch.no_grad():
                
                state = preprocess_frame(state)
                action_from_nn = self.nn(state)
                print(action_from_nn)
                action = torch.max(action_from_nn,0)[1]
                print(action)
                action = action.item()
        else:
            action = env.action_space.sample()
        
        return action
    
    def optimize(self):
        
        if (len(memory) < batch_size):
            return
        
        state, action, new_state, reward, done = memory.sample(batch_size)
        
        state = [ preprocess_frame(frame) for frame in state]
        state = torch.cat(state)
        
        new_state = [ preprocess_frame(frame) for frame in new_state]
        new_state = torch.cat(new_state)
        
        state = Tensor(state).to(device)
        new_state = Tensor(new_state).to(device)
        reward = Tensor(reward).to(device)
        action = LongTensor(action).to(device)
        done = Tensor(done).to(device)


        if double_dqn:
            new_state_indexes = self.nn(new_state).detach()
            max_new_state_indexes = torch.max(new_state_indexes, 0)[1]  
            print('new_state_indexes: ', new_state_indexes)
            print('max_new_state_indexes', max_new_state_indexes)
            new_state_values = self.target_nn(new_state).detach()
            new_state_values.unsqueeze(1)
            print('new_state_values2:  ', new_state_values)
            max_new_state_values = new_state_values.gather(1, max_new_state_indexes.unsqueeze(1)).squeeze(1)
        else:
            new_state_values = self.target_nn(new_state).detach()
            max_new_state_values = torch.max(new_state_values, 1)[0]
        
        print('max_new_state_values: ', max_new_state_values.shape)
        print('reward: ',reward.shape)
        target_value = reward + ( 1 - done ) * gamma * max_new_state_values
  
        predicted_value = self.nn(state).gather(1, action.unsqueeze(1)).squeeze(1)
        
        loss = self.loss_func(predicted_value, target_value)
    
        self.optimizer.zero_grad()
        loss.backward()
        
        if clip_error:
            for param in self.nn.parameters():
                param.grad.data.clamp_(-1,1)
        
        self.optimizer.step()
        
        if self.update_target_counter % update_target_frequency == 0:
            self.target_nn.load_state_dict(self.nn.state_dict())
        
        self.update_target_counter += 1
        
        #Q[state, action] = reward + gamma * torch.max(Q[new_state])

        
        

memory = ExperienceReplay(replay_mem_size)
qnet_agent = QNet_Agent()

rewards_total = []

frames_total = 0 
solved_after = 0
solved = False

start_time = time.time()

for i_episode in range(num_episodes):
    
    state = env.reset()
    
    score = 0
    #for step in range(100):
    while True:
        
        frames_total += 1
        
        epsilon = calculate_epsilon(frames_total)
        
        #action = env.action_space.sample()
        action = qnet_agent.select_action(state, epsilon)
        
        new_state, reward, done, info = env.step(action)
        
        score += reward

        memory.push(state, action, new_state, reward, done)
        qnet_agent.optimize()
        
        state = new_state
        
        if i_episode>0 and  i_episode< 10:
            time.sleep(0.025)
            env.render()
        elif i_episode>100 and  i_episode< 110 :
            time.sleep(0.025)
            env.render()
        elif i_episode>200 and  i_episode< 210:
            time.sleep(0.025)
            env.render()
        elif i_episode>300 and  i_episode< 310:
            time.sleep(0.025)
            env.render()        
        
        if done:
            rewards_total.append(score)
            
            mean_reward_100 = sum(rewards_total[-100:])/100
            
            if (mean_reward_100 > score_to_solve and solved == False):
                print("SOLVED! After %i episodes " % i_episode)
                solved_after = i_episode
                solved = True
            
            if (i_episode % report_interval == 0):
                
                
                
                print("\n*** Episode %i *** \
                      \nAv.reward: [last %i]: %.2f, [last 100]: %.2f, [all]: %.2f \
                      \nepsilon: %.2f, frames_total: %i" 
                  % 
                  ( i_episode,
                    report_interval,
                    sum(rewards_total[-report_interval:])/report_interval,
                    mean_reward_100,
                    sum(rewards_total)/len(rewards_total),
                    epsilon,
                    frames_total
                          ) 
                  )
                  
                elapsed_time = time.time() - start_time
                print("Elapsed time: ", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

            break
        

print("\n\n\n\nAverage reward: %.2f" % (sum(rewards_total)/num_episodes))
print("Average reward (last 100 episodes): %.2f" % (sum(rewards_total[-100:])/100))
if solved:
    print("Solved after %i episodes" % solved_after)
plt.figure(figsize=(12,5))
plt.title("Rewards")
plt.bar(torch.arange(len(rewards_total)), rewards_total, alpha=0.6, color='green')
plt.show()

env.close()
env.env.close()
