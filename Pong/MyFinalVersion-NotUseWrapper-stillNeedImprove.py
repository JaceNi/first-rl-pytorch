#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: udemy
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import gym
import random
import math
import time
import os.path

import matplotlib.pyplot as plt

plt.style.use('ggplot')

# if gpu is to be used
use_cuda = torch.cuda.is_available()

device = torch.device("cuda:0" if use_cuda else "cpu")
Tensor = torch.Tensor
LongTensor = torch.LongTensor


#directory = './PongVideos/'
#env = gym.wrappers.Monitor(env, directory, video_callable=lambda episode_id: episode_id%20==0)
env = gym.make('Boxing-v0')

seed_value = 23
env.seed(seed_value)
torch.manual_seed(seed_value)
random.seed(seed_value)

###### PARAMS ######

learning_rate = 0.0001
num_episodes = 1000
gamma = 0.99

hidden_layer = 512

replay_mem_size = 100000
batch_size = 32

update_target_frequency = 5000

double_dqn = True

egreedy = 0.9
egreedy_final = 0.01
egreedy_decay = 10000

report_interval = 10
score_to_solve = 18

clip_error = False
normalize_image = True

file2save = 'model_save.pth'
save_model_frequency = 10000
resume_previous_training = False

####################

number_of_inputs = env.observation_space.shape[0]
number_of_outputs = env.action_space.n

def calculate_epsilon(steps_done):
    epsilon = egreedy_final + (egreedy - egreedy_final) * \
              math.exp(-1. * steps_done / egreedy_decay )
    return epsilon

def load_model():
    return torch.load(file2save)

def save_model(model):
    torch.save(model.state_dict(), file2save)
    
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop((84,84)),
    transforms.ToTensor() 
    ]
)
    
     
def to_grey(tensor):
    # TODO: make efficient
    R = tensor[0]
    G = tensor[1]
    B = tensor[2]
    tensor[0]=0.333*R+0.333*G+0.334*B
    tensor = tensor[0]
    return tensor
    
def preprocess_frame(frame):
    frame = frame.transpose((2,0,1))
    frame = torch.from_numpy(frame)
    frame = frame.to(device, dtype=torch.float32)
    frame = frame.unsqueeze(1)
    frame = to_grey(frame)
    #frame = transform(frame)
    #print('frame.shape: ', frame.shape)
    frame = torch.unsqueeze(frame, 0)
    return frame

def plot_results():
    plt.figure(figsize=(12,5))
    plt.title("Rewards")
    plt.plot(rewards_total, alpha=0.6, color='red')
    plt.show()
    plt.savefig("Pong-results.png")
    plt.close()

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
        
        self.pool  = nn.MaxPool2d(kernel_size=2)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        
        self.lstm = nn.LSTMCell(22528, hidden_layer)
        
        self.advantage1 = nn.Linear(hidden_layer,hidden_layer)
        self.advantage2 = nn.Linear(hidden_layer, number_of_outputs)
        
        self.value1 = nn.Linear(hidden_layer,hidden_layer)
        self.value2 = nn.Linear(hidden_layer,1)

        self.activation2 = nn.Tanh()
        self.activation = nn.ReLU()
        
        
    def forward(self, x):
        
        if normalize_image:
            x = x / 255
        # print('x.shape', x.shape)
        output_conv = self.conv1(x)
        output_conv = self.activation(output_conv)
        #output_conv = self.pool(output_conv)
        output_conv = self.conv2(output_conv)
        output_conv = self.activation(output_conv)
        #output_conv = self.pool(output_conv)
        output_conv = self.conv3(output_conv)
        output_conv = self.activation(output_conv)
        #output_conv = self.pool(output_conv)
        
        output1 = output_conv.view(output_conv.size(0), -1) # flatten
        
        #output1 = output_conv.view(-1, 64)
        output1, hc = self.lstm(output1)
        
        output_advantage = self.advantage1(output1)
        output_advantage = self.activation(output_advantage)
        output_advantage = self.advantage2(output_advantage)
        output_advantage = self.activation(output_advantage)
        
        output_value = self.value1(output1)
        output_value = self.activation(output_value)
        output_value = self.value2(output_value)
        output_value = self.activation(output_value)
        
        output_final = output_value + output_advantage - output_advantage.mean()

        return output_final
    
class QNet_Agent(object):
    def __init__(self):
        self.nn = NeuralNetwork().to(device)
        self.target_nn = NeuralNetwork().to(device)

        self.loss_func = nn.MSELoss()
        #self.loss_func = nn.SmoothL1Loss()
        
        self.optimizer = optim.Adam(params=self.nn.parameters(), lr=learning_rate)
        #self.optimizer = optim.RMSprop(params=mynn.parameters(), lr=learning_rate)
        
        self.number_of_frames = 0
        
        if resume_previous_training and os.path.exists(file2save):
            print("Loading previously saved model ... ")
            self.nn.load_state_dict(load_model())
        
    def select_action(self,state,epsilon):
        
        random_for_egreedy = torch.rand(1)[0]
        
        if random_for_egreedy > epsilon:      
            
            with torch.no_grad():
                
                state = preprocess_frame(state)
                action_from_nn = self.nn(state)
                
                action = torch.max(action_from_nn,1)[1]
                #print(action)
                action = action[0].item()        
        else:
            action = env.action_space.sample()
        
        return action
    
    def optimize(self):
        
        if (len(memory) < batch_size):
            return
        
        state, action, new_state, reward, done = memory.sample(batch_size)

        state = [ preprocess_frame(frame) for frame in state ] 
        state = torch.cat(state)
        
        new_state = [ preprocess_frame(frame) for frame in new_state ] 
        new_state = torch.cat(new_state)

        reward = Tensor(reward).to(device)
        action = LongTensor(action).to(device)
        done = Tensor(done).to(device)


        if double_dqn:
            new_state_indexes = self.nn(new_state).detach()
            max_new_state_indexes = torch.max(new_state_indexes, 1)[1]  
            
            new_state_values = self.target_nn(new_state).detach()
            #print('new_state_values: ', new_state_values.shape)
            #print('max_new_state_indexes: ', max_new_state_indexes, max_new_state_indexes.shape)
            max_new_state_values = new_state_values.gather(1, max_new_state_indexes.unsqueeze(1)).squeeze(1)                                      #   
        else:
            new_state_values = self.target_nn(new_state).detach()
            max_new_state_values = torch.max(new_state_values, 1)[0]
       
        '''
        # state:                torch.Size([96, 1, 210, 160])
        # new_state:            torch.Size([96, 1, 210, 160])
        # reward:               torch.Size([32])
        # action:               torch.Size([32])
        # done:                 torch.Size([32])
        # max_new_state_values: torch.Size([32])
        # target_value:         torch.Size([32])
        # predicted_value:      torch.Size([32])
       
        
        print('state: ', state, state.shape)
        print('new_state: ', new_state, new_state.shape)
        print('action: ', action, action.shape)
        print('reward: ', reward, reward.shape)
        print('done: ', done, done.shape)
        print('max_new_state_values: ', max_new_state_values, max_new_state_values.shape)
        '''
        target_value = reward + ( 1 - done ) * gamma * max_new_state_values
        # print('target_value ',target_value, target_value.shape)
  
        predicted_value = self.nn(state).gather(1, action.unsqueeze(1)).squeeze(1)
        # print('predicted_value', predicted_value, predicted_value.shape)
        loss = self.loss_func(predicted_value, target_value)
    
        self.optimizer.zero_grad()
        loss.backward()
        
        if clip_error:
            for param in self.nn.parameters():
                param.grad.data.clamp_(-1,1)
        
        self.optimizer.step()
        
        if self.number_of_frames % update_target_frequency == 0:
            self.target_nn.load_state_dict(self.nn.state_dict())
        
        if self.number_of_frames % save_model_frequency == 0 and self.number_of_frames!=0:
            save_model(self.nn)
        
        self.number_of_frames += 1
        
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
    print('i_episode: ', i_episode)
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
        
        #time.sleep(0.00025)
        env.render()
        
        '''
        if i_episode>0 and  i_episode<2:
            #time.sleep(0.025)
            env.render()
        elif i_episode>100 and  i_episode< 102 :
            #time.sleep(0.025)
            env.render()
        elif i_episode>200 and  i_episode< 202:
            #time.sleep(0.025)
            env.render()
        elif i_episode>300 and  i_episode< 302:
            #time.sleep(0.025)
            env.render()
        '''
        
        if done:
            rewards_total.append(score)
            
            mean_reward_100 = sum(rewards_total[-100:])/100
            
            if (mean_reward_100 > score_to_solve and solved == False):
                print("SOLVED! After %i episodes " % i_episode)
                solved_after = i_episode
                solved = True
            
            if (i_episode % report_interval == 0):
                
                #plot_results()
                
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



