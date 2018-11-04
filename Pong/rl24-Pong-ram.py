





# CartPole-DoubleDQN
# 一个神经网络给出action的选择，另一个神经网络基于众多actions的选择打分（最终作出决策）。

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

env = gym.make('Alien-v0')

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
hidden_layer = 100   # int((number_of_inputs+number_of_outputs)/2)


egreedy = 1
egreedy_final = 0.01
egreedy_decay = 10000

report_interval = 10
score_to_solve = 700

clip_error = True

####################

number_of_inputs = env.observation_space.shape[0]
number_of_outputs = env.action_space.n

def calculate_epsilon(steps_done):
    epsilon = egreedy_final + (egreedy - egreedy_final) * \
              math.exp(-1. * steps_done / egreedy_decay )
    return epsilon

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
        self.linear1 = nn.Linear(number_of_inputs,hidden_layer)
        self.linear2 = nn.Linear(hidden_layer,hidden_layer)

        self.lstm = nn.LSTMCell(hidden_layer, hidden_layer)
        
        self.advantage = nn.Linear(hidden_layer,number_of_outputs)
        self.value = nn.Linear(hidden_layer,1)

        self.activation = nn.ReLU()
        self.activation2 = nn.Tanh()
        
        
        
    def forward(self, x):
        output1 = self.linear1(x)
        output1 = self.activation(output1)

        output1 = self.linear2(output1)
        output1 = self.activation(output1)
        output1 = output1.view(-1, hidden_layer)
        output1, hc = self.lstm(output1)
        output1 = self.activation(output1)
        
        output_advantage = self.advantage(output1)
        output_value     = self.value(output1)
        
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
        
        self.update_target_counter = 0
        
    def select_action(self,state,epsilon):
        
        random_for_egreedy = torch.rand(1)[0]
        
        if random_for_egreedy > epsilon:      
            
            with torch.no_grad():
                
                state = Tensor(state).to(device)
                action_from_nn = self.nn(state)
                #print(action_from_nn)
                action = torch.max(action_from_nn,1)[1]
                #print(action)
                action = action.item()       
                #print(action)
        else:
            action = env.action_space.sample()
        
        return action
    
    def optimize(self):
        
        if (len(memory) < batch_size):
            return
        
        state, action, new_state, reward, done = memory.sample(batch_size)
        
        state = Tensor(state).to(device)
        new_state = Tensor(new_state).to(device)
        reward = Tensor(reward).to(device)
        action = LongTensor(action).to(device)
        done = Tensor(done).to(device)
        
        if double_dqn:
            new_state_indexes = self.nn(new_state).detach()
            max_new_state_indexes = torch.max(new_state_indexes, 1)[1]
            
            new_state_values = self.target_nn(new_state).detach()
            max_new_state_values = new_state_values.gather(1, max_new_state_indexes.unsqueeze(1)).squeeze(1)
        else:
            new_state_values = self.target_nn(new_state).detach()
            max_new_state_values = torch.max(new_state_values, 1)[0]
        '''
        batch_size = 64
        # state:                torch.Size([64, 128])
        # new_state:            torch.Size([64, 128])
        # reward:               torch.Size([64])
        # action:               torch.Size([64])
        # done                  torch.Size([64])
        # max_new_state_values: torch.Size([64])
        # target_value:         torch.Size([64])
        # predicted_value:      torch.Size([64])
        '''
        print('done', done, done.shape)
        print('state: ', state, state.shape)
        print('new_state: ', new_state, new_state.shape)
        print('action: ', action, action.shape)
        print('max_new_state_values: ', max_new_state_values, max_new_state_values.shape)
        print('reward: ', reward, reward.shape)
        target_value = reward + ( 1 - done ) * gamma * max_new_state_values
        print('target_value ',target_value, target_value.shape)
        predicted_value = self.nn(state).gather(1, action.unsqueeze(1)).squeeze(1)
        print('predicted_value', predicted_value, predicted_value.shape)
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


for i_episode in range(num_episodes):
    
    state = env.reset()
    start_time = time.time()
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
        '''
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
        '''  
        
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



