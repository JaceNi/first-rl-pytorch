#

# 
'''
LEFT  = 0
DOWN  = 1
RIGHT = 2
UP    = 3
'''



import gym
import time
import torch

import matplotlib.pyplot as plt

from gym.envs.registration import register
'''
register(
        id='FrozenLakeNotSlippery-v0',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name' : '4x4', 'is_slippery': False},
) 
'''

env = gym.make('FrozenLakeNotSlippery-v0')

number_of_states = env.observation_space.n
number_of_actions = env.action_space.n

gamma = 0.88

egreedy = 0.1

# row will be sattes, column will be actions
Q = torch.zeros([number_of_states, number_of_actions])

num_episodes = 1000

steps_total = []
rewards_total = []

for i_episode in range(num_episodes):
    
    state = env.reset()
    
    step = 0
    
    while True: 
        step += 1
        
        #action = env.action_space.sample()
        random_for_egreedy = float(torch.rand(1))
        
        if random_for_egreedy > egreedy:
            random_values = Q[state] + torch.rand(1,number_of_actions)/1000
            action = int(torch.max(random_values,1)[1]) 
        else:
            action = env.action_space.sample()
        
        
        new_state, reward, done, info = env.step(action)
        
        Q[state, action] = reward + gamma * torch.max(Q[new_state])
        
        state = new_state
        
        # time.sleep(0.4)
        
        env.render()
        
        print(new_state)
        print(info)
        print(Q)
        if done:
            steps_total.append(step)
            rewards_total.append(reward)
            print('Episode finished after %i steps' % step)
            break

print('\n \n \n Q value: \n',Q)
print('percentage of episodes_finished successfully: {0}'.format(sum(rewards_total)/num_episodes))
print('percentage of last 100 episodes: {0}'.format(sum(rewards_total[-100:])/100))

print('Average number of last 100 steps: %.2f' % (sum(steps_total[-100:])/100))

plt.figure(figsize=(12,5))
plt.title('Rewards')
plt.bar(torch.arange(len(rewards_total)), rewards_total, alpha=0.6, color='green' )
plt.show

plt.figure(figsize=(12,5))
plt.title('Steps')
plt.bar(torch.arange(len(steps_total)), steps_total, alpha=0.6, color='green' )
plt.show












