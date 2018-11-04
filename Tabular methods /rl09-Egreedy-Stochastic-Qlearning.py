# 
'''
LEFT  = 0
DOWN  = 1
RIGHT = 2
UP    = 3
'''



import gym
import torch

import matplotlib.pyplot as plt


env = gym.make('Taxi-v2')

number_of_states = env.observation_space.n
number_of_actions = env.action_space.n

gamma = 0.95
egreedy = 0.75
egreedy_final = 0
egreedy_decay = 0.999
learning_rate = 0.80
num_episodes = 2500
# row will be sattes, column will be actions
Q = torch.zeros([number_of_states, number_of_actions])


steps_total = []
rewards_total = []
egreedy_total = []

for i_episode in range(num_episodes):
    
    state = env.reset()
    
    step = 0
    score = 0
    
    while True: 
        step += 1
        
        #action = env.action_space.sample()
        random_for_egreedy = float(torch.rand(1))
        
        if random_for_egreedy > egreedy:
            random_values = random_values = Q[state] + torch.rand(1,number_of_actions)/1000
            action = int(torch.max(random_values,1)[1]) 
        else:
            action = env.action_space.sample()
        
        if egreedy > egreedy_final :
            egreedy *= egreedy_decay
        
        new_state, reward, done, info = env.step(action)
        
        Q[state, action] = (1-learning_rate)*Q[state, action] + learning_rate*(reward+gamma*torch.max(Q[new_state]))
        
        state = new_state
        
        score += reward
        
        # time.sleep(0.4)
        
        # env.render()
        
        print(new_state)
        print(info)
        # print(Q)
        if done:
            steps_total.append(step)
            rewards_total.append(score)
            egreedy_total.append(egreedy)
            print('Episode finished after %i steps' % step)
            print(score)
            print('---------step: ', step)
            break

print('\n \n \n Q value: \n',Q)
print('percentage of episodes_finished successfully: {0}'.format(sum(rewards_total)/num_episodes))
print('--------------------------percentage of last 100 episodes score: {0}'.format(sum(rewards_total[-100:])/100))

print('Average number of steps: %.2f' % (sum(steps_total)/1000))
print('Average number of last 100 steps: %.2f' % (sum(steps_total[-100:])/100))
print('maxmum score is: ', max(rewards_total))
print('minimum score is: ', min(steps_total))

plt.figure(figsize=(12,5))
plt.title('Rewards')
plt.bar(torch.arange(len(rewards_total)), rewards_total, alpha=0.6, color='green' )
plt.show()

plt.figure(figsize=(12,5))
plt.title('Steps')
plt.bar(torch.arange(len(steps_total)), steps_total, alpha=0.6, color='red' )
plt.show()

plt.figure(figsize=(12,5))
plt.title('Rewards')
plt.bar(torch.arange(len(rewards_total[-200:])), rewards_total[-200:], alpha=0.6, color='yellow' )
plt.show()

plt.figure(figsize=(12,5))
plt.title('egreedy values')
plt.bar(torch.arange(len(egreedy_total)), egreedy_total, alpha=0.6, color='blue' )
plt.show()

'''
Stochastic Bellman:
percentage of episodes_finished successfully: -15.998
--------------------------percentage of last 100 episodes: 7.73
Average number of steps: 27.19
Average number of last 100 steps: 12.91
maxmum score is:  15
'''

'''
Deterministic Bellman:
percentage of episodes_finished successfully: -13.749
--------------------------percentage of last 100 episodes: 7.65
Average number of steps: 27.05
Average number of last 100 steps: 13.35
maxmum score is:  15
minimum score is:  6
'''

'''
Stochastic Q Learning:
gamma = 0.9
learning_rate = 0.85
num_episodes = 1000
percentage of episodes_finished successfully: -13.121
--------------------------percentage of last 100 episodes: 8.13
Average number of steps: 26.64
Average number of last 100 steps: 12.87
maxmum score is:  15
minimum score is:  6
'''

'''
Stochastic Q Learning:
gamma = 1
learning_rate = 0.85
num_episodes = 1000
percentage of episodes_finished successfully: -17.113
--------------------------percentage of last 100 episodes score: 8.39
Average number of steps: 27.68
Average number of last 100 steps: 12.61
maxmum score is:  15
minimum score is:  6
'''

'''
Egreedy-Decay:
gamma = 0.88
egreedy = 0.7
egreedy_final = 0
egreedy_decay = 0.999
num_episodes = 1000
percentage of episodes_finished successfully: -15.617
--------------------------percentage of last 100 episodes score: 8.47
Average number of steps: 27.82               (more likely an accident)
Average number of last 100 steps: 12.53
maxmum score is:  15
minimum score is:  6
'''

'''
Egreedy-Decay:
gamma = 1
egreedy = 0.7
egreedy_final = 0
egreedy_decay = 0.999
num_episodes = 1000
percentage of episodes_finished successfully: -17.48
--------------------------percentage of last 100 episodes score: 8.17
Average number of steps: 27.31
Average number of last 100 steps: 12.83
maxmum score is:  15
minimum score is:  6
'''

'''
gamma = 0.88
egreedy = 0.7
egreedy_final = 0
egreedy_decay = 0.9999
num_episodes = 1000
percentage of episodes_finished successfully: -6.215
--------------------------percentage of last 100 episodes score: 8.29
Average number of steps: 56.27
Average number of last 100 steps: 12.71
maxmum score is:  15
minimum score is:  6
'''

'''
Egreedy-Stochastic-Qlearning:
gamma = 0.88
egreedy = 0.7
egreedy_final = 0
egreedy_decay = 0.999
learning_rate = 0.85
num_episodes = 1000
percentage of episodes_finished successfully: -15.094
--------------------------percentage of last 100 episodes score: 8.14
Average number of steps: 27.53
Average number of last 100 steps: 12.86
maxmum score is:  15
minimum score is:  6
'''

'''
Egreedy-Stochastic-Qlearning-LongerTerm:
percentage of episodes_finished successfully: -6.309
--------------------------percentage of last 100 episodes score: 8.04
Average number of steps: 56.30
Average number of last 100 steps: 12.78
maxmum score is:  15
minimum score is:  6
'''

'''
Egreedy-Stochastic-Qlearning-LongerTerm(less greedy):  
gamma = 0.88
egreedy = 0.7
egreedy_final = 0
egreedy_decay = 0.999
learning_rate = 0.85
num_episodes = 3000
percentage of episodes_finished successfully: 0.317
--------------------------percentage of last 100 episodes score: 8.29
Average number of steps: 53.09
Average number of last 100 steps: 12.71
maxmum score is:  15
minimum score is:  6
'''

'''
Egreedy-Deterministic-Qlearning(more certain about the future):
gamma = 1
egreedy = 0.7
egreedy_final = 0
egreedy_decay = 0.999
learning_rate = 0.9
num_episodes = 1000
percentage of episodes_finished successfully: -18.027
--------------------------percentage of last 100 episodes score: 7.76
Average number of steps: 27.83
Average number of last 100 steps: 13.24
maxmum score is:  15
minimum score is:  6
'''

'''
Egreedy-Stochastic-Qlearning-withoutExperience:
gamma = 0.88
egreedy = 0.7
egreedy_final = 0
egreedy_decay = 0.999
learning_rate = 1
num_episodes = 1000
percentage of episodes_finished successfully: -14.924
--------------------------percentage of last 100 episodes score: 8.25
Average number of steps: 27.07
Average number of last 100 steps: 12.75
maxmum score is:  15
minimum score is:  6
'''

'''
Egreedy-Stochastic-Qlearning-withoutExperience-LongerTerm:
gamma = 0.88
egreedy = 0.7
egreedy_final = 0
egreedy_decay = 0.999
learning_rate = 1
num_episodes = 3000     # (similar with the 1000, short-term: 8.29)
percentage of episodes_finished successfully: 0.29533333333333334
--------------------------percentage of last 100 episodes score: 8.28
Average number of steps: 53.17
Average number of last 100 steps: 12.72
maxmum score is:  15
minimum score is:  6
'''

'''
Egreedy-Stochastic-Qlearning-littleExperience-LongerTerm:
gamma = 0.88
egreedy = 0.7
egreedy_final = 0
egreedy_decay = 0.999
learning_rate = 0.95
num_episodes = 3000
percentage of episodes_finished successfully: 0.596
--------------------------percentage of last 100 episodes score: 8.3
Average number of steps: 52.35
Average number of last 100 steps: 12.70
maxmum score is:  15
minimum score is:  6
'''

'''
Egreedy-Stochastic-Qlearning-littleExperience-LongerTerm(more greedy first):
gamma = 0.88
egreedy = 0.75
egreedy_final = 0
egreedy_decay = 0.999
learning_rate = 0.95
num_episodes = 3000
percentage of episodes_finished successfully: 0.38466666666666666
--------------------------percentage of last 100 episodes score: 8.55
Average number of steps: 52.97
Average number of last 100 steps: 12.45
maxmum score is:  15
minimum score is:  6
'''

'''
Egreedy-Stochastic-Qlearning-LongerTerm(standard):
gamma = 0.88
egreedy = 0.75
egreedy_final = 0
egreedy_decay = 0.999
learning_rate = 0.85
num_episodes = 2500
percentage of episodes_finished successfully: -1.644
--------------------------percentage of last 100 episodes score: 8.44
Average number of steps: 47.45
Average number of last 100 steps: 12.56
maxmum score is:  15
minimum score is:  6
'''










