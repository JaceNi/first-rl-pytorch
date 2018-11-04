
import gym
import time

import matplotlib.pyplot as plt


env = gym.make('FrozenLake-v0')

num_episodes = 1000

steps_total = []

for i_episode in range(num_episodes):
    
    state = env.reset()
    
    step = 0
    
    while True:
        step += 1
        
        action = env.action_space.sample()
        
        new_state, reward, done, info = env.step(action)
        
        print(new_state)
        print(info)
        
        time.sleep(0.4)
        
        env.render()
        
        if done:
            steps_total.append(step)
            print('Episode finished after %i steps' % step)
            break

print('Average number of steps: %.2f' % (sum(steps_total)/num_episodes))
plt.plot(steps_total)
plt.show()
 



