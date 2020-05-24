import numpy as np
import gym
import random
import os
from time import sleep

# to help with visualization
class bcolors:
    RED= '\u001b[31m'
    GREEN= '\u001b[32m'
    RESET= '\u001b[0m'

# create frozen lake environment
env = gym.make('FrozenLake-v0')

# initialize q-table
action_size = env.action_space.n
state_size = env.observation_space.n
qtable = np.zeros((state_size, action_size))

# define hyperparameters
learning_rate = 0.8
discount_rate = 0.95

# exploration hyperparameters
epsilon = 1.0
decay_rate= 0.005

# to help with game loop
rewards = 0
num_episodes = 10000 # training episodes
max_steps = 100 # max number of steps per episode
games_won = 0
games_lost = 0

'''
# uncomment to watch random agent play

for episode in range(3):
	# reset the environment
	state = env.reset()
	step = 0
	done = False
	
	for step in range(max_steps):
		# clear screen
		os.system('cls')

		print(f"+++++EPISODE {episode+1}+++++")
		print(f"Step {step}")

		action = env.action_space.sample()

		new_state, reward, done, info = env.step(action)

		env.render()
		sleep(0.2)

		# if done, finish episode
		if done == True:
			break
'''


# train the agent
for episode in range(num_episodes):

	'''
	# uncomment to watch training results over time
	os.system('cls')
	print(f"AGENT IS TRAINING: {episode}/{num_episodes} episodes completed")
	print("")
	print(f"Games won: {bcolors.GREEN}{games_won}{bcolors.RESET}")
	print("")
	print(f"Games lost: {bcolors.RED}{games_lost}{bcolors.RESET}")
	print("")
	print(f"% games won: {(games_won/(episode+1))*100:.1f}%")
	'''

	# reset the environment
	state = env.reset()
	step = 0
	done = False
	rewards = 0
    
	for step in range(max_steps):

		# Exploration-exploitation tradeoff
		if random.uniform(0,1) < epsilon:
			# Explore
			action = env.action_space.sample()
		else:
			# Exploit
			action = np.argmax(qtable[state,:])

		# Take an action and observe the reward
		new_state, reward, done, info = env.step(action)

		rewards += reward

		# Q-learning algorithm
		qtable[state,action] = qtable[state,action] + learning_rate * (reward + discount_rate * np.max(qtable[new_state,:])-qtable[state,action])

		# Update to our new state
		state = new_state

		# if done, finish episode
		if done == True:
			if reward <= 0:
				games_lost += 1
			else:
				games_won += 1
			break

	# Decrease epsilon (explore less)
	epsilon = np.exp(-decay_rate*episode)

os.system('cls')
print(f"Training completed over {num_episodes} episodes")
input("Press Enter to watch trained agent...")
os.system('cls')

# watch trained agent
for episode in range(10):

	# reset the environment
	state = env.reset()
	step = 0
	done = False

	for step in range(max_steps):

		os.system('cls')

		print(f"+++++EPISODE {episode+1}+++++")
		print(f"Step {step+1}")

		action = np.argmax(qtable[state,:])

		new_state, reward, done, info = env.step(action)

		env.render()
		sleep(0.1)

		state = new_state

		if done == True:
			if reward == 0:
				print(f"\n{bcolors.RED}lol you fell in a hole{bcolors.RESET}")
			else:
				print(f"\n{bcolors.GREEN}You reached the goal!{bcolors.RESET}")
			sleep(1.5)
			break  

env.close()
