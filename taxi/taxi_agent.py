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

# create Taxi environment
env = gym.make('Taxi-v3')

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
total_rewards = []
rewards = 0
num_episodes = 2000 # training episodes
max_steps = 100 # max number of steps per episode

'''
# un-comment to watch a random agent play

for episode in range(3):
	# reset the environment
	state = env.reset()
	step = 0
	done = False
	rewards = 0

	for step in range(max_steps):
		# clear screen
		os.system('cls')

		print("+++++EPISODE {}+++++".format(episode+1))
		print("Step {}".format(step+1))

		action = env.action_space.sample()

		new_state, reward, done, info = env.step(action)

		rewards += reward

		env.render()

		if rewards < 0:
			print(f"\nScore: {bcolors.RED}{rewards}{bcolors.RESET}")
		else:
			print(f"\nScore: {bcolors.GREEN}{rewards}{bcolors.RESET}")
		sleep(0.01)

		# if done, finish episode
		if done == True:
			break
'''

# train our agent
for episode in range(num_episodes):

	# reset the environment
	state = env.reset()
	step = 0
	done = False

	os.system('cls')
	print(f"AGENT IS TRAINING: {episode}/{num_episodes} episodes completed")
	if rewards < 0:
		print(f"\nEpisode score: {bcolors.RED}{(rewards):.0f}{bcolors.RESET}")
	else:
		print(f"\nEpisode score: {bcolors.GREEN}{(rewards):.0f}{bcolors.RESET}")

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
			break

	total_rewards.append(rewards)

	print(f"\nAverage score over time: {(sum(total_rewards)/(episode+1)):.0f}")

	# Decrease epsilon
	epsilon = np.exp(-decay_rate*episode)

os.system('cls')
print(f"Training completed over {num_episodes} episodes")
input("Press Enter to watch trained agent...")
os.system('cls')

# watch trained agent
for episode in range(5):

	# reset the environment
	state = env.reset()
	step = 0
	done = False
	rewards = 0

	for step in range(max_steps):
		# clear screen
		os.system('cls')

		print(f"TRAINED AGENT")
		print("+++++EPISODE {}+++++".format(episode+1))
		print("Step {}".format(step+1))

		# take the action (index) that has the maximum expected future reward given that state
		action = np.argmax(qtable[state,:])

		# take the action (a) and observe the outcome state (s') and reward (r)
		new_state, reward, done, info = env.step(action)
        
		rewards += reward

		env.render()

		if rewards < 0:
			print(f"\nScore: {bcolors.RED}{rewards}{bcolors.RESET}")
		else:
			print(f"\nScore: {bcolors.GREEN}{rewards}{bcolors.RESET}")
		sleep(0.01)   
		    
		# our new state is state
		state = new_state

		# if done, finish episode
		if done == True:
			sleep(1)
			break  

env.close()
