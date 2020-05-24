# credits to Ashish Gupta @ https://towardsdatascience.com/ai-learning-to-land-a-rocket-reinforcement-learning-84d61f97d055
# for Keras implementation reference

import gym
import numpy as np 
from time import sleep
import os
import random
import tensorflow as tf
from collections import deque

env = gym.make('LunarLander-v2')

train_new = False # set to true to train a new model from scratch
model_filepath = 'lunar_lander_model.h5'

# define hyperparameters
learning_rate = 0.001
discount_rate = 0.99
batch_size = 64
replay_buffer_size = 500000
epsilon_min = 0.01
epsilon_decay = 0.995

class DQN_agent:

	def __init__(self, action_space, state_space):

		self.action_space = action_space
		self.state_space = state_space
		self.replay_buffer = deque(maxlen=replay_buffer_size) 
		self.epsilon = 1.0

		if train_new:
			self.model = self.build_model()
		else:
			self.model = tf.keras.models.load_model(model_filepath)

	def build_model(self):

		model = tf.keras.Sequential([
									tf.keras.layers.InputLayer(input_shape=self.state_space),
									tf.keras.layers.Dense(512, activation='relu'),
									tf.keras.layers.Dense(256, activation='relu'),
									tf.keras.layers.Dense(self.action_space, activation='linear')
									])
		opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
		model.compile(loss='mse', optimizer=opt)

		return model

	def store(self, state, action, reward, next_state, done):

		# store states to replay memory to create a stable input dataset for training
		self.replay_buffer.append((state, action, reward, next_state, done))

	def choose_action(self, state):

		if np.random.rand() <= self.epsilon:
			# explore
			return random.randrange(self.action_space)

		# exploit
		chosen_action = self.model.predict(state)

		return np.argmax(chosen_action[0])

	def update_model(self):

		# once we've collected enough states for stable training
		if len(self.replay_buffer) > batch_size:

			# select random minibatch for training
			batch = random.sample(self.replay_buffer, batch_size)
			states = np.array([i[0] for i in batch])
			actions = np.array([i[1] for i in batch])
			rewards = np.array([i[2] for i in batch])
			next_states = np.array([i[3] for i in batch])
			non_terminal_states = 1 - np.array([i[4] for i in batch])

			states = np.squeeze(states)
			next_states = np.squeeze(next_states)

			target = rewards + discount_rate*(np.amax(self.model.predict_on_batch(next_states), axis=1))*non_terminal_states
			targets = self.model.predict_on_batch(states)
			idx = np.array([i for i in range(batch_size)])
			targets[[idx], [actions]] = target

			self.model.fit(states, targets, epochs=1, verbose=0)

			if self.epsilon > epsilon_min:
				self.epsilon *= epsilon_decay

	def save(self):
		self.model.save(model_filepath)

# variables for controlling training loop
num_episodes = 600
max_steps = 1000
total_rewards = []

agent = DQN_agent(env.action_space.n, env.observation_space.shape[0])

for episode in range(num_episodes):

	state = env.reset()
	state = np.reshape(state, (1,8))
	rewards = 0

	os.system('cls')
	print(f"Training episode: {episode+1}/{num_episodes}")

	done = False

	for step in range(max_steps):
		action = agent.choose_action(state)
		#env.render()

		new_state, reward, done, info = env.step(action)
		rewards += reward
		new_state = np.reshape(new_state, (1, 8))
		agent.store(state, action, reward, new_state, done)
		state = new_state
		agent.update_model()

		if done:
			break

agent.save()

input("Press Enter to see trained agent")
sleep(1)

for episode in range(10):

	state = env.reset()
	state = np.reshape(state, (1,8))
	rewards = 0

	os.system('cls')
	print(f"Episode: {episode+1}")
	done = False

	for step in range(max_steps):
		action = agent.model.predict(state)

		env.render()

		new_state, reward, done, info = env.step(action)

		if done:
			break

env.close()

