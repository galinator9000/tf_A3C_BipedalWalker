#! -*- coding: UTF-8 -*-

from Worker import Worker
from A3CAgentContinuous import A3CAgentContinuous
import tensorflow as tf
import gym

max_episode = 50

env_name = "BipedalWalker-v2"
weight_path = "model/a3c-{}-weight".format(env_name.lower())

gamma = 0.99			# Future reward discount rate.
alpha_actor = 0.00001	# Learning rate for actors.
alpha_critic = 0.0001	# Learning rate for critics.

sess = tf.Session()

env = gym.make(env_name)
obvSpace_dim = env.observation_space.shape
try:
	actSpace_dim = env.action_space.shape[0]
except:
	actSpace_dim = env.action_space.n
actSpace_low = env.action_space.low
actSpace_high = env.action_space.high

with sess.as_default(), sess.graph.as_default():
	# Create main agent.
	Agent = A3CAgentContinuous(
		"Main",
		sess,
		obvSpace_dim,
		actSpace_dim,
		actSpace_low,
		actSpace_high,
		gamma,
		alpha_actor,
		alpha_critic,
		training=False
	)

	saver = tf.train.Saver()

	try:
		saver.restore(sess, weight_path)
		print("[+] Weights loaded.")
	except:
		print("[x] Weights couldn't load, exit...")
		exit()

	for episode in range(max_episode):
		state = env.reset()
		episode_reward_sum = 0.0
		done = False
		while not done:
			env.render()
			action, _ = Agent.act(state)
			next_state, reward, done, _ = env.step(action)
			if reward == -100:
				reward = -2

			episode_reward_sum += reward
			state = next_state

		print(
			"Episode {}".format(episode),
			"EpRew {0:.2f}".format(episode_reward_sum)
		)