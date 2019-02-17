#! -*- coding: UTF-8 -*-

from A3CAgentContinuous import A3CAgentContinuous
import gym

class Worker:
	def __init__(self, name, env_name, gamma, alpha_actor, alpha_critic, sess):
		self.name = name

		self.env = gym.make(env_name)
		obvSpace_dim = self.env.observation_space.shape
		try:
			actSpace_dim = self.env.action_space.shape[0]
		except:
			actSpace_dim = self.env.action_space.n
		actSpace_low = self.env.action_space.low
		actSpace_high = self.env.action_space.high

		self.Agent = A3CAgentContinuous(
			self.name,
			sess,
			obvSpace_dim,
			actSpace_dim,
			actSpace_low,
			actSpace_high,
			gamma,
			alpha_actor,
			alpha_critic,
			training=True
		)

	# Only Main agent should use this.
	# Plays environment with own weights and reports information.
	def play(self, coord, per_global_episode, num_episode, global_episode_name, saver, sess, weight_path):
		print("[*] {} started playing.".format(self.name))
		while not coord.should_stop():
			global_episode = int(self.Agent.get_global_episode(global_episode_name))
			if (global_episode % per_global_episode) == 0:
				reward_record = []
				for ep in range(num_episode):
					state = self.env.reset()
					episode_reward_sum = 0.0
					done = False

					while not done:
						action, mean_sigma = self.Agent.act(state)
						next_state, reward, done, _ = self.env.step(action)
						if reward == -100:
							reward = -2

						episode_reward_sum += reward
						state = next_state
					reward_record.append(episode_reward_sum)

				print(
					"GlobalEp {0:.2f}".format(global_episode),
					"EpPlayed {}".format(num_episode),
					"MeanSig {0:.2f}".format(mean_sigma),
					"AvgRew {0:.3f}".format(sum(reward_record) / float(len(reward_record)))
				)

				# Weights should be saved.
				if global_episode > 0:
					saver.save(sess, weight_path)
					print("[*] Main weights saved.")

	# Workers uses this function to doing experiment on environment,
	# take actions and all, calculate gradients and does update on MAIN network.
	def work(self, coord, max_episode, global_episode_increment_name):
		print("[*] {} started working.".format(self.name))
		try:
			episode = 0

			# Initially, pull weights from main network.
			self.Agent.weights_update()

			while not coord.should_stop() and episode < max_episode:
				state = self.env.reset()
				done = False

				while not done:
					action, _ = self.Agent.act(state)
					next_state, reward, done, _ = self.env.step(action)

					# Decrease penalize.
					if reward == -100:
						reward = -2

					# Remember transition.
					self.Agent.remember(state, action, reward, next_state, done)
					state = next_state

					if done:
						break

				# Train on current played episode and throw the experience away.
				self.Agent.train()

				# Pull weights from main network again.
				self.Agent.weights_update()

				# End of the episode.
				episode += 1

				# Global episode count incremented by only first worker.
				if self.name == "Worker-1":
					self.Agent.increment_global_episode(global_episode_increment_name)
		except Exception as e:
			coord.request_stop(e)
