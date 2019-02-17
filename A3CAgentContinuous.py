#! -*- coding: UTF-8 -*-

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from collections import deque
import random

class Actor:
	def __init__(self, name, obvSpace_dim, actSpace_dim, actSpace_low, actSpace_high, alpha, training):
		self.name = name
		self.training = training

		# Value that encourages exploration.
		# Directly added to sigma (standard deviation value for normal distribution).
		self.explore_value = 1e-5
		self.entropy_value = 0.005

		with tf.variable_scope(self.name):
			self.tf_state = tf.placeholder(tf.float32, shape=(None,)+obvSpace_dim)

			# Forward.
			tf_hidden_1 = tf.layers.dense(
				self.tf_state,
				units=512,
				activation=tf.nn.relu6,
				kernel_initializer=tf.contrib.layers.xavier_initializer()
			)
			tf_hidden_out = tf.layers.dense(
				tf_hidden_1,
				units=256,
				activation=tf.nn.relu6,
				kernel_initializer=tf.contrib.layers.xavier_initializer()
			)

			self.tf_mu = tf.layers.dense(
				tf_hidden_out,
				units=actSpace_dim,
				activation=tf.nn.tanh,
				kernel_initializer=tf.contrib.layers.xavier_initializer()
			) * actSpace_high
			self.tf_sigma = tf.layers.dense(
				tf_hidden_out,
				units=actSpace_dim,
				activation=tf.nn.softplus,
				kernel_initializer=tf.contrib.layers.xavier_initializer()
			)
			if self.training:
				self.tf_sigma = self.tf_sigma + self.explore_value

			self.tf_normal_distribution = tfp.distributions.Normal(
				loc=self.tf_mu,
				scale=self.tf_sigma
			)

			self.tf_output_sample = tf.squeeze(self.tf_normal_distribution.sample(1), axis=0)
			self.tf_output = tf.clip_by_value(
				self.tf_output_sample,
				actSpace_low,
				actSpace_high
			)

			# Backward.
			if self.training:
				self.tf_advantage = tf.placeholder(tf.float32, shape=(None, 1))
				self.tf_action = tf.placeholder(tf.float32, shape=(None, actSpace_dim))

				self.tf_log_loss = self.tf_normal_distribution.log_prob(self.tf_action) * self.tf_advantage
				self.tf_entropy = self.entropy_value * self.tf_normal_distribution.entropy()
				self.tf_loss = tf.reduce_mean(-(self.tf_log_loss + self.tf_entropy))

				self.tf_optimizer = tf.train.RMSPropOptimizer(alpha)

				# Calculate gradients on worker's weights.
				self.tf_gradients = tf.gradients(
					self.tf_loss,
					tf.trainable_variables(self.name)
				)

				# Apply gradients to main network.
				self.tf_train = self.tf_optimizer.apply_gradients(
					zip(
						self.tf_gradients,
						tf.trainable_variables("_".join(["Main", "Actor"]))
					)
				)

class Critic:
	def __init__(self, name, obvSpace_dim, actSpace_dim, alpha, training):
		self.name = name
		self.training = training

		with tf.variable_scope(self.name):
			self.tf_state = tf.placeholder(tf.float32, shape=(None,)+obvSpace_dim)

			# Forward.
			tf_hidden_1 = tf.layers.dense(
				self.tf_state,
				units=512,
				activation=tf.nn.relu6,
				kernel_initializer=tf.contrib.layers.xavier_initializer()
			)
			tf_hidden_out = tf.layers.dense(
				tf_hidden_1,
				units=256,
				activation=tf.nn.relu6,
				kernel_initializer=tf.contrib.layers.xavier_initializer()
			)
			self.tf_output = tf.layers.dense(
				tf_hidden_out,
				units=1,
				activation=None,
				kernel_initializer=tf.contrib.layers.xavier_initializer()
			)

			# Backward.
			if self.training:
				self.tf_value_target = tf.placeholder(tf.float32, shape=(None, 1))

				self.tf_loss = tf.reduce_mean(
					tf.square(
						self.tf_value_target - self.tf_output
					)
				)

				self.tf_optimizer = tf.train.RMSPropOptimizer(alpha)

				# Calculate gradients on worker's weights.
				self.tf_gradients = tf.gradients(
					self.tf_loss,
					tf.trainable_variables(self.name)
				)

				# Apply gradients to main network.
				self.tf_train = self.tf_optimizer.apply_gradients(
					zip(
						self.tf_gradients,
						tf.trainable_variables("_".join(["Main", "Critic"]))
					)
				)

class A3CAgentContinuous:
	def __init__(self, name, sess, obvSpace_dim, actSpace_dim, actSpace_low, actSpace_high, gamma, alpha_actor, alpha_critic, training=True):
		self.name = name
		self.sess = sess
		self.obvSpace_dim = obvSpace_dim
		self.actSpace_dim = actSpace_dim
		self.actSpace_low = actSpace_low
		self.actSpace_high = actSpace_high
		self.gamma = gamma
		self.training = training

		self.name_actor = "_".join([self.name, "Actor"])
		self.name_critic = "_".join([self.name, "Critic"])

		self.memory_max_size = 1000000
		self.done = False

		# Build networks.
		self.actor = Actor(self.name_actor, obvSpace_dim, actSpace_dim, actSpace_low, actSpace_high, alpha_actor, self.training)
		if self.training:
			self.critic = Critic(self.name_critic, obvSpace_dim, actSpace_dim, alpha_critic, self.training)

			# Memory stores only current episode's experiences.
			# At end of the train function, its set to be empty.
			self.Memory = deque(maxlen=self.memory_max_size)

			# Op that pulls weights from global network.
			if self.name != "Main":
				self.tf_main_weights_update = [
					tar_v.assign(src_v)
					for src_v, tar_v in zip(
						tf.trainable_variables("Main"),
						tf.trainable_variables(self.name)
					)
				]

	def act(self, state):
		# With given state, calculates output of policy network.
		with self.sess.as_default(), self.sess.graph.as_default():
			state = np.expand_dims(state, axis=0)
			output, sigma = self.sess.run(
				[self.actor.tf_output, self.actor.tf_sigma],
				feed_dict={
					self.actor.tf_state:state
				}
			)
			return output[0], np.mean(sigma[0])

	# Remember given experience on memory.
	def remember(self, state, action, reward, next_state, done):
		self.Memory.append(
			[
				state,
				action,
				reward,
				next_state
			]
		)
		self.done = done

	# Sample from memory and train the model on it.
	def train(self):
		mini_batch = self.Memory

		state = np.array([np.array(b[0]) for b in mini_batch])
		action = np.array([np.array(b[1]) for b in mini_batch])
		reward = np.array([np.array(b[2]) for b in mini_batch])
		next_state = np.array([np.array(b[3]) for b in mini_batch])
		
		with self.sess.as_default(), self.sess.graph.as_default():
			# Target value for critic.
			if self.done:
				next_value = 0.0
			else:
				next_value = self.sess.run(
					self.critic.tf_output,
					feed_dict={
						self.critic.tf_state:np.expand_dims(next_state[-1], axis=0)
					}
				)[0][0]

			value_target = []
			for r in reward[::-1]:
				next_value = r + self.gamma * next_value
				value_target.append(next_value)
			value_target.reverse()
			value_target = np.expand_dims(np.array(value_target), axis=1)

			value_current = self.sess.run(
				self.critic.tf_output,
				feed_dict={
					self.critic.tf_state:state
				}
			)
			advantage = (value_target - value_current)

			### Train critic.
			self.sess.run(
				self.critic.tf_train,
				feed_dict={
					self.critic.tf_state:state,
					self.critic.tf_value_target:value_target
				}
			)

			### Train actor.
			self.sess.run(
				self.actor.tf_train,
				feed_dict={
					self.actor.tf_state:state,
					self.actor.tf_action:action,
					self.actor.tf_advantage:advantage
				}
			)

		# Set memory back to empty.
		self.Memory = deque(maxlen=self.memory_max_size)

	def weights_update(self):
		if self.name != "Main":
			with self.sess.as_default(), self.sess.graph.as_default():
				self.sess.run(
					self.tf_main_weights_update
				)

	def increment_global_episode(self, global_episode_increment_name):
		with self.sess.as_default(), self.sess.graph.as_default():
			self.sess.run(
				tf.get_default_graph().get_tensor_by_name(global_episode_increment_name)
			)
	def get_global_episode(self, global_episode_name):
		with self.sess.as_default(), self.sess.graph.as_default():
			return self.sess.run(
				tf.get_default_graph().get_tensor_by_name(global_episode_name)
			)