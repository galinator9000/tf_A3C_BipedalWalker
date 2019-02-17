#! -*- coding: UTF-8 -*-

from Worker import Worker
import tensorflow as tf
import multiprocessing, threading

num_workers = multiprocessing.cpu_count()

env_name = "BipedalWalker-v2"
weight_path = "model/a3c-{}-weight".format(env_name.lower())
load_weights = True
max_episode_each_worker = 50000

per_global_episode_play = 25
num_episode_play = 5

gamma = 0.99			# Future reward discount rate.
alpha_actor = 0.00001	# Learning rate for actors.
alpha_critic = 0.0001	# Learning rate for critics.

sess = tf.Session()

with sess.as_default(), sess.graph.as_default():
	# Create main agent.
	MainAgent = Worker("Main", env_name, gamma, alpha_actor, alpha_critic, sess)

	# Create workers.
	Workers = [
		Worker("Worker-{}".format(w_id), env_name, gamma, alpha_actor, alpha_critic, sess)
		for w_id in range(1, num_workers+1)
	]

	global_episode = tf.Variable(0.0, dtype=tf.float32, name="global_episode", trainable=False)
	global_episode_increment_name = global_episode.assign_add(1.0, name="global_episode_increment").name
	coord = tf.train.Coordinator()

	sess.run(tf.global_variables_initializer())

	# Only save main weights to disk.
	# Workers' initial weights are already copied from main network.
	saver = tf.train.Saver(tf.trainable_variables("Main"))

	# Load main weights if exists, or train from scratch.
	if load_weights:
		try:
			saver.restore(sess, weight_path)
			print("[+] Main weights loaded.")
		except:
			print("[!] Main weights couldn't loaded, starting from scratch.")

	threads = []

	# Main agent.
	Thread = threading.Thread(
		target=(
			lambda: MainAgent.play(coord, per_global_episode_play, num_episode_play, global_episode.name, saver, sess, weight_path)
		)
	)
	Thread.start()
	threads.append(Thread)

	# Workers.
	for w in Workers:
		Thread = threading.Thread(
			target=(
				lambda: w.work(coord, max_episode_each_worker, global_episode_increment_name)
			)
		)
		Thread.start()
		threads.append(Thread)

	# Here, all threads starts doing their work.
	coord.join(threads)

	saver.save(sess, weight_path)
	print("[+] Main weights saved, finished.")