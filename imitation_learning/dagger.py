#!/usr/bin/env python

"""
Code for DAgger
Example usage:
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-384/libGL.so python dagger.py \
    experts/Humanoid-v2.pkl Humanoid-v2 --render --num_expert_rollouts=5

Author of this script: luckeciano@gmail.com
"""


import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
from tqdm import tqdm



class DAgger():

	def build_mlp_policy(self, n_hidden_layers = 2, hidden_layer_size = 64, ob_shape=None, action_shape=None):
		ob = tf.placeholder(name='ob', dtype=tf.float32, shape=(None, ob_shape[1]))

		x = ob
		for i in range(n_hidden_layers):
			x = tf.nn.tanh(tf.layers.dense(x, hidden_layer_size, name='fc%i'%(i+1)))
		actor = tf.layers.dense(x, action_shape[2], name = 'actions')

		ac = tf.placeholder(name='expected_actions', dtype=tf.float32, shape=(None,action_shape[1], action_shape[2]))

		actor = tf.reshape(actor, shape=np.array([-1, action_shape[1], action_shape[2]]))
		error = tf.reduce_mean(0.5 * tf.square(actor - ac))
		opt = tf.train.AdamOptimizer(learning_rate=3e-4).minimize(error)
		sess = tf.get_default_session()
		sess.run(tf.global_variables_initializer())

		return [ob, ac, opt, error, actor]

	def train(self, policy, S, A, epochs, batch_size):

		ob, ac, opt, error, actor = policy

		sess = tf.get_default_session()

		number_of_batches = S.shape[0]//batch_size
		sample_index = np.arange(S.shape[0])
		for i in range(epochs):
			np.random.shuffle(sample_index)
			pbar = tqdm(range(number_of_batches))
			for k in pbar:
				batch_index = sample_index[batch_size*k:batch_size*(k+1)]
				s_batch = S[batch_index,:]
				a_batch = A[batch_index,:]
				_, mse_run = sess.run([opt, error], feed_dict={ob: s_batch, ac: a_batch})
				pbar.set_description("Loss %s" % str(mse_run))


		return tf_util.function([ob], actor)

	def collect_rollouts(self, env, render, max_steps, num_rollouts, policy_fn):
		returns = []
		observations = []
		actions = []
		for i in range(num_rollouts):
		    print('iter', i)
		    obs = env.reset()
		    done = False
		    totalr = 0.
		    steps = 0
		    while not done:
		        action = policy_fn(obs[None,:])
		        observations.append(obs)
		        actions.append(action)
		        obs, r, done, _ = env.step(action)
		        totalr += r
		        steps += 1
		        if render:
		            env.render()
		        if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
		        if steps >= max_steps:
		            break
		    returns.append(totalr)
		rollouts_returns = np.array(returns)

		policy_data = {'observations': np.array(observations),
		                   'actions': np.array(actions)}
		return policy_data, rollouts_returns

	def run_dagger(self, env, render, max_steps, num_expert_rollouts, num_dagger_updates, rollouts_per_update, epochs, batch_size, expert_policy_fn, n_hidden_layers=2, hidden_layer_size=64, eval_running_policy=True):


		#1. collect data from expert policy data
		aggregated_data, _ = self.collect_rollouts(env, render, max_steps, num_expert_rollouts, expert_policy_fn)
		
		policy = self.build_mlp_policy(n_hidden_layers, hidden_layer_size, aggregated_data['observations'].shape, aggregated_data['actions'].shape)

		print(aggregated_data['observations'].shape, aggregated_data['actions'].shape)

		all_returns = []
		#2. train from aggregated data
		for i in range(num_dagger_updates):
			policy_fn = self.train(policy, aggregated_data['observations'], aggregated_data['actions'], epochs, batch_size)
			if eval_running_policy:
				_, curr_return = self.collect_rollouts(env, render, max_steps, 10, policy_fn)
				print ("Current Evaluation: " + str(np.mean(curr_return)) + " " + str(np.std(curr_return)))
				all_returns.append(curr_return)

			new_data, _ = self.collect_rollouts(env, render, max_steps, rollouts_per_update, policy_fn)
			new_data['actions'] = expert_policy_fn(new_data['observations'])
			
			new_data['actions'] = new_data['actions'].reshape((-1, aggregated_data['actions'].shape[1],aggregated_data['actions'].shape[2]))
			aggregated_data['observations'] = np.concatenate([aggregated_data['observations'], new_data['observations']])
			aggregated_data['actions'] = np.concatenate([aggregated_data['actions'], new_data['actions']])

		print(all_returns)
		for el in all_returns:
			print(np.mean(el), np.std(el))
		return policy_fn



def evaluate_policy(envname, render, max_timesteps, num_rollouts, policy_fn):

	returns = []
	observations = []
	actions = []
	for i in range(num_rollouts):
	    print('iter', i)
	    obs = env.reset()
	    done = False
	    totalr = 0.
	    steps = 0
	    while not done:
	        action = policy_fn(obs[None,:])
	        observations.append(obs)
	        actions.append(action)
	        obs, r, done, _ = env.step(action)
	        totalr += r
	        steps += 1
	        if render:
	            env.render()
	        if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
	        if steps >= max_steps:
	            break
	    returns.append(totalr)

	print('returns', returns)
	print('mean return', np.mean(returns))
	print('std of return', np.std(returns))

	expert_data = {'observations': np.array(observations),
	                   'actions': np.array(actions)}
	return expert_data


def main():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('expert_policy_file', type=str)
	parser.add_argument('envname', type=str)
	parser.add_argument('--render', action='store_true')
	parser.add_argument("--max_timesteps", type=int)
	parser.add_argument('--num_expert_rollouts', type=int, default=20,
	                    help='Number of expert roll outs')
	parser.add_argument('--num_dagger_updates', type=int, default=20,
						help='Number of dagger iterations')
	parser.add_argument('--rollouts_per_update', type=int, default=1,
						help='Number of rollouts collected per dagger iteration')
	parser.add_argument('--epochs', type=int, default=100)
	parser.add_argument('--batch_size', type=int, default=32)
	
	args = parser.parse_args()
	print('loading and building expert policy')
	expert_policy_fn = load_policy.load_policy(args.expert_policy_file)
	print('loaded and built')

	env = gym.make(args.envname)
	max_steps = args.max_timesteps or env.spec.timestep_limit
	with tf.Session():
		tf_util.initialize()
		dagger_policy_fn = DAgger().run_dagger(env, args.render, max_steps, args.num_expert_rollouts,
											args.num_dagger_updates, args.rollouts_per_update, 
											args.epochs, args.batch_size, expert_policy_fn)


if __name__ == '__main__':
    main()

