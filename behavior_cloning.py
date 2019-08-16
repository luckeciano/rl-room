import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
from tqdm import tqdm



class BehaviorCloning():

	def build_mlp_policy(self, n_hidden_layers = 2, hidden_layer_size = 64, ob_shape=None, action_shape=None):
		ob = tf.placeholder(name='ob', dtype=tf.float32, shape=(None, ob_shape[1]))

		x = ob
		for i in range(n_hidden_layers):
			x = tf.nn.tanh(tf.layers.dense(x, hidden_layer_size, name='fc%i'%(i+1)))
		actor = tf.layers.dense(x, action_shape[2], name = 'actions')
		return [ob, actor]

	def train(self, policy, S, A, epochs, batch_size):
		ac = tf.placeholder(name='expected_actions', dtype=tf.float32, shape=(None,A.shape[1],A.shape[2]))
		ob, actor = policy
		actor = tf.reshape(actor, shape=np.array([-1, A.shape[1], A.shape[2]]))
		error = tf.reduce_mean(0.5 * tf.square(actor - ac))
		opt = tf.train.AdamOptimizer(learning_rate=3e-4).minimize(error)

		sess = tf.get_default_session()
		sess.run(tf.global_variables_initializer())


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


def run_policy(envname, render, max_timesteps, num_rollouts, policy_fn):
	env = gym.make(envname)
	max_steps = max_timesteps or env.spec.timestep_limit

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
	parser.add_argument('--num_rollouts', type=int, default=20,
	                    help='Number of expert roll outs')
	args = parser.parse_args()

	print('loading and building expert policy')
	policy_fn = load_policy.load_policy(args.expert_policy_file)
	print('loaded and built')

	with tf.Session():
		tf_util.initialize()
		expert_data = run_policy(args.envname, args.render, args.max_timesteps, args.num_rollouts, policy_fn)
	    
		print(expert_data['observations'].shape, expert_data['actions'].shape)
		bc = BehaviorCloning()
		policy = bc.build_mlp_policy(ob_shape=expert_data['observations'].shape, action_shape=expert_data['actions'].shape)
		policy = bc.train(policy, S = expert_data['observations'], A = expert_data['actions'], epochs=100, batch_size=32)

		imitation_data = run_policy(args.envname, args.render, args.max_timesteps, args.num_rollouts, policy)



if __name__ == '__main__':
    main()

