""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """

import tensorflow as tf
import numpy as np
import gym
import time


class PolicyNetwork():
    """
    Policy Function approximator. 
    """

    def __init__(self, learning_rate, scope="policy_network"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(dtype=tf.float32, shape=[None, 8], name="state")
            self.action = tf.placeholder(dtype=tf.int32, shape=[None, ], name="action")
            self.reward = tf.placeholder(dtype=tf.float32, shape=[None, ], name="reward")
            # reward = tf.Print(self.reward, [self.reward], summarize=100)

            # conv1 = tf.layers.conv2d(self.state, filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same")
            # pooling1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=[2, 2], padding="same")
            # conv2 = tf.layers.conv2d(pooling1, filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same")
            # pooling2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=[2, 2], padding="same")
            #
            # flatten = tf.reshape(pooling2, shape=[-1, 20 * 20 * 16])

            # FC1
            fc1 = tf.layers.dense(
                inputs=self.state,
                units=16,
                activation=tf.nn.tanh,  # tanh activation
                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                bias_initializer=tf.constant_initializer(0.1),
                name='FC1'
            )

            # FC2
            fc2 = tf.layers.dense(
                inputs=fc1,
                units=32,
                activation=tf.nn.tanh,  # tanh activation
                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                bias_initializer=tf.constant_initializer(0.1),
                name='FC2'
            )

            # FC3
            logits = tf.layers.dense(
                inputs=fc2,
                units=4,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                bias_initializer=tf.constant_initializer(0.1),
                name='FC3'
            )

            for t in tf.trainable_variables():
                tf.summary.histogram(t.name.replace(":", ""), t)

            self.summaries = tf.summary.merge_all()

            self.action_prob = tf.nn.softmax(logits)
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.action)

            self.loss = tf.reduce_mean(neg_log_prob * self.reward)
            # Loss and train op
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.action_prob, {self.state: state})

    def update(self, state, reward, action, writer, num_step, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state, self.reward: reward, self.action: action}
        summary, _, loss = sess.run([self.summaries, self.train_op, self.loss], feed_dict)
        writer.add_summary(summary, num_step)
        return loss


# UP = 2
# DOWN = 3
# # action index
# action_to_index = {UP: 0, DOWN: 1}
# index_to_action = {0: UP, 1: DOWN}

# hyperparameters
learning_rate = 0.005
gamma = 0.99  # discount factor for reward
resume = False  # resume from previous checkpoint?
render = False

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        # if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


env = gym.make("LunarLander-v2")
observation = env.reset()
state_list, action_list, reward_list = [], [], []
running_reward = None
reward_sum = 0
episode_number = 0
num_step = 0

policy_network = PolicyNetwork(learning_rate)

# saver
saver = tf.train.Saver()
# session
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

if resume:
    saver.restore(sess, "_models/model.ckpt")

writer = tf.summary.FileWriter("_models/histogram", graph=tf.get_default_graph())
start = time.time()

while True:
    if render: env.render()

    x = observation

    # # preprocess the observation, set input to network to be difference image
    # cur_x = prepro(observation)
    # x = cur_x - prev_x if prev_x is not None else np.zeros(shape=(80 * 80))
    # prev_x = cur_x

    # forward the policy network and sample an action from the returned probability
    action_prob = policy_network.predict(x[np.newaxis, :], sess)
    action = np.random.choice(a=4, p=action_prob.ravel())

    # record various intermediates
    action_list.append(action)
    state_list.append(x)  # observation
    # step the environment and get new measurements

    observation, reward, done, info = env.step(action)
    reward_sum += reward

    reward_list.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)

    if done:  # an episode finished
        episode_number += 1

        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        state_batch = np.vstack(state_list)
        action_batch = np.array(action_list)
        reward_batch = np.array(reward_list)

        state_list, action_list, reward_list = [], [], []  # reset array memory

        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards(reward_batch)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        policy_network.update(state_batch, discounted_epr, action_batch, writer, episode_number, sess)

        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01

        if episode_number % 30 == 0: saver.save(sess, "_models/model.ckpt")
        observation = env.reset()  # reset env
        prev_x = None

        print('ep %d: game finished, reward: %f, running_reward: %f' % (
            episode_number, reward_sum, running_reward))
        # reset reward_sum
        reward_sum = 0
