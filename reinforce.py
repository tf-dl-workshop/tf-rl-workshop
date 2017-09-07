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
            self.state = tf.placeholder(dtype=tf.float32, shape=[None, 80, 80, 1], name="state")
            self.action = tf.placeholder(dtype=tf.int32, shape=[None], name="action")
            self.reward = tf.placeholder(dtype=tf.float32, shape=[None], name="target")

            conv1 = tf.layers.conv2d(self.state, filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same")
            pooling1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=[2, 2], padding="same")
            conv2 = tf.layers.conv2d(pooling1, filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same")
            pooling2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=[2, 2], padding="same")

            flatten = tf.reshape(pooling2, shape=[-1, 20 * 20 * 16])
            fully_connected = tf.layers.dense(flatten, units=128, activation=tf.nn.relu)
            self.logits = tf.layers.dense(fully_connected, units=2)
            self.sample_action = tf.multinomial(self.logits, 1)

            one_hot_action = tf.one_hot(self.action, 2)
            action_prob = tf.nn.softmax(self.logits)
            action_prob = tf.Print(action_prob, [action_prob], summarize=1000)
            picked_action_prob = tf.matmul(one_hot_action, action_prob, transpose_b=True)
            self.loss = tf.reduce_mean(-tf.log(picked_action_prob) * self.reward)

            # Loss and train op
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.sample_action, {self.state: state})

    def update(self, state, reward, action, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state, self.reward: reward, self.action: action}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


UP = 2
DOWN = 3
# action index
action_to_index = {UP: 0, DOWN: 1}
index_to_action = {0: UP, 1: DOWN}

# hyperparameters
batch_size = 2  # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99  # discount factor for reward
resume = False  # resume from previous checkpoint?
render = False


def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return np.expand_dims(I, 3)


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


env = gym.make("Pong-v0")
x = env.unwrapped.get_action_meanings()
observation = env.reset()
prev_x = None  # used in computing the difference frame
state_list, action_list, reward_list = [], [], []
running_reward = None
reward_sum = 0
episode_number = 0
num_steps = 0

policy_network = PolicyNetwork(learning_rate)

# saver
saver = tf.train.Saver()
# session
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

if resume:
    saver.restore(sess, "_models/model.ckpt")

start = time.time()
while True:
    if render: env.render()
    # preprocess the observation, set input to network to be difference image
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(shape=(80, 80, 1))
    prev_x = cur_x

    # forward the policy network and sample an action from the returned probability
    a_index = policy_network.predict(np.expand_dims(x, 0))
    action = index_to_action[int(a_index)]

    # record action_index
    action_list.append(int(a_index))

    # record various intermediates (needed later for backprop)
    state_list.append(x)  # observation
    # step the environment and get new measurements
    observation, reward, done, info = env.step(action)
    num_steps += 1
    reward_sum += reward

    reward_list.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)

    if done:  # an episode finished
        episode_number += 1

        if episode_number % batch_size == 0:
            # perform rmsprop parameter update every batch_size episodes
            print("updating weights !!!")
            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            state_batch = np.stack(state_list)
            reward_batch = np.stack(reward_list)
            action_batch = np.stack(action_list)

            state_list, action_list, reward_list = [], [], []  # reset array memory

            # compute the discounted reward backwards through time
            discounted_epr = discount_rewards(reward_batch)
            # standardize the rewards to be unit normal (helps control the gradient estimator variance)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            loss = policy_network.update(state_batch, discounted_epr, action_batch)
            print("loss " + str(loss))

        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01

        if episode_number % 10 == 0: saver.save(sess, "_models/model.ckpt")
        reward_sum = 0
        observation = env.reset()  # reset env
        prev_x = None

    if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends.
        end = time.time()
        print('ep %d: game finished, reward: %f, num_steps: %f, time per steps %f' % (
            episode_number, int(reward), int(num_steps), ((end - start) / num_steps)) + (
                  '' if reward == -1 else ' !!!!!!!!'))
        # reset num_steps
        start = time.time()
        num_steps = 0
