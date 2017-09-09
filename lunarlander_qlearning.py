""" Trains an agent with (stochastic) Policy Gradients on LunarLander. Uses OpenAI Gym. """
import tensorflow as tf
import numpy as np
import gym
from gym.utils.play import *
from collections import deque, namedtuple
import random


class ModelParametersCopier():
    """
    Copy model parameters of one estimator to another.
    """

    def __init__(self, estimator1, estimator2):
        """
        Defines copy-work operation graph.  
        Args:
          estimator1: Estimator to copy the paramters from
          estimator2: Estimator to copy the parameters to
        """
        e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
        e1_params = sorted(e1_params, key=lambda v: v.name)
        e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
        e2_params = sorted(e2_params, key=lambda v: v.name)

        self.update_ops = []
        for e1_v, e2_v in zip(e1_params, e2_params):
            op = e2_v.assign(e1_v)
            self.update_ops.append(op)

    def make(self, sess):
        """
        Makes copy.
        Args:
            sess: Tensorflow session instance
        """
        sess.run(self.update_ops)


class ValueNetwork():
    """
    Policy Function approximator. 
    """

    def __init__(self, learning_rate, scope):
        self.scope = scope
        with tf.variable_scope(self.scope):
            self.state = tf.placeholder(dtype=tf.float32, shape=[None, 8], name="state")
            self.action = tf.placeholder(dtype=tf.int32, shape=[None, ], name="action")
            self.target_value = tf.placeholder(dtype=tf.float32, shape=[None, ], name="target_value")

            # FC1
            fc1 = tf.layers.dense(
                inputs=self.state,
                units=64,
                activation=tf.nn.relu,  # tanh activation
                name='FC1'
            )

            # FC2
            fc2 = tf.layers.dense(
                inputs=fc1,
                units=64,
                activation=tf.nn.relu,  # tanh activation
                name='FC2'
            )

            # prediction
            self.prediction = tf.layers.dense(
                inputs=fc2,
                units=4,
                activation=None,
                name='FC3'
            )

            # Get the predictions for the chosen actions only
            onehot_action = tf.one_hot(self.action, 4)
            self.action_predictions = tf.reduce_sum(tf.multiply(self.prediction, onehot_action), axis=1)

            # Calcualte the loss
            self.losses = tf.squared_difference(self.target_value, self.action_predictions)
            self.loss = tf.reduce_mean(self.losses)

            # train op
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state, sess):
        return sess.run(self.prediction, {self.state: state})

    def update(self, state, target_value, action, sess):
        feed_dict = {self.state: state, self.target_value: target_value, self.action: action}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


# hyperparameters
learning_rate = 0.005
gamma = 0.99  # discount factor for reward
resume = False  # resume from previous checkpoint?
render = False  # render the graphic ?
max_episode_number = 2000  # how many episode we want to run ?
max_replay_memory = 10000
batch_size = 64
train_freq = 16
epsilon = 1.0
random_decay = 0.995
model_path = "_models/qlearning/model.ckpt"  # path for saving the model

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

env = gym.make("LunarLander-v2")
replay_memory = []
running_reward = None
reward_sum = 0
episode_number = 0
num_steps = 0
loss = None

q_estimator = ValueNetwork(learning_rate, scope="q_estimator")


def epsilon_greedy(q_values, epsilon, action_space_size):
    if np.random.rand() < epsilon:
        return np.random.randint(0, action_space_size)
    else:
        return np.argmax(q_values)


# saver
saver = tf.train.Saver()
# session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

if resume:
    saver.restore(sess, model_path)

# start the first episode
observation = env.reset()
while episode_number < max_episode_number:
    if render: env.render()
    num_steps += 1

    current_state = observation

    # forward the policy network and sample an action from the returned probability
    q_values = q_estimator.predict(current_state[np.newaxis, :], sess)

    action = epsilon_greedy(q_values.ravel(), epsilon, 4)

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action)
    reward_sum += reward

    # record transition into replay memory
    if len(replay_memory) > max_replay_memory:
        replay_memory.pop(0)
    replay_memory.append(Transition(current_state, action, reward, observation, done))

    if num_steps % train_freq == 0:
        # Sample a minibatch from the replay memory
        samples = random.sample(replay_memory, min(len(replay_memory), batch_size))
        states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

        # Calculate q values and targets
        q_values_next = q_estimator.predict(next_states_batch, sess)
        targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * gamma * np.amax(
            q_values_next, axis=1)

        # Perform gradient descent update
        loss = q_estimator.update(states_batch, targets_batch, action_batch, sess)

    if done:
        print("loss: " + str(loss))
        episode_number += 1
        epsilon *= random_decay
        print("epsilon: " + str(epsilon))
        observation = env.reset()  # reset env

        # record running_reward to get overview of the improvement so far
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01

        print('ep %d: game finished, reward: %.2f, running_reward: %.2f' % (
            episode_number, reward_sum, running_reward))

        # reset reward_sum
        reward_sum = 0

        # save the model every 30 episodes
        if episode_number % 30 == 0: saver.save(sess, model_path)
