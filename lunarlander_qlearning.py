""" Trains an agent with (stochastic) Policy Gradients on LunarLander. Uses OpenAI Gym. """
import tensorflow as tf
import numpy as np
import gym
from collections import namedtuple
import random

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
                activation=tf.nn.relu,
                name='FC1'
            )

            # FC2
            fc2 = tf.layers.dense(
                inputs=fc1,
                units=64,
                activation=tf.nn.relu,
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


# hyperparameters and configurations
learning_rate = 0.001
gamma = 0.99  # discount factor for reward
resume = False  # resume from previous checkpoint?
render = False  # render the graphic ?
is_train = True # training mode ?
max_episode_number = 1000  # how many episode we want to run ?
max_replay_memory = 2000 # maximum size of replay memory
epsilon_decay_steps = 500 # epsilon decay over time until ?
batch_size = 32 # training examples per batch
train_freq = 4 # train the model every "train_freq" steps
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
# The epsilon decay schedule
epsilons = np.linspace(1.0, 0.1, epsilon_decay_steps)


def epsilon_greedy(q_values, epsilon, action_space_size):
    """
    choose action randomly with epsilon probability
    otherwise choose action with maximum value
    
    :param q_values: estimated q values for all actions
    :param epsilon: exploration probability
    :param action_space_size: size of action space
    :return: a chosen action
    """
    if np.random.rand() < epsilon:
        action = np.random.randint(0, action_space_size)
    else:
        action = np.argmax(q_values)

    return action

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

    current_state = observation

    # forward the value network and get predicted q values
    q_values = q_estimator.predict(current_state[np.newaxis, :], sess)

    if is_train:
        epsilon = epsilons[min(episode_number, epsilon_decay_steps-1)]
    else:
        epsilon = 0.1
    action = epsilon_greedy(q_values.ravel(), epsilon, 4)

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action)
    reward_sum += reward

    # record transition into replay memory
    if len(replay_memory) > max_replay_memory:
        replay_memory.pop(0)
    replay_memory.append(Transition(current_state, action, reward, observation, done))

    num_steps += 1

    if num_steps % train_freq == 0 and is_train:
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
        print("epsilon " + str(epsilon))
        episode_number += 1

        observation = env.reset()  # reset env

        # record running_reward to get overview of the improvement so far
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01

        print('ep %d: game finished, reward: %.2f, running_reward: %.2f' % (
            episode_number, reward_sum, running_reward))

        # reset reward_sum
        reward_sum = 0

        # save the model every 20 episodes
        if episode_number % 20 == 0: saver.save(sess, model_path)
