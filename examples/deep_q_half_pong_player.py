# This is heavily based off https://github.com/asrivat1/DeepLearningVideoGames
# deep q learning agent that runs against Half-Pong. Runs on a much smaller screen and with fewer layers.
# Performs significantly above random, but still has someway to go to match google deep mind performance...
# To see a trained version of this network start it with the kwargs checkpoint_path="deep_q_half_pong_networks_40x40_8"
# and playback_mode="True"

import os
import random
from collections import deque

import tensorflow as tf
import numpy as np
import cv2
from pygame.constants import K_DOWN, K_UP

from pygame_player import PyGamePlayer


class DeepQHalfPongPlayer(PyGamePlayer):
    ACTIONS_COUNT = 3  # number of valid actions. In this case up, still and down
    FUTURE_REWARD_DISCOUNT = 0.99  # decay rate of past observations
    OBSERVATION_STEPS = 50000.  # time steps to observe before training
    EXPLORE_STEPS = 500000.  # frames over which to anneal epsilon
    INITIAL_RANDOM_ACTION_PROB = 1.0  # starting chance of an action being random
    FINAL_RANDOM_ACTION_PROB = 0.05  # final chance of an action being random
    MEMORY_SIZE = 500000  # number of observations to remember
    MINI_BATCH_SIZE = 200  # size of mini batches
    STATE_FRAMES = 4  # number of frames to store in the state
    OBS_LAST_STATE_INDEX, OBS_ACTION_INDEX, OBS_REWARD_INDEX, OBS_CURRENT_STATE_INDEX, OBS_TERMINAL_INDEX = range(5)
    SAVE_EVERY_X_STEPS = 10000
    LEARN_RATE = 1e-6
    STORE_SCORES_LEN = 200.
    SCREEN_WIDTH = 40
    SCREEN_HEIGHT = 40

    def __init__(self,
                 # to see a trained network change checkpoint_path="deep_q_half_pong_networks_40x40_8" and
                 # playback_mode="True"
                 checkpoint_path="deep_q_half_pong_networks",
                 playback_mode=True,
                 verbose_logging=True):
        """
        Example of deep q network for pong

        :param checkpoint_path: directory to store checkpoints in
        :type checkpoint_path: str
        :param playback_mode: if true games runs in real time mode and demos itself running
        :type playback_mode: bool
        :param verbose_logging: If true then extra log information is printed to std out
        :type verbose_logging: bool
        """
        self._playback_mode = playback_mode
        self.last_score = 0
        super(DeepQHalfPongPlayer, self).__init__(force_game_fps=8, run_real_time=playback_mode)
        self.verbose_logging = verbose_logging
        self._checkpoint_path = checkpoint_path
        self._session = tf.Session()
        self._input_layer, self._output_layer = self._create_network()

        self._action = tf.placeholder("float", [None, self.ACTIONS_COUNT])
        self._target = tf.placeholder("float", [None])

        readout_action = tf.reduce_sum(tf.mul(self._output_layer, self._action), reduction_indices=1)

        cost = tf.reduce_mean(tf.square(self._target - readout_action))
        self._train_operation = tf.train.AdamOptimizer(self.LEARN_RATE).minimize(cost)

        self._observations = deque()
        self._last_scores = deque()

        # set the first action to do nothing
        self._last_action = np.zeros(self.ACTIONS_COUNT)
        self._last_action[1] = 1

        self._last_state = None
        self._probability_of_random_action = self.INITIAL_RANDOM_ACTION_PROB
        self._time = 0

        self._session.run(tf.initialize_all_variables())

        if not os.path.exists(self._checkpoint_path):
            os.mkdir(self._checkpoint_path)
        self._saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(self._checkpoint_path)

        if checkpoint and checkpoint.model_checkpoint_path:
            self._saver.restore(self._session, checkpoint.model_checkpoint_path)
            print("Loaded checkpoints %s" % checkpoint.model_checkpoint_path)
        elif playback_mode:
            raise Exception("Could not load checkpoints for playback")

    def get_keys_pressed(self, screen_array, reward, terminal):
        # images will be black or white
        _, screen_binary = cv2.threshold(cv2.cvtColor(screen_array, cv2.COLOR_BGR2GRAY), 1, 255,
                                         cv2.THRESH_BINARY)

        if reward != 0.0:
            self._last_scores.append(reward)
            if len(self._last_scores) > self.STORE_SCORES_LEN:
                self._last_scores.popleft()

        # first frame must be handled differently
        if self._last_state is None:
            # the _last_state will contain the image data from the last self.STATE_FRAMES frames
            self._last_state = np.stack(tuple(screen_binary for _ in range(self.STATE_FRAMES)), axis=2)
            return DeepQHalfPongPlayer._key_presses_from_action(self._last_action)

        screen_binary = np.reshape(screen_binary,
                                   (self.SCREEN_WIDTH, self.SCREEN_HEIGHT, 1))
        current_state = np.append(self._last_state[:, :, 1:], screen_binary, axis=2)

        if not self._playback_mode:
            # store the transition in previous_observations
            self._observations.append((self._last_state, self._last_action, reward, current_state, terminal))

            if len(self._observations) > self.MEMORY_SIZE:
                self._observations.popleft()

            # only train if done observing
            if len(self._observations) > self.OBSERVATION_STEPS:
                self._train()
                self._time += 1

        # update the old values
        self._last_state = current_state

        self._last_action = self._choose_next_action()

        if not self._playback_mode:
            # gradually reduce the probability of a random actionself.
            if self._probability_of_random_action > self.FINAL_RANDOM_ACTION_PROB \
                    and len(self._observations) > self.OBSERVATION_STEPS:
                self._probability_of_random_action -= \
                    (self.INITIAL_RANDOM_ACTION_PROB - self.FINAL_RANDOM_ACTION_PROB) / self.EXPLORE_STEPS

            print("Time: %s random_action_prob: %s reward %s scores differential %s" %
                  (self._time, self._probability_of_random_action, reward,
                   sum(self._last_scores) / self.STORE_SCORES_LEN))

        return DeepQHalfPongPlayer._key_presses_from_action(self._last_action)

    def _choose_next_action(self):
        new_action = np.zeros([self.ACTIONS_COUNT])

        if (not self._playback_mode) and (random.random() <= self._probability_of_random_action):
            # choose an action randomly
            action_index = random.randrange(self.ACTIONS_COUNT)
        else:
            # choose an action given our last state
            readout_t = self._session.run(self._output_layer, feed_dict={self._input_layer: [self._last_state]})[0]
            if self.verbose_logging:
                print("Action Q-Values are %s" % readout_t)
            action_index = np.argmax(readout_t)

        new_action[action_index] = 1
        return new_action

    def _train(self):
        # sample a mini_batch to train on
        mini_batch = random.sample(self._observations, self.MINI_BATCH_SIZE)
        # get the batch variables
        previous_states = [d[self.OBS_LAST_STATE_INDEX] for d in mini_batch]
        actions = [d[self.OBS_ACTION_INDEX] for d in mini_batch]
        rewards = [d[self.OBS_REWARD_INDEX] for d in mini_batch]
        current_states = [d[self.OBS_CURRENT_STATE_INDEX] for d in mini_batch]
        agents_expected_reward = []
        # this gives us the agents expected reward for each action we might take
        agents_reward_per_action = self._session.run(self._output_layer, feed_dict={self._input_layer: current_states})
        for i in range(len(mini_batch)):
            if mini_batch[i][self.OBS_TERMINAL_INDEX]:
                # this was a terminal frame so there is no future reward...
                agents_expected_reward.append(rewards[i])
            else:
                agents_expected_reward.append(
                    rewards[i] + self.FUTURE_REWARD_DISCOUNT * np.max(agents_reward_per_action[i]))

        # learn that these actions in these states lead to this reward
        self._session.run(self._train_operation, feed_dict={
            self._input_layer: previous_states,
            self._action: actions,
            self._target: agents_expected_reward})

        # save checkpoints for later
        if self._time % self.SAVE_EVERY_X_STEPS == 0:
            self._saver.save(self._session, self._checkpoint_path + '/network', global_step=self._time)

    def _create_network(self):
        # network weights
        convolution_weights_1 = tf.Variable(tf.truncated_normal([8, 8, self.STATE_FRAMES, 32], stddev=0.01))
        convolution_bias_1 = tf.Variable(tf.constant(0.01, shape=[32]))

        convolution_weights_2 = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.01))
        convolution_bias_2 = tf.Variable(tf.constant(0.01, shape=[64]))

        feed_forward_weights_1 = tf.Variable(tf.truncated_normal([256, 256], stddev=0.01))
        feed_forward_bias_1 = tf.Variable(tf.constant(0.01, shape=[256]))

        feed_forward_weights_2 = tf.Variable(tf.truncated_normal([256, self.ACTIONS_COUNT], stddev=0.01))
        feed_forward_bias_2 = tf.Variable(tf.constant(0.01, shape=[self.ACTIONS_COUNT]))

        input_layer = tf.placeholder("float", [None, self.SCREEN_WIDTH, self.SCREEN_HEIGHT,
                                               self.STATE_FRAMES])

        hidden_convolutional_layer_1 = tf.nn.relu(
            tf.nn.conv2d(input_layer, convolution_weights_1, strides=[1, 4, 4, 1], padding="SAME") + convolution_bias_1)

        hidden_max_pooling_layer_1 = tf.nn.max_pool(hidden_convolutional_layer_1, ksize=[1, 2, 2, 1],
                                                    strides=[1, 2, 2, 1], padding="SAME")

        hidden_convolutional_layer_2 = tf.nn.relu(
            tf.nn.conv2d(hidden_max_pooling_layer_1, convolution_weights_2, strides=[1, 2, 2, 1],
                         padding="SAME") + convolution_bias_2)

        hidden_max_pooling_layer_2 = tf.nn.max_pool(hidden_convolutional_layer_2, ksize=[1, 2, 2, 1],
                                                    strides=[1, 2, 2, 1], padding="SAME")

        hidden_convolutional_layer_3_flat = tf.reshape(hidden_max_pooling_layer_2, [-1, 256])

        final_hidden_activations = tf.nn.relu(
            tf.matmul(hidden_convolutional_layer_3_flat, feed_forward_weights_1) + feed_forward_bias_1)

        output_layer = tf.matmul(final_hidden_activations, feed_forward_weights_2) + feed_forward_bias_2

        return input_layer, output_layer

    @staticmethod
    def _key_presses_from_action(action_set):
        if action_set[0] == 1:
            return [K_DOWN]
        elif action_set[1] == 1:
            return []
        elif action_set[2] == 1:
            return [K_UP]
        raise Exception("Unexpected action")

    def get_feedback(self):
        from games.half_pong import score

        # get the difference in score between this and the last run
        score_change = (score - self.last_score)
        self.last_score = score

        return float(score_change), score_change == -1

    def start(self):
        super(DeepQHalfPongPlayer, self).start()

        from games.half_pong import run
        run(screen_width=self.SCREEN_WIDTH, screen_height=self.SCREEN_HEIGHT)


if __name__ == '__main__':
    # to see a trained network add the args checkpoint_path="deep_q_half_pong_networks_40x40_8" and
    # playback_mode="True"
    player = DeepQHalfPongPlayer()
    player.start()
