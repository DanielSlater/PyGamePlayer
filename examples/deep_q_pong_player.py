# This is heavily based off https://github.com/asrivat1/DeepLearningVideoGames
import random
from collections import deque
from pong_player import PongPlayer
import tensorflow as tf
import numpy as np
import cv2
from pygame.constants import K_DOWN, K_UP


class DeepQPongPlayer(PongPlayer):
    ACTIONS_COUNT = 3  # number of valid actions. In this case up, still and down
    FUTURE_REWARD_DISCOUNT = 0.99  # decay rate of past observations
    OBSERVATION_STEPS = 1000.  # 500000.  # time steps to observe before training
    EXPLORE_STEPS = 2000000.  # frames over which to anneal epsilon
    INITIAL_RANDOM_ACTION_PROB = 1.0  # starting chance of an action being random
    FINAL_RANDOM_ACTION_PROB = 0.05  # final chance of an action being random
    REPLAY_MEMORY = 590000  # number of previous transitions to remember
    MINI_BATCH_SIZE = 32  # size of mini batches
    FRAMES_PER_ACTION = 4  # only select a new action every x frames, repeat prev for others
    STATE_FRAMES = 4  # number of frames to store in the state
    RESIZED_SCREEN_X, RESIZED_SCREEN_Y = (80, 80)
    OBS_LAST_STATE_INDEX, OBS_ACTION_INDEX, OBS_REWARD_INDEX, OBS_CURRENT_STATE_INDEX, OBS_TERMINAL_INDEX = range(5)

    def __init__(self):
        super(DeepQPongPlayer, self).__init__()
        self._session = tf.InteractiveSession()
        self._input_layer, self._output_layer, hidden_activations = DeepQPongPlayer._create_network()

        self._action = tf.placeholder("float", [None, self.ACTIONS_COUNT])
        self._output = tf.placeholder("float", [None])
        readout_action = tf.reduce_sum(tf.mul(self._output_layer, self._action), reduction_indices=1)
        cost = tf.reduce_mean(tf.square(self._output - readout_action))
        self._train_operation = tf.train.AdamOptimizer(1e-6).minimize(cost)

        # store the previous observations in replay memory
        self._previous_observations = deque()

        # set the first action to do nothing
        self._last_action = np.zeros(self.ACTIONS_COUNT)
        self._last_action[1] = 1

        self._last_state = None
        self._probability_of_random_action = self.INITIAL_RANDOM_ACTION_PROB
        self._time = 0
        self._frames_since_last_action = 0

        self._session.run(tf.initialize_all_variables())

    def get_keys_pressed(self, screen_array, reward, terminal):
        # scale down game image
        screen_resized_grayscaled = cv2.cvtColor(cv2.resize(screen_array,
                                                            (self.RESIZED_SCREEN_X, self.RESIZED_SCREEN_Y)),
                                                 cv2.COLOR_BGR2GRAY)

        # set the grayscale to have values in the 0.0 to 1.0 range
        ret, screen_resized_grayscaled = cv2.threshold(screen_resized_grayscaled, 1, 255, cv2.THRESH_BINARY)

        # first frame must be handled differently
        if self._last_state is None:
            # the _last_state will contain the image data from the last self.STATE_FRAMES(4) frames
            self._last_state = np.stack(tuple(screen_resized_grayscaled for _ in range(self.STATE_FRAMES)), axis=2)
            return DeepQPongPlayer._key_presses_from_action(self._last_action)

        screen_resized_grayscaled = np.reshape(screen_resized_grayscaled,
                                               (self.RESIZED_SCREEN_X, self.RESIZED_SCREEN_Y, 1))
        current_state = np.append(screen_resized_grayscaled, self._last_state[:, :, 1:], axis=2)

        # store the transition in previous_observations
        self._previous_observations.append((self._last_state, self._last_action, reward, current_state, terminal))

        if len(self._previous_observations) > self.REPLAY_MEMORY:
            self._previous_observations.popleft()

        self._frames_since_last_action += 1

        if self._frames_since_last_action >= self.FRAMES_PER_ACTION:
            # select a new action
            self._frames_since_last_action = 0

            # only train if done observing
            if self._time > self.OBSERVATION_STEPS:
                self._train()

            # update the old values
            self._last_state = current_state
            self._time += 1

            self._last_action = self._choose_next_action()

            # gradually reduce the probability of a random action
            if self._probability_of_random_action > self.FINAL_RANDOM_ACTION_PROB \
                    and self._time > self.OBSERVATION_STEPS:
                self._probability_of_random_action -= \
                    (self.INITIAL_RANDOM_ACTION_PROB - self.FINAL_RANDOM_ACTION_PROB) / self.EXPLORE_STEPS

        return DeepQPongPlayer._key_presses_from_action(self._last_action)

    def _choose_next_action(self):
        new_action = np.zeros([self.ACTIONS_COUNT])

        if self._time <= self.OBSERVATION_STEPS or random.random() <= self._probability_of_random_action:
            # choose an action randomly
            action_index = random.randrange(self.ACTIONS_COUNT)
        else:
            # choose an action given our last state
            readout_t = self._output_layer.eval(feed_dict={self._input_layer: [self._last_state]})[0]
            action_index = np.argmax(readout_t)

        new_action[action_index] = 1
        return new_action

    def _train(self):
        # sample a mini_batch to train on
        mini_batch = random.sample(self._previous_observations, self.MINI_BATCH_SIZE)
        # get the batch variables
        previous_states = [d[self.OBS_LAST_STATE_INDEX] for d in mini_batch]
        actions = [d[self.OBS_ACTION_INDEX] for d in mini_batch]
        rewards = [d[self.OBS_REWARD_INDEX] for d in mini_batch]
        current_states = [d[self.OBS_CURRENT_STATE_INDEX] for d in mini_batch]
        agents_expected_reward = []
        # this gives us the agents expected reward for each action we might
        agents_reward_per_action = self._output_layer.eval(feed_dict={self._input_layer: current_states})
        for i in range(len(mini_batch)):
            if mini_batch[i][self.OBS_TERMINAL_INDEX]:
                # this was a terminal frame so need so scale future reward...
                agents_expected_reward.append(rewards[i])
            else:
                agents_expected_reward.append(
                    rewards[i] + self.FUTURE_REWARD_DISCOUNT * np.max(agents_reward_per_action[i]))

        # learn that these actions in these states lead to this reward
        self._train_operation.run(feed_dict={
            self._input_layer: previous_states,
            self._action: actions,
            self._output: agents_expected_reward})

    @staticmethod
    def _create_network():
        # network weights
        convolution_weights_1 = DeepQPongPlayer._weight_variable([8, 8, DeepQPongPlayer.STATE_FRAMES, 32])
        convolution_bias_1 = DeepQPongPlayer._bias_variable([32])

        convolution_weights_2 = DeepQPongPlayer._weight_variable([4, 4, 32, 64])
        convolution_bias_2 = DeepQPongPlayer._bias_variable([64])

        convolution_weights_3 = DeepQPongPlayer._weight_variable([3, 3, 64, 64])
        convolution_bias_3 = DeepQPongPlayer._bias_variable([64])

        feed_forward_weights_1 = DeepQPongPlayer._weight_variable([1600, 512])
        feed_forward_bias_1 = DeepQPongPlayer._bias_variable([512])

        feed_forward_weights_2 = DeepQPongPlayer._weight_variable([512, DeepQPongPlayer.ACTIONS_COUNT])
        feed_forward_bias_2 = DeepQPongPlayer._bias_variable([DeepQPongPlayer.ACTIONS_COUNT])

        # input layer
        input_placeholder = tf.placeholder("float", [None, DeepQPongPlayer.RESIZED_SCREEN_X, DeepQPongPlayer.RESIZED_SCREEN_Y,
                                                     DeepQPongPlayer.STATE_FRAMES])

        # hidden layers
        hidden_convolutional_layer_1 = tf.nn.relu(
            DeepQPongPlayer._convolution_2d(input_placeholder, convolution_weights_1, 4) + convolution_bias_1)

        hidden_max_pooling_layer = DeepQPongPlayer._max_pool_2x2(hidden_convolutional_layer_1)

        hidden_convolutional_layer_2 = tf.nn.relu(
            DeepQPongPlayer._convolution_2d(hidden_max_pooling_layer, convolution_weights_2, 2) + convolution_bias_2)

        hidden_convolutional_layer_3 = tf.nn.relu(
            DeepQPongPlayer._convolution_2d(hidden_convolutional_layer_2, convolution_weights_3,
                                            1) + convolution_bias_3)

        hidden_convolutional_layer_3_flat = tf.reshape(hidden_convolutional_layer_3, [-1, 1600])

        final_hidden_activations = tf.nn.relu(
            tf.matmul(hidden_convolutional_layer_3_flat, feed_forward_weights_1) + feed_forward_bias_1)

        # readout layer
        readout = tf.matmul(final_hidden_activations, feed_forward_weights_2) + feed_forward_bias_2

        return input_placeholder, readout, final_hidden_activations

    @staticmethod
    def _key_presses_from_action(action_set):
        if action_set[0] == 1:
            return [K_DOWN]
        elif action_set[1] == 1:
            return []
        elif action_set[2] == 1:
            return [K_UP]
        raise Exception("Unexpected action")

    @staticmethod
    def _weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    @staticmethod
    def _bias_variable(shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    @staticmethod
    def _convolution_2d(x, weights, stride):
        return tf.nn.conv2d(x, weights, strides=[1, stride, stride, 1], padding="SAME")

    @staticmethod
    def _max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


if __name__ == '__main__':
    player = DeepQPongPlayer()
    player.playing = True

    # importing pong will start the game playing
    import games.pong
