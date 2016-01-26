import random
from collections import deque
from pong_player import PongPlayer
import tensorflow as tf
import numpy as np
import cv2
from pygame.constants import K_DOWN, K_UP


class DeepQPongPlayer(PongPlayer):
    ACTIONS = 3  # number of valid actions
    GAMMA = 0.99  # decay rate of past observations
    OBSERVE = 500000.  # timesteps to observe before training
    EXPLORE = 2000000.  # frames over which to anneal epsilon
    FINAL_EPSILON = 0.05  # final value of epsilon
    INITIAL_EPSILON = 1.0  # starting value of epsilon
    REPLAY_MEMORY = 590000  # number of previous transitions to remember
    BATCH = 32  # size of minibatch
    K = 2  # only select an action every Kth frame, repeat prev for others

    def __init__(self):
        super(DeepQPongPlayer, self).__init__()
        self._session = tf.InteractiveSession()
        self._input_layer, self._output_layer, hidden_activations = self._create_network()

        self._action = tf.placeholder("float", [None, self.ACTIONS])
        self._output = tf.placeholder("float", [None])
        readout_action = tf.reduce_sum(tf.mul(self._output_layer, self._action), reduction_indices=1)
        cost = tf.reduce_mean(tf.square(self._output - readout_action))
        self._train_operation = tf.train.AdamOptimizer(1e-6).minimize(cost)

        # store the previous observations in replay memory
        self._previous_observations = deque()

        # set the first action to do nothing
        self._last_action = np.zeros(self.ACTIONS)
        self._last_action[1] = 1

        self._last_state = None
        self._epsilon = self.INITIAL_EPSILON
        self._time = 0
        self._frames_since_last_action = 0

        self._session.run(tf.initialize_all_variables())

    def get_keys_pressed(self, screen_array, reward, terminal):
        # scale down game image
        screen_resized_grayscaled = cv2.cvtColor(cv2.resize(screen_array, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, screen_resized_grayscaled = cv2.threshold(screen_resized_grayscaled, 1, 255, cv2.THRESH_BINARY)

        # first frame must be handled differently
        if self._last_state is None:
            self._last_state = np.stack((screen_resized_grayscaled, screen_resized_grayscaled,
                                         screen_resized_grayscaled, screen_resized_grayscaled), axis=2)
            return DeepQPongPlayer._key_presses_from_action(self._last_action)

        screen_resized_grayscaled = np.reshape(screen_resized_grayscaled, (80, 80, 1))
        current_state = np.append(screen_resized_grayscaled, self._last_state[:, :, 1:], axis=2)

        # store the transition in previous_observations
        self._previous_observations.append((self._last_state, self._last_action, reward, current_state, terminal))

        if len(self._previous_observations) > self.REPLAY_MEMORY:
            self._previous_observations.popleft()

        self._frames_since_last_action += 1

        if self._frames_since_last_action < self.K:
            # just repeat the last action until we hit another recalc point
            return DeepQPongPlayer._key_presses_from_action(self._last_action)
        else:
            self._frames_since_last_action = 0

            # only train if done observing
            if self._time > self.OBSERVE:
                self._train()

            # update the old values
            self._last_state = current_state
            self._time += 1

            self._last_action = self._get_new_action()

            # scale down epsilon
            if self._epsilon > self.FINAL_EPSILON and self._time > self.OBSERVE:
                self._epsilon -= (self.INITIAL_EPSILON - self.FINAL_EPSILON) / self.EXPLORE

            return DeepQPongPlayer._key_presses_from_action(self._last_action)

    def _get_new_action(self):
        readout_t = self._output_layer.eval(feed_dict={self._input_layer: [self._last_state]})[0]
        new_action = np.zeros([self.ACTIONS])
        if self._time <= self.OBSERVE or random.random() <= self._epsilon:
            action_index = random.randrange(self.ACTIONS)
            new_action[action_index] = 1
        else:
            action_index = np.argmax(readout_t)
            new_action[action_index] = 1

        return new_action

    def _train(self):
        # sample a minibatch to train on
        minibatch = random.sample(self._previous_observations, self.BATCH)
        # get the batch variables
        s_j_batch = [d[0] for d in minibatch]
        a_batch = [d[1] for d in minibatch]
        r_batch = [d[2] for d in minibatch]
        s_j1_batch = [d[3] for d in minibatch]
        y_batch = []
        readout_j1_batch = self._output_layer.eval(feed_dict={self._input_layer: s_j1_batch})
        for i in range(0, len(minibatch)):
            # if terminal only equals reward
            if minibatch[i][4]:
                y_batch.append(r_batch[i])
            else:
                y_batch.append(r_batch[i] + self.GAMMA * np.max(readout_j1_batch[i]))

        # perform gradient step
        self._train_operation.run(feed_dict={
            self._output: y_batch,
            self._action: a_batch,
            self._input_layer: s_j_batch})

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
    def _conv2d(x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

    @staticmethod
    def _max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    def _create_network(self):
        # network weights
        convolution_weights_1 = DeepQPongPlayer._weight_variable([8, 8, 4, 32])
        convolution_bias_1 = DeepQPongPlayer._bias_variable([32])

        convolution_weights_2 = DeepQPongPlayer._weight_variable([4, 4, 32, 64])
        convolution_bias_2 = DeepQPongPlayer._bias_variable([64])

        convolution_weights_3 = DeepQPongPlayer._weight_variable([3, 3, 64, 64])
        convolution_bias_3 = DeepQPongPlayer._bias_variable([64])

        feed_forward_weights_1 = DeepQPongPlayer._weight_variable([1600, 512])
        feed_forward_bias_1 = DeepQPongPlayer._bias_variable([512])

        feed_forward_weights_2 = DeepQPongPlayer._weight_variable([512, self.ACTIONS])
        feed_forward_bias_2 = DeepQPongPlayer._bias_variable([self.ACTIONS])

        # input layer
        input = tf.placeholder("float", [None, 80, 80, 4])

        # hidden layers
        h_conv1 = tf.nn.relu(DeepQPongPlayer._conv2d(input, convolution_weights_1, 4) + convolution_bias_1)
        h_pool1 = DeepQPongPlayer._max_pool_2x2(h_conv1)

        h_conv2 = tf.nn.relu(DeepQPongPlayer._conv2d(h_pool1, convolution_weights_2, 2) + convolution_bias_2)
        # h_pool2 = DeepQPongPlayer._max_pool_2x2(h_conv2)

        h_conv3 = tf.nn.relu(DeepQPongPlayer._conv2d(h_conv2, convolution_weights_3, 1) + convolution_bias_3)
        # h_pool3 = DeepQPongPlayer._max_pool_2x2(h_conv3)

        # h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
        h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

        final_hidden_activations = tf.nn.relu(tf.matmul(h_conv3_flat, feed_forward_weights_1) + feed_forward_bias_1)

        # readout layer
        readout = tf.matmul(final_hidden_activations, feed_forward_weights_2) + feed_forward_bias_2

        return input, readout, final_hidden_activations


if __name__ == '__main__':
    player = DeepQPongPlayer()
    player.playing = True

    # importing pong will start the game playing
    import games.pong
