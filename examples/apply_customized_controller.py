import traceback

from simglucose.simulation.user_interface import simulate
from simglucose.controller.base import Controller, Action
import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt



class MyController(Controller):



    def __init__(self, gym_id, init_state=None):
        super().__init__(init_state)
        self.gym_id = gym_id
        env = gym.make(self.gym_id)

        self.num_states = env.observation_space.shape[0]
        print("Size of State Space ->  {}".format(self.num_states))
        self.num_actions = env.action_space.shape[0]
        print("Size of Action Space ->  {}".format(self.num_actions))

        self.upper_bound = env.action_space.high[0]
        self.lower_bound = env.action_space.low[0]

        print("Max Value of Action ->  {}".format(self.upper_bound))
        print("Min Value of Action ->  {}".format(self.lower_bound))

        self.actor_model = self.get_actor()
        self.critic_model = self.get_critic()

        self.target_actor = self.get_actor()
        self.target_critic = self.get_critic()

        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())

        self.buffer = Buffer(50000, 64,self)

        self.ep_reward_list = []
        self.avg_reward_list = []
        self.episodic_reward = 0
    def policy(self, observation, reward, done, **info):
        '''
        Every controller must have this implementation!
        ----
        Inputs:
        observation - a namedtuple defined in simglucose.simulation.env. For
                      now, it only has one entry: blood glucose level measured
                      by CGM sensor.
        reward      - current reward returned by environment
        done        - True, game over. False, game continues
        info        - additional information as key word arguments,
                      simglucose.simulation.env.T1DSimEnv returns patient_name
                      and sample_time
        ----
        Output:
        action - a namedtuple defined at the beginning of this file. The
                 controller action contains two entries: basal, bolus
        '''

        action = self.internal_policy(observation, ou_noise)


        return action
    def learn(self, prev_state,action,reward,state):
        self.buffer.record((prev_state, action, reward, state))
        self.episodic_reward += reward

        self.buffer.learn()
        try:
            self.update_target(self.target_actor.variables, self.actor_model.variables, tau)
        except:
            traceback.print_exc()

        self.update_target(self.target_critic.variables, self.critic_model.variables, tau)

        self.ep_reward_list.append(self.episodic_reward)

        # Mean of last 40 episodes
        avg_reward = np.mean(self.ep_reward_list[-40:])

        self.avg_reward_list.append(avg_reward)

    def reset(self):
        '''
        Reset the controller state to inital state, must be implemented
        '''
        self.state = self.init_state

    # This update target parameters slowly
    # Based on rate `tau`, which is much less than one.
    @tf.function
    def update_target(self,target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    """
    Here we define the Actor and Critic networks. These are basic Dense models
    with `ReLU` activation.
    Note: We need the initialization for last layer of the Actor to be between
    `-0.003` and `0.003` as this prevents us from getting `1` or `-1` output values in
    the initial stages, which would squash our gradients to zero,
    as we use the `tanh` activation.
    """

    def get_actor(self):
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(self.num_states,))
        out = layers.Dense(256, activation="relu")(inputs)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)

        # Our upper bound is 2.0 for Pendulum.
        outputs = outputs * self.upper_bound
        model = tf.keras.Model(inputs, outputs)
        return model

    def get_critic(self):
        # State as input
        state_input = layers.Input(shape=(self.num_states))
        state_out = layers.Dense(16, activation="relu")(state_input)
        state_out = layers.Dense(32, activation="relu")(state_out)

        # Action as input
        action_input = layers.Input(shape=(self.num_actions))
        action_out = layers.Dense(32, activation="relu")(action_input)

        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(256, activation="relu")(concat)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)

        return model

    """
    `policy()` returns an action sampled from our Actor network plus some noise for
    exploration.
    """

    def internal_policy(self,state, noise_object):
        try:
            temp = self.actor_model(state)
            sampled_actions = tf.squeeze(temp)
        except:
            traceback.print_exc()

        noise = noise_object()
        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + noise

        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, self.lower_bound, self.upper_bound)

        return [np.squeeze(legal_action)]


if __name__ == '__main__':
    ctrller = MyController(0)
    simulate(controller=ctrller)


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64,Controller=None):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size
        self.Controller:MyController = Controller
        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, self.Controller.num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, self.Controller.num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, self.Controller.num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.

    def update(
            self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = self.Controller.target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * self.Controller.target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = self.Controller.critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.Controller.critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, self.Controller.critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self.Controller.actor_model(state_batch, training=True)
            critic_value = self.Controller.critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.Controller.actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, self.Controller.actor_model.trainable_variables)
        )
# We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)


std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

# Making the weights equal initially

# Learning rate for actor-critic models
critic_lr = 0.9
actor_lr = 0.8

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

total_episodes = 100
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005