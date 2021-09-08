from PIL import Image
import gym
import numpy as np
from tensorflow.keras import layers
from pathlib import Path
from typing import Tuple, Union, List
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
import tensorflow as tf
import time
import os
from enum import Enum
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Use CPU instead of GPU device
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Show fatal error only


class CPVer(Enum):
    V0 = 'CartPole-v0'
    V1 = 'CartPole-v1'


class CartPoleActorCritic(tf.keras.Model):
    def __init__(self, num_actions: int, num_hidden_units: int, cartPoleVersion: CPVer = CPVer.V0, seed: int = 42, learning_rate=0.01):
        """Initialize."""
        super().__init__()

        # Initialize env and seed for experiment reproducibility
        self._env = gym.make(cartPoleVersion.value)
        self._env.seed(seed)
        tf.random.set_seed(seed)
        np.random.seed(seed)

        # Small epsilon value for stabilizing division operations
        self._eps = np.finfo(np.float32).eps.item()

        # Loss function and optimizer
        self._huber_loss = Huber(reduction=tf.keras.losses.Reduction.SUM)
        self._optimizer = Adam(learning_rate)

        # Init Actor-Critic network
        self._common = layers.Dense(num_hidden_units, activation="relu")
        self._actor = layers.Dense(num_actions)
        self._critic = layers.Dense(1)

    def _forward(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """ Forward propagation. Calculate actor and critic values. """
        x = self._common(inputs)
        return self._actor(x), self._critic(x)

    def _backward(self, tape: tf.GradientTape, episode_result: List[tf.Tensor]):
        """ Backward propagation. Calculate loss and update gradients """
        action_probs, values, returns = episode_result
        loss = self._compute_loss(action_probs, values, returns)

        # Compute the gradients from the loss
        grads = tape.gradient(loss, self.trainable_variables)

        # Apply the gradients to the model's parameters
        self._optimizer.apply_gradients(zip(grads, self.trainable_variables))

    def _compute_loss(self, action_probs: tf.Tensor, values: tf.Tensor, returns: tf.Tensor) -> tf.Tensor:
        """Computes the combined actor-critic loss."""

        delta = returns - values

        action_log_probs = tf.math.log(action_probs)
        actor_loss = -tf.math.reduce_sum(action_log_probs * delta)

        critic_loss = self._huber_loss(values, returns)

        return actor_loss + critic_loss

    def _py_env_step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns state, reward and done flag given an action."""
        state, reward, done, _ = self._env.step(action)
        return (np.array(state, 'float32'), np.array(reward, 'int32'), np.array(done, 'int32'))

    def _tf_env_step(self, action: tf.Tensor) -> Tuple[tf.Tensor]:
        return tf.numpy_function(self._py_env_step, [action], [tf.float32, tf.int32, tf.int32])

    def _get_expected_return(self, rewards: tf.Tensor, gamma: float, standardize: bool = True) -> tf.Tensor:
        """Compute expected returns per timestep."""

        n = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=n)

        # Start from the end of `rewards` and accumulate reward sums
        # into the `returns` array
        rewards = tf.cast(rewards[::-1], dtype=tf.float32)
        discounted_sum: tf.Tensor = tf.constant(0.0)
        discounted_sum_shape = discounted_sum.shape
        for i in tf.range(n):
            discounted_sum = rewards[i] + gamma * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)
        returns = returns.stack()[::-1]

        if standardize:
            returns = ((returns - tf.math.reduce_mean(returns)) /
                       (tf.math.reduce_std(returns) + self._eps))

        return returns

    def _run_episode(self, max_steps: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Runs a single episode to collect training data."""

        action_probs: tf.TensorArray = tf.TensorArray(
            dtype=tf.float32, size=0, dynamic_size=True)
        values: tf.TensorArray = tf.TensorArray(
            dtype=tf.float32, size=0, dynamic_size=True)
        rewards: tf.TensorArray = tf.TensorArray(
            dtype=tf.int32, size=0, dynamic_size=True)

        initial_state: tf.Tensor = tf.constant(
            self._env.reset(), dtype=tf.float32)
        initial_state_shape = initial_state.shape
        state = initial_state

        for t in tf.range(max_steps):
            # Convert state into a batched tensor (batch size = 1)
            state = tf.expand_dims(state, 0)

            # Run the model and to get action probabilities and critic value
            actor_action, critic_reward = self._forward(state)

            # Store critic values
            values = values.write(t, tf.squeeze(critic_reward))

            # Sample next action from the action probability distribution
            action = tf.random.categorical(actor_action, 1)[0, 0]
            action_probs_t = tf.nn.softmax(actor_action)

            # Store log probability of the action chosen
            action_probs = action_probs.write(t, action_probs_t[0, action])

            # Apply action to the environment to get next state and reward
            state, reward, done = self._tf_env_step(action)
            state.set_shape(initial_state_shape)

            # Store reward
            rewards = rewards.write(t, reward)

            if tf.cast(done, tf.bool):
                break

        return action_probs.stack(), values.stack(), rewards.stack()

    @tf.function
    def _tf_train_step(self, gamma: float, max_steps_per_episode: int) -> tf.Tensor:
        """Runs a model training step."""

        with tf.GradientTape() as tape:
            # Run the model for one episode to collect training data
            action_probs, values, rewards = self._run_episode(
                max_steps_per_episode)

            # Calculate expected returns
            returns = self._get_expected_return(rewards, gamma)

            # Convert training data to appropriate TF tensor shapes
            episode_result: Tuple[tf.Tensor] = [
                tf.expand_dims(x, 1) for x in [action_probs, values, returns]]

            # Update the model's parameters
            self._backward(tape, episode_result)

        episode_reward = tf.math.reduce_sum(rewards)
        return episode_reward

    def train_step(self, gamma: float, max_steps_per_episode: int) -> tf.Tensor:
        self._env.reset()
        return self._tf_train_step(gamma, max_steps_per_episode)

    def render_episode(self, max_steps: int, sleep_sec: float = 0.1):
        """ Render episode with (trained) model """

        state = tf.constant(self._env.reset(), dtype=tf.float32)
        images = []
        for _ in range(max_steps):
            # Render screen every steps per sleep_sec seconds
            if sleep_sec > 0.0:
                time.sleep(sleep_sec)

            # Render screen and save it to array
            screen = self._env.render(mode='rgb_array')
            images.append(Image.fromarray(screen))

            state = tf.expand_dims(state, 0)
            action_probs, _ = self._forward(state)
            action = np.argmax(np.squeeze(action_probs))

            state, _, done, _ = self._env.step(action)
            state = tf.constant(state, dtype=tf.float32)

            if done:
                break

        return images

    def episode_into_gif(self, filename: Union[str, Path], max_steps: int = 200, sleep_sec: float = 0.1):
        images = self.render_episode(max_steps, sleep_sec)
        # loop=0: loop forever, duration=1: play each frame for 1ms
        images[0].save(str(filename), save_all=True,
                       append_images=images, loop=0, duration=1)
