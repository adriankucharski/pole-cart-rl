from ActorCritic import CPVer, CartPoleActorCritic
import tqdm
import statistics
from collections import deque
import tensorflow as tf
from matplotlib import pyplot as plt

if __name__ == '__main__':
    max_steps_per_episode = 200
    num_actions = 2  # env.action_space.n
    num_hidden_units = 128
    ai = CartPoleActorCritic(num_actions, num_hidden_units, CPVer.V0)
    ai.load_model('model.pickle')

    ai.render_episode(max_steps_per_episode, sleep_sec=0.0025)
    # ai.episode_into_gif('game.gif', text="sadgf")
