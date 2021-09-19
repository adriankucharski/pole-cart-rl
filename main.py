from ActorCritic import CPVer, CartPoleActorCritic
import tqdm
import statistics
from collections import deque
import tensorflow as tf
from matplotlib import pyplot as plt

if __name__ == '__main__':
    num_actions = 2  # env.action_space.n
    num_hidden_units = 128

    ai = CartPoleActorCritic(num_actions, num_hidden_units, CPVer.V0)
    # Training goals
    min_episodes = 100
    max_episodes = 1000

    # Cartpole V0 is terminated after up to 200 steps, V1 after 500
    max_steps_per_episode = 200

    # Cartpole-v0 is considered solved if average reward is >= 195 over 100 consecutive trials
    # Cartpole-v0 is considered solved if average reward is >= 475 over 100 consecutive trials
    reward_threshold = 195

    # Discount factor for future rewards
    gamma = 0.99

    # Keep last episodes reward
    episodes_reward = deque(maxlen=min_episodes)

    # Reward array for statistic
    rewards = []
    episodes = []
    rendered = []
    with tqdm.tqdm(range(max_episodes)) as t:
        for i in t:
            episode_reward = int(ai.train_step(gamma, max_steps_per_episode))
            episodes_reward.append(episode_reward)

            running_reward = statistics.mean(episodes_reward)
            rewards.append(running_reward)
            episodes.append(episode_reward)

            t.set_description(f'Episode {i}')
            t.set_postfix(episode_reward=episode_reward,
                          running_reward=running_reward)

            # If True we got it!
            if running_reward > reward_threshold and i >= min_episodes:
                ai.save_model('model.pickle')
                break
        
    plt.plot(rewards)
    plt.show()

    plt.plot(episodes)
    plt.show()


    # ai.load_model('model.pickle')
    rendered[0].save(str('game.gif'), save_all=True,
                     append_images=rendered, loop=0, duration=20)

    # ai.render_episode(max_steps_per_episode, sleep_sec=0.0025)
    # ai.episode_into_gif('game.gif')
