from ActorCritic import CartPoleActorCritic
import tqdm
import statistics
import collections
import tensorflow as tf

if __name__ == '__main__':
    num_actions = 2  # env.action_space.n
    num_hidden_units = 128

    ai = CartPoleActorCritic(num_actions, num_hidden_units)
    # time
    min_episodes_criterion = 200
    max_episodes = 1000
    max_steps_per_episode = 1000

    # Cartpole-v0 is considered solved if average reward is >= 195 over 100
    # consecutive trials
    reward_threshold = 195
    running_reward = 0

    # Discount factor for future rewards
    gamma = 0.99

    # Keep last episodes reward
    episodes_reward: collections.deque = collections.deque(
        maxlen=min_episodes_criterion)

    with tqdm.trange(max_episodes) as t:
        for i in t:
            initial_state = tf.constant(ai.env.reset(), dtype=tf.float32)
            episode_reward = int(ai.train_step(gamma, max_steps_per_episode))

            episodes_reward.append(episode_reward)
            running_reward = statistics.mean(episodes_reward)

            t.set_description(f'Episode {i}')
            t.set_postfix(
                episode_reward=episode_reward, running_reward=running_reward)

            # Show average episode reward every 10 episodes
            if i % 10 == 0:
                pass  # print(f'Episode {i}: average reward: {avg_reward}')

            if running_reward > reward_threshold and i >= min_episodes_criterion:
                break

            print(
                f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')

    ai.render_episode(max_steps_per_episode)