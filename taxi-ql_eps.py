import gymnasium as gym
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from gymnasium import Wrapper

# Declaración de constantes
SLIPPERY = False
T_MAX = 2500
NUM_EPISODES = 10000
GAMMA = 0.95
LEARNING_RATE = 0.5
EPSILON = 0.3

def test_episode(agent, env):
    env.reset()
    is_done = False
    t = 0

    while not is_done:
        action = agent.select_action()
        state, reward, is_done, truncated, info = env.step(action)
        t += 1
    return state, reward, is_done, truncated, info

def draw_rewards(rewards):
    data = pd.DataFrame({'Episode': range(1, len(rewards) + 1), 'Reward': rewards})
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Episode', y='Reward', data=data)

    plt.title('Rewards Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.tight_layout()

    plt.show()
    
def print_policy(policy):
    visual_help = {0:'v', 1:'^', 2:'>', 3:'<', 4:'P', 5:'D'}
    policy_arrows = [visual_help[x] for x in policy]
    print(np.array(policy_arrows).reshape([-1, 4]))


class QLearningAgent:
    def __init__(self, env, gamma, learning_rate, epsilon, t_max):
        self.env = env
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.t_max = t_max

    def select_action(self, state, training=True):
        if training and random.random() <= self.epsilon:
            return np.random.choice(self.env.action_space.n)
        else:
            return np.argmax(self.Q[state,])
        
    def update_Q(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.Q[next_state,])
        td_target = reward + self.gamma * self.Q[next_state, best_next_action]
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.learning_rate * td_error
        
    def learn_from_episode(self):
        state, _ = env.reset()
        total_reward = 0
        for i in range(self.t_max):
            action = self.select_action(state)
            new_state, new_reward, is_done, truncated, _ = self.env.step(action)
            total_reward += new_reward
            self.update_Q(state, action, new_reward, new_state)
            if is_done:
                break
            state = new_state
        return total_reward

    def policy(self):
        policy = np.zeros(env.observation_space.n) 
        for s in range(env.observation_space.n):
            policy[s] = np.argmax(np.array(self.Q[s]))        
        return policy


env = gym.make('Taxi-v3', render_mode='ansi')

agent = QLearningAgent(env, gamma=GAMMA, learning_rate=LEARNING_RATE, epsilon=EPSILON, t_max=T_MAX)
rewards = []
dr = (EPSILON-0.001) / NUM_EPISODES
for i in range(NUM_EPISODES):
    reward = agent.learn_from_episode()
    rewards.append(reward)
    EPSILON = EPSILON - dr
    if (EPSILON < 0.001):
        EPSILON = 0.001
    print(f"EPSILON = {EPSILON}")
    print("New reward: " + str(reward))
draw_rewards(rewards)

policy = agent.policy()
print_policy(policy)


# Probar agente en el entorno una vez entrenado
is_done = False
rewards = []
sum_r = 0.0
for n_ep in range(NUM_EPISODES):
    state, _ = env.reset()
    print('Episode: ', n_ep)
    total_reward = 0
    for i in range(T_MAX):
        action = agent.select_action(state, training=False)
        state, reward, is_done, truncated, _ = env.step(action)
        total_reward = total_reward + reward
        env.render()
        if is_done:
            break
    rewards.append(total_reward)
    sum_r = sum_r + total_reward
print(f"Average reward is {sum_r/NUM_EPISODES}")
draw_rewards(rewards)