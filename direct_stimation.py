import gymnasium as gym
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import collections

# Declaración de constantes
SLIPPERY = False
T_MAX = 15
NUM_EPISODES = 20000
GAMMA = 0.95
REWARD_THRESHOLD = 0.9


env = gym.make('Taxi-v3' , render_mode='ansi') #

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
    
def check_improvements():
    reward_test = 0.0
    for i in range(NUM_EPISODES):
        total_reward = 0.0
        state, _ = env.reset()
        for i in range(T_MAX):
            action = agent.select_action(state)
            new_state, new_reward, is_done, truncated, _ = env.step(action)
            total_reward += new_reward
            if is_done: 
                break
            state = new_state
        reward_test += total_reward
    reward_avg = reward_test / NUM_EPISODES
    return reward_avg

def train(agent): 
    rewards = []
    max_diffs = []
    t = 0
    best_reward = 0.0
     
    while best_reward < REWARD_THRESHOLD:
        _, max_diff = agent.value_iteration()
        max_diffs.append(max_diff)
        print("After value iteration, max_diff = " + str(max_diff))
        t += 1
        reward_test = check_improvements()
        rewards.append(reward_test)
               
        if reward_test > best_reward:
            print(f"Best reward updated {reward_test:.2f} at iteration {t}") 
            best_reward = reward_test
    
    return rewards, max_diffs
    
def print_policy(policy):
    visual_help = {0:'v', 1:'^', 2:'>', 3:'<', 4:'P', 5:'D'}
    policy_arrows = [visual_help[x] for x in policy]
    print(np.array(policy_arrows).reshape([-1, 6]))


class DirectEstimationAgent:
    def __init__(self, env, gamma, num_trajectories):
        self.env = env
        self.state, _ = self.env.reset()
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.V = np.zeros(self.env.observation_space.n)
        self.gamma = gamma
        self.num_trajectories = num_trajectories

    def play_n_random_steps(self, count):
        for _ in range(count):
            action = self.env.action_space.sample()
            new_state, reward, is_done, truncated, _ = self.env.step(action)
            self.rewards[(self.state, action, new_state)] = reward
            self.transits[(self.state, action)][new_state] += 1
            if is_done:
                self.state, _ = self.env.reset() 
            else: 
                self.state = new_state

    def calc_action_value(self, state, action):
        target_counts = self.transits[(state, action)]
        total = sum(target_counts.values())
        action_value = 0.0
        for s_, count in target_counts.items():
            r = self.rewards[(state, action, s_)]
            prob = (count / total)
            action_value += prob*(r + self.gamma * self.V[s_])
        return action_value

    def select_action(self, state):
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.calc_action_value(state, action)
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    def value_iteration(self):
        self.play_n_random_steps(self.num_trajectories)
        max_diff = 0
        for state in range(self.env.observation_space.n):
            state_values = [
                self.calc_action_value(state, action)
                for action in range(self.env.action_space.n)
            ]
            new_V = max(state_values)
            diff = abs(new_V - self.V[state])
            if diff > max_diff:
                max_diff = diff
            self.V[state] = new_V
        return self.V, max_diff
    
    def policy(self):   
        policy = np.zeros(env.observation_space.n) 
        for s in range(env.observation_space.n):
            Q_values = [self.calc_action_value(s,a) for a in range(self.env.action_space.n)] 
            policy[s] = np.argmax(np.array(Q_values))        
        return policy




agent = DirectEstimationAgent(env, gamma=GAMMA, num_trajectories=30000)
train(agent)


# Probar agente en el entorno una vez entrenado
is_done = False
rewards = []
sum_r = 0.0
for n_ep in range(NUM_EPISODES):
    state, _ = env.reset()
    #print('Episode: ', n_ep)
    total_reward = 0
    for i in range(T_MAX):
        action = agent.select_action(state)
        state, reward, is_done, truncated, _ = env.step(action)
        total_reward = total_reward + reward
        env.render()
        if is_done:
            break
    rewards.append(total_reward)
    sum_r = sum_r + total_reward
print(f"Average reward is {sum_r/NUM_EPISODES}")
draw_rewards(rewards)