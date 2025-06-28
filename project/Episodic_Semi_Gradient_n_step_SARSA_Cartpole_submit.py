import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt

NUM_RUNS = 20
NUM_EPISODES = 1000
MAX_STEPS = 2000
ALPHA = 2e-3
GAMMA = 1.0
EPSILON_START = 1.0
EPSILON_END = 0.0
DECAY_EVERY = 50
DECAY_FACTOR = 0.5
EVAL_EPISODES = 20
PRINT_EVERY = 50

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim1=128, hidden_dim2=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim2, action_dim)

    def forward(self, state):
        x = self.fc1(state)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        q_values = self.fc3(x)
        return q_values


def epsilon_greedy_policy(q_network, state, epsilon, action_dim):
    if random.random() < epsilon:
        return random.randint(0, action_dim - 1)
    else:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = q_network(state_tensor)
        max_q = torch.max(q_values)
        optimal_actions = (q_values == max_q).nonzero(as_tuple=True)[1]
        optimal_actions = optimal_actions.tolist()
        return random.choice(optimal_actions)


def get_epsilon(episode, epsilon_start, epsilon_end, decay_every, decay_factor):
    n_decays = (episode - 1) // decay_every
    epsilon = epsilon_start * (decay_factor ** n_decays)
    return max(epsilon, epsilon_end)


# def evaluate_policy(q_network, env, episodes=10):
#     q_network.eval()
#     total_rewards = []
#     for ep in range(1, episodes + 1):
#         state, _ = env.reset()
#         done = False
#         truncated = False
#         ep_reward = 0
#         while not (done or truncated):
#             action = epsilon_greedy_policy(q_network, state, epsilon=0.0, action_dim=env.action_space.n)
#             state, reward, done, truncated, _ = env.step(action)
#             ep_reward += reward
#         total_rewards.append(ep_reward)
#     avg_reward = np.mean(total_rewards)
#     print(f"\nAverage Reward over {episodes} Evaluation Episodes: {avg_reward:.2f}")
#     q_network.train()


def run_n_step_SARSA(run_id, env, state_dim, action_dim):
    q_network = QNetwork(state_dim, action_dim)
    MSELossCriteria = nn.MSELoss()

    returns_per_episode = []

    optimizer = optim.Adam(q_network.parameters(), lr=ALPHA)


    for episode in range(1, NUM_EPISODES + 1):
        state, _ = env.reset()
        epsilon = get_epsilon(episode, EPSILON_START, EPSILON_END, DECAY_EVERY, DECAY_FACTOR)
        action = epsilon_greedy_policy(q_network, state, epsilon, action_dim)

        total_reward = 0

        for step in range(MAX_STEPS):
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward

            if done or truncated:
                target = torch.FloatTensor([reward])
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_value = q_network(state_tensor)[0, action]
                loss = MSELossCriteria(q_value, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                break
            else:
                next_action = epsilon_greedy_policy(q_network, next_state, epsilon, action_dim)

                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                with torch.no_grad():
                    next_q_values = q_network(next_state_tensor)
                    next_q = next_q_values[0, next_action]
                target = torch.FloatTensor([reward + GAMMA * next_q])

                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_value = q_network(state_tensor)[0, action]

                loss = MSELossCriteria(q_value, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                state = next_state
                action = next_action

        returns_per_episode.append(total_reward)

        if episode % PRINT_EVERY == 0:
            print(f"Run {run_id + 1}, Episode {episode}/{NUM_EPISODES} completed. Return: {total_reward:.2f}")

    return returns_per_episode

def plot_learning_curve(episodes,mean_returns, std_returns):
    plt.figure(figsize=(12, 8))
    plt.plot(episodes, mean_returns, label='Average Return', color='blue')
    plt.fill_between(episodes, mean_returns - std_returns, mean_returns + std_returns, color='blue', alpha=0.2,
                     label='Standard Deviation')
    plt.title('Learning Curve: Average Return vs. Number of Episodes')
    plt.xlabel('Number of Episodes')
    plt.ylabel('Average Return')
    plt.legend()
    plt.savefig("episodic_semi_graident_v4.png")
    plt.grid(True)

def main():
    env = gym.make('CartPole-v1')

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print("Starting Training Runs...\n")

    all_runs_returns = []

    for runs in range(NUM_RUNS):
        print(f"Starting Run {runs + 1}/{NUM_RUNS}")
        returns = run_n_step_SARSA(runs, env, state_dim, action_dim)
        all_runs_returns.append(returns)

    env.close()

    all_runs_returns = np.array(all_runs_returns)

    mean_returns = np.mean(all_runs_returns, axis=0)
    std_returns = np.std(all_runs_returns, axis=0)

    episodes = np.arange(1, NUM_EPISODES + 1)

    print("runs completed. plotting curves")

    plot_learning_curve(episodes,mean_returns, std_returns)

    print("Completed")


if __name__ == '__main__':
    main()
