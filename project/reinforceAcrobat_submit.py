import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.distributions as dist

NUM_RUNS = 20
NUM_EPISODES = 1000
ALPHA_THETA = 0.001
ALPHA_W = 0.001
GAMMA = 0.99


#Different networks ere tested by changing the values here of network
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size1=128, hidden_size2=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, action_dim)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        logits = self.fc3(x)
        return logits

    def get_action_probs(self, state):
        logits = self.forward(state)
        return torch.softmax(logits, dim=-1)


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_size1=64, hidden_size2=32):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        value = self.fc3(x)
        return value

def generate_episode(env, policy_net):
    states = []
    actions = []
    rewards = []

    state, _ = env.reset()
    done = False

    while not done:
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_probs = policy_net.get_action_probs(state_t).squeeze(0).detach().numpy()
        action = np.random.choice(len(action_probs), p=action_probs)

        states.append(state)
        actions.append(action)

        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rewards.append(reward)

    return states, actions, rewards

def compute_returns(rewards):
    n = len(rewards)
    returns = []

    gamma_values = [GAMMA ** i for i in range(n)]

    for t, reward in enumerate(rewards):
        G_t = sum(gamma_values[k - t] * rewards[k] for k in range(t, n))
        returns.append(G_t)

    return returns

def run_episode(env, policy_net, value_net):
    states, actions, rewards = generate_episode(env, policy_net)
    returns = compute_returns(rewards)

    states_t = torch.tensor(states, dtype=torch.float32)
    actions_t = torch.tensor(actions, dtype=torch.int64)
    returns_t = torch.tensor(returns, dtype=torch.float32).unsqueeze(-1)

    values = value_net(states_t)
    advantages = returns_t - values.detach()

    action_probs = policy_net.get_action_probs(states_t)
    action_log_probs = torch.log(action_probs.gather(1, actions_t.unsqueeze(1)))

    policy_loss = (action_log_probs * advantages).mean()

    value_loss = nn.MSELoss()(values, returns_t)

    policy_net.zero_grad()
    policy_loss.backward()


    with torch.no_grad():
        for param in policy_net.parameters():
            param += ALPHA_THETA * param.grad
    policy_net.zero_grad()

    value_net.zero_grad()
    value_loss.backward()
    with torch.no_grad():
        for param in value_net.parameters():
            param -= ALPHA_W * param.grad
    value_net.zero_grad()

    total_return = sum(rewards)
    return total_return

def reinforce(env):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = PolicyNetwork(state_dim, action_dim)
    value_net = ValueNetwork(state_dim)

    returns_per_episode = []
    for episode in range(NUM_EPISODES):
        G = run_episode(env, policy_net, value_net)
        returns_per_episode.append(G)

        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}/{NUM_EPISODES}: Return={G:.2f}")

    return returns_per_episode

def plot_learning_curve(avg_returns, std_returns):
    plt.figure(figsize=(10, 7))
    plt.plot(avg_returns, label='Average Return (20 runs)', color='blue')
    plt.fill_between(range(NUM_EPISODES),
                     avg_returns - std_returns,
                     avg_returns + std_returns,
                     alpha=0.2, color='blue', label='Standard Deviation')
    plt.xlabel('Episodes')
    plt.ylabel('Average Return')
    plt.title('REINFORCE with Baseline : Acrobot')
    plt.legend()
    plt.grid(True)
    plt.savefig("ACROBOT_LEARNING_CURVE_2.png")
    print("Experiment complete.")

def main():
    all_returns = []

    for run_i in range(NUM_RUNS):
        print(f"Starting run {run_i + 1}/{NUM_RUNS}...")
        env = gym.make("Acrobot-v1", render_mode=None)
        returns = reinforce(env)
        all_returns.append(returns)
        env.close()
        print(f"Completed run {run_i + 1}/{NUM_RUNS}.")

    all_returns = np.array(all_returns)
    avg_returns = np.mean(all_returns, axis=0)
    std_returns = np.std(all_returns, axis=0)

    print("All runs completed. Generating plot...")

    plot_learning_curve(avg_returns, std_returns)

    print("Completed")

if __name__ == "__main__":
    main()
