import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt

NUM_RUNS = 20
NUM_EPISODES = 2000
MAX_STEPS = 2000
ALPHA = 2e-4
GAMMA = 1.0
EPSILON_START = 1.0
EPSILON_END = 0.0
DECAY_EVERY = 50
DECAY_FACTOR = 0.5
EVAL_EPISODES = 20
PRINT_EVERY = 50


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim1=16, hidden_dim2=8):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim2, 4)
        self.fc4 = nn.Linear(4, action_dim)

    def forward(self, state):
        x = self.fc1(state)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = nn.ReLU()(x)
        q_values = self.fc4(x)
        return q_values


GAMMA = 0.925
ALPHA = 0.05
EPSILON = 0.1
NUM_ROWS, NUM_COLS = 5, 5
FORBIDDEN_FURN = [(2, 1), (2, 2), (2, 3), (3, 2)]
FOOD = [(4, 4)]
MONSTERS = [(0, 3), (4, 1)]
ACTIONS = ["AU", "AD", "AR", "AL"]
PROB_INTENDED, PROB_CONFUSED_RIGHT, PROB_CONFUSED_LEFT, PROB_SLEEPY_STAY = 0.7, 0.12, 0.12, 0.06

ACTION_MAP = {
    "AU": "↑",
    "AD": "↓",
    "AR": "→",
    "AL": "←"
}

ACTION_MAP_INDEX = {
    0: "AU",
    1: "AD",
    2: "AR",
    3: "AL"
}

OPTIMAL_VALUE_FUNCTION = {
    (0, 0): 2.6638, (0, 1): 2.9969, (0, 2): 2.8117, (0, 3): 3.6671, (0, 4): 4.8497,
    (1, 0): 2.9713, (1, 1): 3.5101, (1, 2): 4.0819, (1, 3): 4.8497, (1, 4): 7.1648,
    (2, 0): 2.5936, (2, 1): 0.0000, (2, 2): 0.0000, (2, 3): 0.0000, (2, 4): 8.4687,
    (3, 0): 2.0992, (3, 1): 1.0849, (3, 2): 0.0000, (3, 3): 8.6097, (3, 4): 9.5269,
    (4, 0): 1.0849, (4, 1): 4.9465, (4, 2): 8.4687, (4, 3): 9.5269, (4, 4): 0.0000,
}

def move_specified(row, col, action):
    if action == "AU":
        next_state = max(row - 1, 0), col
    elif action == "AD":
        next_state = min(row + 1, NUM_ROWS - 1), col
    elif action == "AR":
        next_state = row, min(col + 1, NUM_COLS - 1)
    else:
        next_state = row, max(col - 1, 0)
    return (row, col) if next_state in FORBIDDEN_FURN else next_state


def confused_move_right(row, col, action):
    if action == "AU":
        next_state = row, min(col + 1, NUM_COLS - 1)
    elif action == "AD":
        next_state = row, max(col - 1, 0)
    elif action == "AR":
        next_state = min(row + 1, NUM_ROWS - 1), col
    else:
        next_state = max(row - 1, 0), col
    return (row, col) if next_state in FORBIDDEN_FURN else next_state


def confused_move_left(row, col, action):
    if action == "AU":
        next_state = row, max(col - 1, 0)
    elif action == "AD":
        next_state = row, min(col + 1, NUM_COLS - 1)
    elif action == "AR":
        next_state = max(row - 1, 0), col
    else:
        next_state = min(row + 1, NUM_ROWS - 1), col
    return (row, col) if next_state in FORBIDDEN_FURN else next_state


def sleepy_stay(row, col):
    return row, col


def select_next_state(row, col, action):
    intended_next = move_specified(row, col, action)
    right_next = confused_move_right(row, col, action)
    left_next = confused_move_left(row, col, action)
    stay_next = sleepy_stay(row, col)
    transitions = [intended_next, right_next, left_next, stay_next]
    probabilities = [PROB_INTENDED, PROB_CONFUSED_RIGHT, PROB_CONFUSED_LEFT, PROB_SLEEPY_STAY]
    next_state = random.choices(transitions, weights=probabilities, k=1)[0]
    return next_state


def epsilon_greedy_policy(q_network, state, epsilon, action_dim):
    actions = [0, 1, 2, 3]
    if random.random() < epsilon:
        return random.randint(0, 4)
    else:
        # state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = []
        for action in actions:
            state_action = one_hot_encode_state(state, NUM_ROWS, NUM_COLS, action)
            tensor = torch.tensor(state_action, dtype=torch.float32)
            q_value = q_network(tensor)
            q_values.append(q_value.item())
        max_q = np.max(q_values)
        # Find all actions with max Q-value to handle ties
        optimal_actions = [action for action, q_value in enumerate(q_values) if q_value == max_q]
        return random.choice(optimal_actions)


def display_policy(policy):
    print("Greedy Policy:")
    for row in policy:
        print("  ".join(row))


def rewards(next_state):
    if next_state in FOOD:
        return 10
    elif next_state in MONSTERS:
        return -8
    else:
        return -0.05


def compute_v_from_q(q_network):
    value = np.zeros((NUM_ROWS, NUM_COLS))
    actions = [0, 1, 2, 3]
    for row in range(NUM_ROWS):
        for col in range(NUM_COLS):
            state = (row, col)
            if state in FORBIDDEN_FURN or state in FOOD:
                value[row, col] = 0.0
            else:
                q_values = []
                for action in actions:
                    state_action = one_hot_encode_state(state, NUM_ROWS, NUM_COLS, action)
                    state_tensor = torch.tensor(state_action, dtype=torch.float32)
                    q_value = q_network(state_tensor)
                   # print(q_value,row,col,action)
                    q_values.append(q_value.item())
                    # print(state_tensor,q_values, row, col)
                value[row][col] = np.nanmax(q_values)
    return value


def compute_mse(value, optimal_v):
    return np.mean((value - optimal_v) ** 2)


def get_epsilon(episode, epsilon_start, epsilon_end, decay_every, decay_factor):
    n_decays = (episode - 1) // decay_every
    epsilon = epsilon_start * (decay_factor ** n_decays)
    return max(epsilon, epsilon_end)


def evaluate_policy(q_network, env, episodes=10):
    q_network.eval()
    total_rewards = []
    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        done = False
        truncated = False
        ep_reward = 0
        while not (done or truncated):
            action = epsilon_greedy_policy(q_network, state, epsilon=0.0, action_dim=env.action_space.n)
            state, reward, done, truncated, _ = env.step(action)
            ep_reward += reward
        total_rewards.append(ep_reward)
    avg_reward = np.mean(total_rewards)
    print(f"\nAverage Reward over {episodes} Evaluation Episodes: {avg_reward:.2f}")
    q_network.train()


def one_hot_encode_state(state, num_rows, num_cols, action):
    index = (state[0] * num_cols + state[1]) * 4 + action
    one_hot = np.zeros(num_rows * num_cols * 4, dtype=np.float32)
    if state[0]==4 and state[1]==4:
        return one_hot
    one_hot[index] = 1
    return one_hot


def run_single_training_run(run_id, env, state_dim, action_dim):
    q_network = QNetwork(state_dim, action_dim)
    optimizer = torch.optim.Adam(q_network.parameters(), lr=ALPHA)
    criterion = nn.MSELoss()

    returns_per_episode = []
    actions = [0, 1, 2, 3]

    for episode in range(1, NUM_EPISODES + 1):
        state = random.choice([(i, j) for i in range(NUM_ROWS) for j in range(NUM_COLS) if
                               (i, j) not in FORBIDDEN_FURN and (i, j) not in FOOD])
        i = state[0]
        j = state[1]
        # state = one_hot_encode_state(state,NUM_ROWS, NUM_COLS)

        epsilon = get_epsilon(episode, EPSILON_START, EPSILON_END, DECAY_EVERY, DECAY_FACTOR)

        action = epsilon_greedy_policy(q_network, state, epsilon, action_dim)

        total_reward = 0

        while True:
            if i == 4 and j == 4:
                break
            # print(ACTION_MAP_INDEX.get(action))

            next_state = select_next_state(i, j, ACTION_MAP_INDEX.get(action))
            next_action = epsilon_greedy_policy(q_network, next_state, epsilon, action_dim)
            reward = rewards(next_state)
            next_row, next_col = next_state
            # next_state = one_hot_encode_state(next_state, NUM_ROWS, NUM_COLSm)

            epsilon = 0.1

            total_reward += reward
            state_action_encoded = torch.tensor(one_hot_encode_state(state, NUM_ROWS, NUM_COLS, action))
            q_value = q_network(state_action_encoded)
            if next_row == 4 and next_col == 4:
                target = reward
                # state = one_hot_encode_state(state, NUM_ROWS, NUM_COLS)
                # state_tensor = torch.FloatTensor(state).unsqueeze(0)
                # print(q_value)
                delta = target - q_value
                loss = torch.square(delta)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                # next_action = epsilon_greedy_policy(q_network, next_state, epsilon, action_dim)
                # next_state = one_hot_encode_state(next_state, NUM_ROWS, NUM_COLS)
                # next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                state_action_encoded_next = torch.tensor(
                    one_hot_encode_state(next_state, NUM_ROWS, NUM_COLS, next_action))
                next_q = q_network(state_action_encoded_next)
                target = reward + GAMMA * next_q.detach()
                # state = one_hot_encode_state(state, NUM_ROWS, NUM_COLS)

                delta = target - q_value
                loss = torch.square(delta)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            #print(state,next_state)
            state = next_state
            i = next_row
            j = next_col
            action = next_action

        returns_per_episode.append(total_reward)
        value = compute_v_from_q(q_network)
        print(value)
        optimal_v = np.array(list(OPTIMAL_VALUE_FUNCTION.values())).reshape(NUM_ROWS, NUM_COLS)

        mse = compute_mse(value, optimal_v)

        if episode % PRINT_EVERY == 0:
            print(
                f"Run {run_id + 1}, Episode {episode}/{NUM_EPISODES} completed. Return: {total_reward:.2f}, MSE: {mse}")

    return returns_per_episode


def main():
    env = gym.make('CartPole-v1')

    state_dim = 25 * 4
    action_dim = 1

    print("Starting Training Runs...\n")

    all_runs_returns = []

    for run in range(NUM_RUNS):
        print(f"Starting Run {run + 1}/{NUM_RUNS}")
        returns = run_single_training_run(run, env, state_dim, action_dim)
        all_runs_returns.append(returns)

    env.close()

    all_runs_returns = np.array(all_runs_returns)

    mean_returns = np.mean(all_runs_returns, axis=0)
    std_returns = np.std(all_runs_returns, axis=0)

    episodes = np.arange(1, NUM_EPISODES + 1)

    plt.figure(figsize=(12, 8))
    plt.plot(episodes, mean_returns, label='Average Return', color='blue')
    plt.fill_between(episodes, mean_returns - std_returns, mean_returns + std_returns, color='blue', alpha=0.2,
                     label='Standard Deviation')
    plt.title('Learning Curve: Average Return vs. Number of Episodes')
    plt.xlabel('Number of Episodes')
    plt.ylabel('Average Return')
    plt.legend()
    plt.savefig("episodic_semi_graident_CVM_v3.png")
    plt.grid(True)


if __name__ == '__main__':
    main()
