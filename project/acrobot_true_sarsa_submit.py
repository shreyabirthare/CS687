import gym
import numpy as np
import matplotlib.pyplot as plt
import math

N = 20
NUM_EPISODES = 1000
GAMMA = 1
LAMBDA = 0.08
ALPHA = 0.5
BINS = [6,6,6,6,10,20]

def discretize_state(state):
    cos_theta1, sin_theta1, cos_theta2, sin_theta2, theta_dot1, theta_dot2 = state
    theta1_bin = np.digitize(cos_theta1, np.linspace(-1, 1, BINS[0])) - 1
    theta2_bin = np.digitize(cos_theta2, np.linspace(-1, 1, BINS[1])) - 1
    theta1_sin_bin = np.digitize(sin_theta1, np.linspace(-1, 1, BINS[0])) - 1
    theta2_sin_bin = np.digitize(sin_theta2, np.linspace(-1, 1, BINS[1])) - 1
    theta_dot1_bin = np.digitize(theta_dot1, np.linspace(-4 * math.pi, 4 * math.pi, BINS[2])) - 1
    theta_dot2_bin = np.digitize(theta_dot2, np.linspace(-9 * math.pi, 9 * math.pi, BINS[3])) - 1
    arr = np.array([theta1_bin, theta1_sin_bin, theta2_bin, theta2_sin_bin, theta_dot1_bin, theta_dot2_bin])
    return arr


def getFeatureVector(state, action, num_actions):
    total_bins = np.prod(BINS)
    feature_vector_size = total_bins * num_actions
    feature_vector = np.zeros(feature_vector_size)
    index = np.ravel_multi_index(state, BINS)
    feature_index = index * num_actions + action
    feature_vector[feature_index] = 1
    return feature_vector


def selectAction(state, w, num_actions, epsilon):
    q_values = np.zeros(num_actions)
    for action in range(num_actions):
        x_vec = getFeatureVector(state, action, num_actions)
        q_values[action] = np.dot(w, x_vec)
    
    if np.random.rand() < epsilon:
        return np.random.randint(num_actions)
    else:
        return np.argmax(q_values)

def trueSARSA():
    env = gym.make('Acrobot-v1')
    all_rewards = []
    for i in range(N):
        epsilon = 0.5
        print(f"Run {i+1}")
        w = np.zeros(np.prod(BINS) * 3)
        episode_rewards = []
        for j in range(NUM_EPISODES):
            # print(f"Episode {j}")
            if j != 0 and j % 50 == 0:
                epsilon /= 2
            observation, info = env.reset()
            num_actions = env.action_space.n
            initial_state = discretize_state(observation)
            action = selectAction(initial_state, w, num_actions, epsilon)
            x_vec = getFeatureVector(initial_state, action, num_actions)
            z = np.zeros_like(w)
            q_old = 0
            total_reward = 0

            while True:
                next_state, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
                next_state = discretize_state(next_state)
                next_action = selectAction(next_state, w, num_actions, epsilon)
                x_vec_next = getFeatureVector(next_state, next_action, num_actions)

                Q = np.dot(w, x_vec)
                Q_next = np.dot(w, x_vec_next)
                
                delta = reward + GAMMA * Q_next - Q
                z = (GAMMA * LAMBDA * z) + (1 - ALPHA * GAMMA * LAMBDA * np.dot(z.T, x_vec)) * x_vec

                w = w + ALPHA * (delta + Q - q_old) * z - ALPHA * (Q - q_old) * x_vec
                q_old = Q_next
                x_vec = x_vec_next
                action = next_action

                total_reward += reward
                # if(j%10==0):
                # print("reward: ", total_reward)
            episode_rewards.append(total_reward)
        all_rewards.append(episode_rewards)
    
    env.close()

    all_rewards = np.array(all_rewards)
    avg_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)

    return avg_rewards, std_rewards

def plot_learning_curve():
    avg_rewards, std_rewards = trueSARSA()

    plt.figure(figsize=(10, 6))
    plt.plot(avg_rewards, label='Average Return', color='blue')
    plt.fill_between(range(NUM_EPISODES), avg_rewards - std_rewards, avg_rewards + std_rewards, 
                     color='blue', alpha=0.2, label='Standard Deviation')

    plt.title('Learning Curve: True Online SARSA (lambda) Algo on Acrobot')
    plt.xlabel('Number of Episodes')
    plt.ylabel('Average Return')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    plot_learning_curve()
