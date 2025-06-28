import gym
import numpy as np
import matplotlib.pyplot as plt

N = 20
NUM_EPISODES = 1000
GAMMA = 1
LAMBDA = 0.08
ALPHA = 0.5
# EPSILON = 0.5
BINS = [15, 15, 15, 15]

def discretize_state(state):
    # print(state)
    position, velocity, angle, angular_velocity = state
    position_bin = np.digitize(position, np.linspace(-4.8, 4.8, BINS[0])) - 1
    velocity_bin = np.digitize(velocity, np.linspace(-5, 5, BINS[1])) - 1
    angle_bin = np.digitize(angle, np.linspace(-0.418, 0.418, BINS[2])) - 1
    angular_velocity_bin = np.digitize(angular_velocity, np.linspace(-5, 5, BINS[3])) - 1
    arr = np.array([position_bin, velocity_bin, angle_bin, angular_velocity_bin])
    return arr

def getFeatureVector(state, action, num_actions):
    total_bins = np.prod(BINS)
    feature_vector_size = total_bins * num_actions
    feature_vector = np.zeros(feature_vector_size)
    # print(state)
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
    env = gym.make('CartPole-v1')
    all_rewards = []
    for i in range(N):
        epsilon = 0.5
        print(f"Run {i+1}")
        # print("epsilon: ", epsilon)
        # print(env.observation_space.shape[0])
        # w = np.zeros(env.observation_space.shape[0])
        # w = np.zeros(observation_space.shape[0] + env.action_space.n)
        w = np.zeros(15*15*15*15*2)
        episode_rewards = []
        for j in range(NUM_EPISODES):
            if(j!=0 and j%50==0):
                epsilon/=2
                # print("epsilon: ", epsilon)
            observation, info = env.reset()
            num_actions = env.action_space.n
            # print(observation) # [pos, vel, angle, angular vel]
            # initialize S
            initial_state = observation
            initial_state = discretize_state(initial_state)
            # initialize A
            action = selectAction(initial_state, w, num_actions, epsilon)
            # print("action: ", action)
            # feature vector x
            x_vec = getFeatureVector(initial_state, action, num_actions)
            # initalize Z
            z = np.zeros_like(w)
            # q_old
            q_old = 0
            total_reward = 0
            # loop for each step of episode
            while True:
                next_state, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
                next_state = discretize_state(next_state)
                # select next A'
                next_action = selectAction(next_state, w, num_actions, epsilon)
                # x' = x(S', A')
                x_vec_next = getFeatureVector(next_state, next_action, num_actions)
                # Q = w.t * x
                Q = np.dot(w, x_vec)
                # Q' = w.t * x'
                Q_next = np.dot(w, x_vec_next)
                
                # delta
                # if terminated:
                #     delta = reward - Q 
                # else:
                #     delta = reward + GAMMA*Q_next - Q
                delta = reward + GAMMA*Q_next - Q

                # z
                z = (GAMMA*LAMBDA*z) + (1 - ALPHA*GAMMA*LAMBDA* z.T * x_vec)*x_vec
                # w
                w = w + ALPHA*(delta + Q - q_old)*z - ALPHA*(Q - q_old)*x_vec
                # q_old = Q'
                q_old = Q_next
                # x = x'
                x_vec = x_vec_next
                # A = A'
                action = next_action

                total_reward += reward

                # if terminated or truncated:
                #     break
            # if(j%20==0):
                # print("Reward: ", total_reward)
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

    plt.title('Learning Curve: True Online SARSA (Lambda) Algorithm on CartPole')
    plt.xlabel('Number of Episodes')
    plt.ylabel('Average Return')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__=='__main__':
    plot_learning_curve()