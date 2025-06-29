# CS687 Reinforcement Learning Homeworks and Project
HWs and Project for CS 687 Reinforcement Learning. Each folder has the coding implementation and solved document which consists of both written questions and the report for the coding section

## Homework 1: Policy Performance Estimation in a Discrete MDP

**Description:**  
This assignment focused on analyzing and estimating the expected discounted return \( J(\pi) \) for a given stochastic policy in a custom-defined Markov Decision Process (MDP) with discrete states and actions. The MDP comprised seven states with defined transition probabilities, rewards, and a stochastic policy.

**Key Tasks:**  
- Derived a closed-form, analytic expression for \( J(\pi) \) as a function of the discount factor \(\gamma\), using first principles and probability theory.  
- Implemented a simulation function `runEpisode` to empirically estimate the discounted return by running multiple episodes of the policy interacting with the MDP.  
- Conducted extensive Monte Carlo simulations (150,000 episodes) to estimate \( J(\pi) \) and plotted the convergence of the estimated returns over episodes.  
- Computed and reported statistics such as average return and variance of returns from simulations.  
- Evaluated the estimation for different discount factors \(\gamma \in \{0.25, 0.5, 0.75, 0.99\}\), comparing empirical results against the analytic solution to validate correctness.  
- Performed a deterministic policy search using brute force to identify the policy with the highest expected return at \(\gamma=0.75\), estimating its performance from 350,000 simulated episodes.

**Evaluations:**  
- Compared Monte Carlo estimates of policy returns with analytically computed values to verify accuracy.  
- Analyzed the convergence behavior of return estimates over large numbers of episodes.  
- Identified optimal deterministic policies by exhaustive search and quantified their performance gains.  


---

## Homework 2: Evolution Strategies for the Inverted Pendulum Swing-Up

**Description:**  
This project involved implementing the Inverted Pendulum Swing-Up control task, a continuous-state, continuous-action benchmark problem. The goal was to learn policies that swing the pendulum upright and balance it using torque control. The environment dynamics were modeled deterministically with physics-based equations governing pendulum motion.

**Key Tasks:**  
- Developed the environment from scratch, representing the pendulum state as angle and angular velocity, applying physics equations to simulate transitions.  
- Implemented a deterministic neural network policy with adjustable weights mapping continuous states to continuous actions (torques).  
- Designed and implemented the Evolution Strategies (ES) black-box optimization algorithm for policy search, adapting parameters of the neural network policy through iterative perturbations and weighted updates.  
- Conducted hyperparameter tuning experiments over parameters including the number of perturbations, exploration noise (\(\sigma\)), step size (\(\alpha\)), and neural network architecture (layers and neurons per layer).  
- Ran multiple ES trials (up to 25 runs) for selected hyperparameters, tracking learning progress and plotting average return and variance over iterations.  
- Analyzed the effect of different hyperparameter configurations on learning stability, convergence speed, and final policy performance.  

**Evaluations:**  
- Measured cumulative returns as the primary performance metric, comparing them to near-optimal returns expected for this task.  
- Evaluated stability and variance of learning curves across multiple runs.  
- Discussed challenges and insights regarding the difficulty of policy search in continuous control tasks with ES.

---

## Homework 3: Value Iteration in the Cat-vs-Monsters Domain

**Description:**  
Implemented the Value Iteration algorithm from scratch to find the optimal value function and policy in a 5x5 grid world called the Cat-vs-Monsters domain. The cat moves in a stochastic environment with walls, forbidden furniture, monsters, and food as the terminal state.

**Domain Details:**  
- States represent cat locations on a 5x5 grid.  
- Actions include moving Up, Down, Left, Right with stochastic outcomes (intended direction, confusion right/left, or no movement).  
- Rewards are mostly small negative step costs (-0.05), with large positive reward (+10) for reaching food and large negative reward (-8) for entering monster states.  
- Forbidden furniture blocks movement and paralyzes the cat if entered.  
- Food state is terminal.  
- Discount factor \(\gamma\) primarily 0.925, with experiments using 0.2 and other values.  
- Initial state fixed at the cat’s bed (0,0).  

**Key Tasks and Experiments:**  
- Ran Value Iteration until convergence, presenting final value functions and policies.  
- Compared results and iterations needed for convergence at different discount factors (\(\gamma=0.925\) vs \(\gamma=0.2\)).  
- Modified the environment by adding a catnip state with varying rewards and terminal/non-terminal settings, analyzing effects on optimal policy and values.  
- Investigated how changing \(\gamma\) and catnip reward affected the optimal policy behavior.  

**Evaluations:**  
- Reported final value functions and policies in grid format.  
- Measured iterations required for convergence under different scenarios.  
- Provided qualitative explanations of learned policies and their intuitive behaviors.  


---

## Homework 4: Monte Carlo Policy Evaluation and Control in the Cat-vs-Monsters Domain

**Description:**  
Implemented First-Visit and Every-Visit Monte Carlo algorithms to estimate the value function of the optimal policy in the Cat-vs-Monsters domain, along with Monte Carlo with \(\epsilon\)-soft policies for near-optimal policy improvement. The entire environment and algorithms were built from scratch.

**Key Tasks:**  
- Estimated the optimal value function \( \hat{v} \) using First-Visit and Every-Visit Monte Carlo methods, starting from a uniform initial state distribution (excluding forbidden furniture).  
- Compared sample complexity by reporting the number of trajectories needed to achieve a Max-Norm error of at most 0.1 between estimated and true value functions.  
- Implemented Monte Carlo control with \(\epsilon\)-soft policies to estimate the optimal action-value function \( \hat{q} \) and derive improved policies \( \hat{\pi} \).  
- Tested multiple fixed values of \(\epsilon\) (e.g., 0.2, 0.1, 0.05) and an \(\epsilon\)-decay schedule, evaluating their impact on learning accuracy and convergence.  
- Generated learning curves showing mean squared error between estimates and true optimal value function over iterations.

**Evaluations:**  
- Reported value function estimates for different Monte Carlo variants and \(\epsilon\) values.  
- Plotted mean squared error learning curves to assess convergence rates.  
- Analyzed the effectiveness of fixed versus decaying \(\epsilon\) schedules for policy improvement.  
- Discussed empirical trade-offs in accuracy and sample efficiency among the Monte Carlo approaches implemented.

---
## Homework 5: Temporal Difference Learning and Control Algorithms

**Description:**  
Implemented three Temporal Difference (TD) algorithms—TD-Learning for policy evaluation, and SARSA and Q-Learning for control—from scratch, applying them to the Cat-vs-Monsters domain. Additionally, deployed SARSA and Q-Learning on the Inverted Pendulum Swing-Up domain, which required discretizing continuous states and actions.

### 1. TD-Learning on Cat-vs-Monsters (20 Points)  
- Implemented TD-Learning for policy evaluation with uniform initial state distribution excluding forbidden furniture.  
- Ran algorithm 50 times to analyze convergence.  
- Reported learning rate \(\alpha\) used to balance convergence speed and accuracy.  
- Computed average estimated value function and max-norm error vs. the optimal value function.  
- Reported mean and standard deviation of episodes to convergence.

### 2. SARSA on Cat-vs-Monsters (20 Points)  
- Implemented SARSA for control to estimate optimal action-value function and policy.  
- Made design choices for learning rate, q-function initialization, exploration strategy (e.g., \(\epsilon\)-greedy), and exploration decay.  
- Reported and justified design decisions.  
- Generated two learning curves:  
  - Actions taken vs. episodes completed (average over 20 runs).  
  - Episodes vs. mean squared error of value function estimate against optimal.  
- Presented the final greedy policy learned by SARSA.

### 3. Q-Learning on Cat-vs-Monsters (20 Points)  
- Implemented Q-Learning with similar analysis and reporting as SARSA.  
- Computed value function as max over q-values for policy evaluation in error measurements.  
- Presented final greedy policy.

### 4. SARSA on Inverted Pendulum Swing-Up (20 Points)  
- Discretized continuous states and actions into bins for tabular SARSA implementation.  
- Reported five key design decisions: learning rate, q-init, exploration method, exploration decay, and discretization levels.  
- Generated learning curve of average return (with standard deviation) vs. episodes over 20 runs.

### 5. Q-Learning on Inverted Pendulum Swing-Up (20 Points)  
- Similar approach as SARSA with same analysis and reporting.

### 6. Extra Credit (16 Points)  
- Experimented with varying discretization levels (±25%, ±50%) to study impact on performance and runtime.  
- Tested two optimistic q-function initialization methods with zero exploration to analyze effects on learning and runtime.

---

## Project: Reinforcement Learning Algorithm Implementations and Evaluations

**Project Overview:**  
Implemented and evaluated three reinforcement learning algorithms—REINFORCE with Baseline, True Online SARSA, and Semi-Gradient N-Step SARSA—on classic RL benchmark domains Cartpole and Acrobot. The goal was to study the algorithms’ behaviors and performance characteristics across these environments.

**Details:**  
- **Algorithms and Implementers:**  
  - REINFORCE with Baseline: Yukti Sharma  
  - True Online SARSA: Shreya Birthare  
  - Semi-Gradient N-Step SARSA: Yukti Sharma (Cartpole only)  

- **Environments:**  
  - Cartpole  
  - Acrobot (not applicable for Semi-Gradient N-Step SARSA)

**Contributions:**  
- Implemented algorithms from scratch without using existing RL libraries.  
- Tuned hyperparameters for each algorithm and environment.  
- Presented learning curves and performance metrics demonstrating algorithm effectiveness.  
- Compared convergence rates and policy quality across algorithms and domains.

---


