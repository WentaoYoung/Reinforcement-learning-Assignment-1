import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class AssetAllocationEnvironment:
    """Discrete-time asset allocation environment with unconstrained shorting and leverage.
    
    Attributes:
        initial_wealth (float): Starting wealth value
        T (int): Total number of time steps
        r (float): Risk-free rate per period
        a (float): Upside return of risky asset
        b (float): Downside return of risky asset 
        p (float): Probability of upside return
        risk_aversion (float): Absolute risk aversion coefficient
        wealth (float): Current wealth state
        time_step (int): Current time step
    
    Methods:
        reset: Reset environment to initial state
        step: Execute one time step transition
    """
    
    def __init__(self, initial_wealth=1.0, T=10, r=0.01, a=1.2, b=0.8, p=0.5):
        
        if not (a > r > b):
            raise ValueError("Must Satisfy a>r>b!")
        
        self.initial_wealth = initial_wealth
        self.T = T
        self.r = r
        self.a = a
        self.b = b
        self.p = p
        self.risk_aversion = a
        
        self.wealth = initial_wealth
        self.time_step = 0
        
    def reset(self):
        """Reset environment to initial state.
        
        Returns:
            tuple: (current_time_step, current_wealth)
        """ 

        self.wealth = self.initial_wealth
        self.time_step = 0
        return (self.time_step, self.wealth)
    
    def step(self, action):
        """Execute one time step transition.
        
        Args:
            action (float): Amount invested in risky asset
            
        Returns:
            tuple: (next_state, reward, done)
        """
        next_wealth = (self.wealth - action) * (1 + self.r) + action * (
            1 + (self.a if np.random.random() < self.p else self.b))
        
        self.wealth = next_wealth
        self.time_step += 1
        
        reward = -np.exp(-self.risk_aversion * next_wealth)/self.risk_aversion if self.time_step == self.T else 0
        return (self.time_step, self.wealth), reward, (self.time_step >= self.T)

class QLearningAgent:
    """Q-learning agent for discrete-time asset allocation problem.
    
    Attributes:
        env (AssetAllocationEnvironment): Associated environment
        q_table (np.ndarray): Q-value table [time, wealth_bins, action_bins]
        epsilon (float): Exploration rate
        learning_rate (float): TD learning rate
        discount_factor (float): Future reward discount factor
    
    Methods:
        discretize_wealth: Convert continuous wealth to discrete bins
        discretize_action: Convert continuous action to discrete bins
        get_action: Select action using ε-greedy policy
        update: Update Q-table using TD learning
        train: Main training loop
    """
    def __init__(self, env, learning_rate=0.1, discount_factor=1.0, epsilon=0.1, epsilon_decay=0.999,
                 min_epsilon=0.01, wealth_discretization=20, action_discretization=20,
                 action_multiplier_range=(-2.0, 3.0)):
        
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.action_multiplier_range = action_multiplier_range

        # Discretization setup
        self.wealth_discretization = wealth_discretization
        self.action_discretization = action_discretization
        self.q_table = np.zeros((env.T + 1, wealth_discretization, action_discretization))
        self.max_wealth = env.initial_wealth * 2  # Dynamic wealth scaling

        # Convergence monitoring
        self.delta_history = []
        self.convergence_window = []
        self.analytical_mse = []

        self.a = env.a
        self.b = env.b
        self.p = env.p
        self.r = env.r
        self.T = env.T
        self.risk_aversion = env.risk_aversion

    def discretize_wealth(self, wealth):
        """Discretize continuous wealth value into bins.
        
        Args:
            wealth (float): Current wealth value
            
        Returns:
            int: Discrete wealth bin index
        """
        scaled = min(1.0, abs(wealth) / self.max_wealth)
        return np.clip(int(scaled * self.wealth_discretization), 0, self.wealth_discretization-1)

    def discretize_action(self, action, wealth):
        """Discretize continuous action value into bins.
        
        Args:
            action (float): Raw action value
            wealth (float): Current wealth value
            
        Returns:
            int: Discrete action bin index
        """
        if wealth == 0:
            return 0
        multiplier = action / wealth
        multiplier = np.clip(multiplier, *self.action_multiplier_range)
        normalized = (multiplier - self.action_multiplier_range[0]) / (
            self.action_multiplier_range[1] - self.action_multiplier_range[0])
        return int(normalized * self.action_discretization)

    def action_from_index(self, action_idx, wealth):
        """Convert discrete action index back to continuous action value.
        
        Args:
            action_idx (int): Discrete action index from Q-table
            wealth (float): Current wealth value for action scaling
            
        Returns:
            float: Actual action value (investment amount in risky asset)
        """      
        multiplier = self.action_multiplier_range[0] + (
            (self.action_multiplier_range[1] - self.action_multiplier_range[0]) * 
            (action_idx / self.action_discretization))
        return multiplier * wealth

    def get_action(self, state):
        """Select action using ε-greedy policy.
        
        Args:
            state (tuple): Current (time, wealth) state
            
        Returns:
            float: Selected action value
        """

        t, wealth = state
        wealth_idx = self.discretize_wealth(wealth)
        
        if np.random.rand() < self.epsilon:
            return self.action_from_index(np.random.randint(self.action_discretization), wealth)
        return self.action_from_index(np.argmax(self.q_table[t, wealth_idx]), wealth)

    def update(self, state, action, reward, next_state, done):
        """Update Q-table using temporal difference learning.
        
        Args:
            state: Current state tuple
            action: Executed action
            reward: Observed reward
            next_state: Next state tuple
            done: Terminal state flag
        """

        t, wealth = state
        next_t, next_wealth = next_state
        
        wealth_idx = self.discretize_wealth(wealth)
        action_idx = self.discretize_action(action, wealth)
        next_wealth_idx = self.discretize_wealth(next_wealth)

        # Update Q-table using TD learning
        target = reward + (0 if done else self.discount_factor * 
                          np.max(self.q_table[next_t, next_wealth_idx]))
        self.q_table[t, wealth_idx, action_idx] += self.learning_rate * (
            target - self.q_table[t, wealth_idx, action_idx])

    def train(self, max_episodes=10000, conv_threshold=1e-5, conv_window=5, check_analytical_every=100):
        """Execute Q-learning training loop.
        
        Args:
            max_episodes: Maximum training iterations
            conv_threshold: Convergence threshold
            conv_window: Convergence check window size
            check_analytical_every: Validation frequency
        """
        self.convergence_window = []
        for episode in tqdm(range(max_episodes)):
            prev_q = np.copy(self.q_table)
            
            state = self.env.reset()
            while not self.env.step(state[1])[2]:  # Run episode
                action = self.get_action(state)
                next_state, reward, done = self.env.step(action)
                self.update(state, action, reward, next_state, done)
                state = next_state

            # Improved convergence checking
            delta = np.max(np.abs(self.q_table - prev_q))
            self.delta_history.append(delta)
            self.convergence_window.append(delta)
            if len(self.convergence_window) > conv_window:
                self.convergence_window.pop(0)
                
            # Validate against analytical solution periodically
            if episode % check_analytical_every == 0:
                self.analytical_mse.append(self._calculate_analytical_mse())

            # Check sustained convergence
            if len(self.convergence_window) == conv_window and all(d < conv_threshold for d in self.convergence_window):
                print(f"\nConverged at episode {episode} with sustained delta < {conv_threshold}")
                break


            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def _compute_K(self, t):
        """Computes analytical solution constant K for validation.
        
        Args:
            t (int): Current time step
            
        Returns:
            float: Analytical constant K
        """
        numerator = (self.a - self.r) * self.p
        denominator = (self.r - self.b) * (1 - self.p)
        ln_term = np.log(numerator / denominator)
        
        term1 = -((self.a - self.r)/(self.a - self.b)) * ln_term
        term2 = -((self.b - self.r)/(self.a - self.b)) * ln_term
        
        return self.p * np.exp(term1) + (1 - self.p) * np.exp(term2)

    def _compute_analytical_q(self):
        """Computes analytical Q-values for validation.
        
        Returns:
            np.ndarray: Array of analytical Q-values matching q_table dimensions
        """
        Q_analytical = np.zeros_like(self.q_table)
        for t in range(self.env.T):
            K = self._compute_K(t)
            for w_idx in range(self.wealth_discretization):
                wealth = (w_idx / self.wealth_discretization) * self.max_wealth
                for x_idx in range(self.action_discretization):
                    action = self.action_from_index(x_idx, wealth)
                    exponent = -self.risk_aversion * ((1 + self.r)**(self.T - t)) * wealth
                    term1 = np.exp(-self.risk_aversion * ((1 + self.r)**(self.T - t - 1)) * action * (self.a - self.r))
                    term2 = np.exp(-self.risk_aversion * ((1 + self.r)**(self.T - t - 1)) * action * (self.b - self.r))
                    Q_val = -((K**(self.T - t - 1) / self.risk_aversion)) * np.exp(exponent) * \
                            (self.p * term1 + (1 - self.p) * term2)
                    Q_analytical[t, w_idx, x_idx] = Q_val
        return Q_analytical

    def _calculate_analytical_mse(self):
        """Calculates validation MSE between current Q-values and analytical solution.
        
        Returns:
            float: Mean squared error between learned and analytical Q-values
        """
        analytical_Q = self._compute_analytical_q()
        return np.mean((self.q_table[:self.T] - analytical_Q[:self.T])**2)
    
    def plot_results(self):
        """Here we present two graphs. First is the maximum differnce Q-values in each episoids for convergence visualization.
        Second is the MSE between the values of Q function computed by TD method and the ones of the analytical solution. 
        """
        plt.figure(figsize=(15,5))
        
        plt.subplot(1,2,1)
        plt.semilogy(self.delta_history)
        plt.title('Q-Value Convergence')
        plt.xlabel('Episodes')
        plt.ylabel('Max ΔQ')
        plt.grid(True)
        
        plt.subplot(1,2,2)
        plt.plot(np.arange(len(self.analytical_mse))*100, self.analytical_mse)
        plt.title('Analytical Validation')
        plt.xlabel('Episodes')
        plt.ylabel('MSE')
        plt.yscale('log')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

# Usage Example
if __name__ == "__main__":
    env = AssetAllocationEnvironment(
        initial_wealth=1.0,
        T=10,
        r=0.03,
        a=0.15,
        b=-0.06,
        p=0.6
    )
    
    agent = QLearningAgent(
        env,
        action_multiplier_range=(-2.0, 3.0),  # Allow shorting up to 200% and leverage up to 300%
        wealth_discretization=100,
        action_discretization=50,
        learning_rate=0.01,
        epsilon_decay=0.995
    )
    
    agent.train(
        max_episodes=10000,
        conv_threshold=1e-5,
        conv_window=1  # Require 5 consecutive episodes below threshold
    )
    
    agent.plot_results()