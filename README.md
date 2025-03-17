# Q-Learning Asset Allocation

This project implements a Q-learning based approach for solving dynamic asset allocation problems in a discrete-time investment environment. The implementation includes a simulation environment for asset allocation and a Q-learning agent that learns optimal investment strategies.

## Project Structure

The project consists of the following main files:

- `code.py`: Contains the core implementation of the Q-learning algorithm, including the AssetAllocationEnvironment and QLearningAgent classes
- `report.ipynb`: Jupyter notebook containing the full experimental report with detailed analysis and results
- `test_code.py`: Unit tests for verifying the functionality of both environment and agent implementations

## Requirements

- Python 3.6+
- NumPy
- Matplotlib
- tqdm

## Installation

1. Clone the repository
2. Install required packages:
   ```bash
   pip install numpy matplotlib tqdm
   ```

## Usage

### Basic Example

```python
# Create environment
env = AssetAllocationEnvironment(
    initial_wealth=1.0,
    T=10,
    r=0.03,  # risk-free rate
    a=0.15,  # upside return
    b=-0.06, # downside return
    p=0.6    # probability of upside
)

# Initialize agent
agent = QLearningAgent(
    env=env,
    learning_rate=0.01,
    epsilon=0.1,
    wealth_discretization=100,
    action_discretization=50
)

# Train agent
agent.train(max_episodes=10000)
```

### Environment Parameters

- `initial_wealth`: Starting wealth value
- `T`: Total number of time steps
- `r`: Risk-free rate per period
- `a`: Upside return of risky asset
- `b`: Downside return of risky asset
- `p`: Probability of upside return

### Agent Parameters

- `learning_rate`: TD learning rate
- `discount_factor`: Future reward discount factor
- `epsilon`: Initial exploration rate
- `epsilon_decay`: Exploration rate decay
- `min_epsilon`: Minimum exploration rate
- `wealth_discretization`: Number of wealth state bins
- `action_discretization`: Number of action bins
- `action_multiplier_range`: Range for action scaling

## Testing

The project includes unit tests for both the environment and agent. Run tests using:

```bash
python -m unittest test_code.py
```
