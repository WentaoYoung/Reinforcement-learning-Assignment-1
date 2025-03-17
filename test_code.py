import unittest
import numpy as np
from code import AssetAllocationEnvironment, QLearningAgent

class TestAssetAllocationEnvironment(unittest.TestCase):
    def setUp(self):
        self.env = AssetAllocationEnvironment(
            initial_wealth=1.0,
            T=10,
            r=0.03,
            a=0.15,
            b=-0.06,
            p=0.6
        )

    def test_initialization(self):
        """Test environment initialization parameters"""
        self.assertEqual(self.env.initial_wealth, 1.0)
        self.assertEqual(self.env.T, 10)
        self.assertEqual(self.env.r, 0.03)
        self.assertEqual(self.env.a, 0.15)
        self.assertEqual(self.env.b, -0.06)
        self.assertEqual(self.env.p, 0.6)
        self.assertEqual(self.env.wealth, 1.0)
        self.assertEqual(self.env.time_step, 0)

    def test_invalid_parameters(self):
        """Test invalid parameter initialization"""
        with self.assertRaises(ValueError):
            AssetAllocationEnvironment(r=0.2, a=0.1, b=0.05)  # Condition a > r > b not satisfied

    def test_reset(self):
        """Test environment reset functionality"""
        self.env.wealth = 2.0
        self.env.time_step = 5
        state = self.env.reset()
        self.assertEqual(self.env.wealth, 1.0)
        self.assertEqual(self.env.time_step, 0)
        self.assertEqual(state, (0, 1.0))

    def test_step(self):
        """Test state transition and reward calculation"""
        np.random.seed(42)  # Fix random seed to ensure reproducible test results
        
        # Test risk-free asset investment
        state, reward, done = self.env.step(0.0)
        expected_wealth = 1.0 * (1 + 0.03)
        self.assertAlmostEqual(state[1], expected_wealth)
        self.assertEqual(state[0], 1)
        self.assertFalse(done)
        
        # Reset environment and test risky asset investment
        self.env.reset()
        state, reward, done = self.env.step(1.0)
        self.assertEqual(state[0], 1)
        self.assertFalse(done)

    def test_terminal_state(self):
        """Test terminal state and final reward"""
        self.env.time_step = self.env.T - 1
        state, reward, done = self.env.step(0.0)
        self.assertTrue(done)
        self.assertNotEqual(reward, 0)  # Terminal state should have non-zero reward

class TestQLearningAgent(unittest.TestCase):
    def setUp(self):
        self.env = AssetAllocationEnvironment(
            initial_wealth=1.0,
            T=10,
            r=0.03,
            a=0.15,
            b=-0.06,
            p=0.6
        )
        self.agent = QLearningAgent(
            env=self.env,
            action_multiplier_range=(-2.0, 3.0),
            wealth_discretization=100,
            action_discretization=50,
            learning_rate=0.01,
            epsilon_decay=0.995
        )

    def test_initialization(self):
        """Test agent initialization parameters"""
        self.assertEqual(self.agent.wealth_discretization, 100)
        self.assertEqual(self.agent.action_discretization, 50)
        self.assertEqual(self.agent.learning_rate, 0.01)
        self.assertEqual(self.agent.epsilon_decay, 0.995)
        self.assertEqual(self.agent.action_multiplier_range, (-2.0, 3.0))

    def test_discretize_wealth(self):
        """Test wealth discretization"""
        # Test boundary cases
        self.assertEqual(self.agent.discretize_wealth(0), 0)
        self.assertEqual(self.agent.discretize_wealth(self.agent.max_wealth), 99)
        
        # Test middle values
        mid_wealth = self.agent.max_wealth / 2
        self.assertGreater(self.agent.discretize_wealth(mid_wealth), 0)
        self.assertLess(self.agent.discretize_wealth(mid_wealth), 99)

    def test_discretize_action(self):
        """Test action discretization"""
        # Test zero wealth case
        self.assertEqual(self.agent.discretize_action(0, 0), 0)
        
        # Test normal case
        wealth = 1.0
        action = 2.0  # 200% investment
        action_idx = self.agent.discretize_action(action, wealth)
        self.assertGreaterEqual(action_idx, 0)
        self.assertLess(action_idx, self.agent.action_discretization)

    def test_action_from_index(self):
        """Test conversion from index back to actual action value"""
        wealth = 1.0
        for idx in range(self.agent.action_discretization):
            action = self.agent.action_from_index(idx, wealth)
            self.assertGreaterEqual(action, wealth * self.agent.action_multiplier_range[0])
            self.assertLessEqual(action, wealth * self.agent.action_multiplier_range[1])

    def test_get_action(self):
        """Test action selection strategy"""
        state = (0, 1.0)
        # Test exploration
        self.agent.epsilon = 1.0
        action1 = self.agent.get_action(state)
        action2 = self.agent.get_action(state)
        self.assertNotEqual(action1, action2)  # Should choose random actions in exploration mode
        
        # Test exploitation
        self.agent.epsilon = 0.0
        action1 = self.agent.get_action(state)
        action2 = self.agent.get_action(state)
        self.assertEqual(action1, action2)  # Should choose same action in exploitation mode

    def test_short_training(self):
        """Test short training process"""
        self.agent.train(max_episodes=10, conv_threshold=1e-5, conv_window=1)
        self.assertGreater(len(self.agent.delta_history), 0)
        self.assertGreater(len(self.agent.analytical_mse), 0)

if __name__ == '__main__':
    unittest.main()