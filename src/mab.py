import numpy as np

class MultiArmedBandit:
    def __init__(self, n_arms, probabilities):
        self.n_arms = n_arms
        self.probabilities = probabilities
        self.reset()

    def reset(self):
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)
        self.total_reward = 0
        self.total_rounds = 0

    def select_arm(self, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(self.n_arms)
        return np.argmax(self.values)

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        self.values[arm] = ((n - 1) / n) * value + (1 / n) * reward
        self.total_reward += reward
        self.total_rounds += 1

    def get_reward(self, arm):
        return np.random.random() < self.probabilities[arm]

    def run_simulation(self, n_rounds, epsilon=0.1):
        self.reset()
        history = []
        cumulative_rewards = []

        for _ in range(n_rounds):
            chosen_arm = self.select_arm(epsilon)
            reward = self.get_reward(chosen_arm)
            self.update(chosen_arm, reward)
            history.append((chosen_arm, reward))
            cumulative_rewards.append(self.total_reward / self.total_rounds)

        return {
            'history': history,
            'arm_counts': self.counts,
            'arm_values': self.values,
            'best_arm': np.argmax(self.values),
            'mean_reward': self.total_reward / n_rounds,
            'total_clicks': int(self.total_reward),
            'cumulative_rewards': cumulative_rewards
        }