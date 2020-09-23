import numpy as np
EPS = 1e-8

class ColabAgent:
    def __init__(self, strategy, time_limit=150):
        self.time_limit = time_limit
        self.strategy = strategy
        self.weights = (5, 1)
        self.t = 0
        self.teammate_prev_pos = np.array([20,20])

    def speak(self, observation, teammate_action):
        x_self, x_teammate, x_goal, x_teammate_goal, x_adv = observation
        goal_diff = x_goal - x_self
        goal_ctrl = goal_diff / (np.linalg.norm(goal_diff) + EPS) ** 2
        adv_diff = x_self - x_adv
        adv_ctrl = adv_diff / (np.linalg.norm(adv_diff) + EPS)**2
        control = self.weights[0] * goal_ctrl + self.weights[1] * adv_ctrl
        return control

    def listen(self, observation, teammate_action):
        x_self, x_teammate, x_goal, x_teammate_goal, x_adv = observation
        goal_diff = x_goal - x_self
        goal_ctrl = goal_diff / (np.linalg.norm(goal_diff) + EPS) ** 2
        # Estimate the location of unseen adversary
        teammate_diff = x_teammate_goal - x_teammate
        teammate_pull = teammate_diff / (np.linalg.norm(teammate_diff) + EPS) ** 2
        g = (teammate_action - self.weights[0] * teammate_pull) / self.weights[1]
        x_other_adv = x_teammate - g / np.linalg.norm(g) ** 2
        # Use speak() to compute control without the knowledge of the unseen adversary
        partial_ctrl = self.speak((x_self, x_teammate, x_goal, None, x_adv), None)
        other_diff = x_self - x_other_adv
        other_ctrl = other_diff / (np.linalg.norm(other_diff) + EPS)**2
        control = partial_ctrl + self.weights[1] * other_ctrl
        return control

    def control_based_on_strategy(self, observation, teammate_action, strategy):
        if strategy == 0:
            return self.speak(observation, teammate_action)
        else:
            return self.listen(observation, teammate_action)

    def control(self, observation, teammate_action):
        self.t += 1
        if self.t <= self.time_limit / 3:
            return self.control_based_on_strategy(observation, teammate_action,
                    self.strategy[0])
        elif self.t <= 2 * self.time_limit / 3:
            return self.control_based_on_strategy(observation, teammate_action,
                    self.strategy[1])
        else:
            return self.control_based_on_strategy(observation, teammate_action,
                    self.strategy[2])
