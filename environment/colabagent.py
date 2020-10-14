import numpy as np
from .colabenv import ColabSim

EPS = 1e-8

class ColabCtrlAgent:
    def __init__(self, strategy, time_limit=100):
        self.time_limit = time_limit
        self.strategy = strategy
        self.t = 0
        ############################ Strategy parameters ##############################
        self.leader_weights = np.array((5, 1)) #(goal_pull, ad_push)
        self.follower_weights = np.array((0.33, 2)) #(leader_action_multiplier, ad_push)
        self.attention_radius = 1.5
        ###############################################################################

    def goal_control(self, x_self, x_teammate, x_goal, x_teammate_goal, x_adv, sim_param):
        goal_diff = (x_goal + x_teammate_goal - x_self - x_teammate) / 2
        return goal_diff / (np.linalg.norm(goal_diff) + EPS)

    def self_goal_control(self, x_self, x_teammate, x_goal, x_teammate_goal, x_adv,
            sim_param):
        goal_diff = (x_goal - x_self)
        return goal_diff / (np.linalg.norm(goal_diff) + EPS)

    def angle_control(self, x_self, x_teammate, x_goal, x_teammate_goal, x_adv,
            sim_param):
        team_diff = x_teammate - x_self
        theta = np.arctan(team_diff[1]/team_diff[0])
        dtheta = min(0-theta, 2*np.pi-theta, key=abs)
        angle_ctrl = np.array([team_diff[1], -team_diff[0]]) * dtheta
        return angle_ctrl / (np.linalg.norm(angle_ctrl) + EPS)

    def adv_control(self, x_self, x_teammate, x_goal, x_teammate_goal, x_adv, sim_param):
        adv_diff = x_self - x_adv
        adv_diff = adv_diff / (np.linalg.norm(adv_diff, axis=1)[:, None] \
                - self.attention_radius + EPS) ** 2
        adv_diff = adv_diff.sum(0)
        return adv_diff

    def control_components(self, observation):
        goal_ctrl = self.goal_control(*observation)
        angle_ctrl = self.angle_control(*observation)
        adv_ctrl = self.adv_control(*observation)
        return np.array([goal_ctrl, angle_ctrl, adv_ctrl])

    def goal_dist(self, x_ego, x_goal):
        distance = np.linalg.norm(x_ego - x_goal)
        return distance

    def speak(self, observation, teammate_action):
        x_self, x_teammate, x_goal, x_teammate_goal, x_adv, _ = observation
        goal_direction = self.self_goal_control(*observation)
        adv_direction = self.adv_control(*observation)
        leading_ctrl = (self.leader_weights[0])*goal_direction + \
                (self.leader_weights[1])*adv_direction
        dist_to_goal = self.goal_dist(x_self, x_goal)
        # Also engage in orientation ctrl after reaching the destination
        angle_weight = min(1, 1/(dist_to_goal + EPS))
        control = (1-angle_weight) * leading_ctrl + \
                angle_weight * self.angle_control(*observation)
        return control

    def listen(self, observation, teammate_action):
        x_self, x_teammate, x_goal, x_teammate_goal, x_adv, _ = observation
        angle_ctrl = self.angle_control(*observation)
        adv_ctrl = self.adv_control(*observation)
        following_ctrl = self.follower_weights[0] * teammate_action +\
                self.follower_weights[1] * adv_ctrl
        # Adjust for angle more aggressively when teammate close to finish
        teammate_dist_to_goal = self.goal_dist(x_teammate, x_teammate_goal)
        angle_weight = min(1, 1/(teammate_dist_to_goal + EPS))
        control = (1-angle_weight) * following_ctrl +\
                angle_weight * angle_ctrl
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

class ColabMPCAgent(ColabCtrlAgent):
    def __init__(self, solver, time_limit=100, step_horizon=5):
        super().__init__(None, time_limit)
        self.solver = solver
        self.step_horizon = step_horizon
        self.i = -1
        self.current_strategy = None

    def generate_game(self, observation):
        pos, x, dx, penalty_weight, self.i = observation[-1]
        sim = ColabSim(2,2,penalty_weight=penalty_weight, initial_pos=pos,
                initial_dx=dx, time_limit=self.step_horizon*3)
        return sim

    def MPC(self, observation):
        sim = self.generate_game(observation)
        ne = self.solver.find_nash(sim)
        self.current_strategy = self.solver.profile_to_env_strat(ne)[self.i]
        return self.current_strategy

    def control(self, observation, teammate_action):
        controls = None
        if self.t % self.step_horizon == 0:
            if self.t + self.step_horizon * 3 < self.time_limit:
                self.MPC(observation)
            else:
                self.current_strategy.pop(0)

        controls = self.control_based_on_strategy(observation, teammate_action,
                 self.current_strategy[0])
        return controls
