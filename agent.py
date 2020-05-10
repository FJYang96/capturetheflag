import numpy as np

class Agent:
    def __init__(self):
        raise NotImplementedError
    def get_action(self, self_pos, self_flag_pos, enemy_flag_pos, 
                   teammate_pos, enemy_pos):
        raise NotImplementedError

class PotentialAgent(Agent):
    def __init__(self, pull_strength=4, push_strength=0.2):
        self.pull_strength = pull_strength
        self.push_strength = push_strength

    def get_action(self, self_pos, self_flag_pos, enemy_flag_pos, 
                   teammate_pos, enemy_pos):
        pos_diff = self_pos - enemy_pos
        push_forces = pos_diff / (np.linalg.norm(pos_diff, axis=1)**2)[:, None]
        flag_diff = enemy_flag_pos - self_pos
        pull_force = flag_diff / np.linalg.norm(flag_diff)**2
        control = push_forces.sum(0) * self.push_strength + \
                  pull_force * self.pull_strength
        return control

class CollisionAgent(Agent):
    def __init__(self, assignment, enemy_attraction=0.9):
        # Assignment specifies the index of the enemy agent to collide with
        self.assignment = assignment
        self.alpha = enemy_attraction

    def get_action(self, self_pos, self_flag_pos, enemy_flag_pos, 
                   teammate_pos, enemy_pos):
        enemy_pull = enemy_pos[self.assignment] - self_pos
        goal_pull = self_flag_pos - self_pos
        control = self.alpha * enemy_pull + (1 - self.alpha) * goal_pull
        return control
