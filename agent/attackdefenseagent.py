import numpy as np
from .utils import softmax
from .abstractagent import TwoTierAgent

EPS = 1e-4

class AttackDefenseAgent(TwoTierAgent):
    ''' Abstract class
    This class is a specific type of two-tier agents whose meta-strategies are
    attack and defend. The specific meta-strategies are implemented, but the
    algorithm to decide on meta-strategy from observation is not implemented.
    '''
    def __init__(self,
                 num_players,
                 pull_strength=3,
                 push_strength=0.2,
                 enemy_attraction=0.8):
        super().__init__(num_players)
        self.pull_strength = pull_strength
        self.push_strength = push_strength
        self.alpha = enemy_attraction

    def get_meta_strategy(self, observation):
        raise NotImplementedError

    def attack_action(self, observation):
        ''' Returns a 2D numpy array
        Compute control for current step if one wants to attack. The algorithm is
        a potential field method
        '''
        self_pos, self_flag_pos, enemy_flag_pos, \
            teammate_pos, enemy_pos = observation
        pos_diff = self_pos - enemy_pos
        push_forces = pos_diff / (np.linalg.norm(pos_diff, axis=1)**2)[:, None]
        flag_diff = enemy_flag_pos - self_pos
        pull_force = flag_diff / np.linalg.norm(flag_diff)**2
        control = push_forces.sum(0) * self.push_strength + \
                  pull_force * self.pull_strength
        self.meta_strategy_target = -1
        return control

    def assign_opponent_to_defend(self, observation):
        ''' Returns an integer
        Assigns an opponent to defend based on a softmax of their inverse
        to self goal
        '''
        _, self_flag_pos, _, _, enemy_pos = observation
        enemy_distance_inv = 1 / \
            (np.linalg.norm(self_flag_pos - enemy_pos, axis=1) + EPS)
        enemy_weight = softmax(enemy_distance_inv, sharpening=100)
        assignment = np.random.choice(enemy_pos.shape[0], p = enemy_weight)
        return assignment

    def defend_action(self, observation):
        ''' Returns a 2D numpy array
        Computes the control action for defending.
        Side effect: assigns self.meta_strategy_target
        '''
        self_pos, self_flag_pos, _, teammate_pos, enemy_pos = observation
        assignment = self.assign_opponent_to_defend(observation)
        enemy_pull = enemy_pos[assignment] - self_pos
        # Pulled towards the goal
        goal_pull = self_flag_pos - self_pos
        # Sum the control
        control = self.alpha * enemy_pull + (1 - self.alpha) * goal_pull
        self.meta_strategy_target = assignment
        return control

    def get_action(self, observation):
        raise NotImplementedError


class DynamicADAgent(AttackDefenseAgent):
    '''
    Dynamic Attack-Defense Agents decides on their meta-strategy based on their
    observation. The observation is processed using a featurizer. The meta
    strategy is then decided by dotting a weight vector with the features and
    sampling the meta-strategy in a soft-max fashion.
    '''
    def __init__(self, 
                 num_players,
                 featurizer,
                 parameters,
                 pull_strength=2,
                 push_strength=0.15,
                 enemy_attraction=0.7):
        super().__init__(num_players, pull_strength, push_strength, 
                         enemy_attraction)
        self.parameters = parameters
        self.featurizer = featurizer

    def get_meta_strategy(self, observation):
        ''' Returns an integer
        Computes the meta strategy from observation using featurizer and weights.
        Side effect: assigns self.meta_strategy
        '''
        features = self.featurizer(observation)
        s_meta_weights = np.dot(self.parameters, features)
        s_meta_prob = softmax(s_meta_weights)
        s_meta = np.random.choice(len(s_meta_prob), p=s_meta_prob)
        self.meta_strategy = s_meta
        return s_meta

    def get_action(self, observation):
        ''' Returns a 2D numpy array
        Computes the action to take
        '''
        s_meta = self.get_meta_strategy(observation)
        action = None
        if s_meta == 0:
            action = self.attack_action(observation)
        else:
            action = self.defend_action(observation)
        return action
