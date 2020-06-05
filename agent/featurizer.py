import numpy as np

def unpack_observation(observation):
    ''' Returns a list
    General interface for unpacking observation into items.
    '''
    self_pos, self_flag_pos, enemy_flag_pos, \
        teammate_pos, enemy_pos = observation
    return self_pos, self_flag_pos, enemy_flag_pos, teammate_pos, enemy_pos

def court_length(self_flag_pos, enemy_flag_pos):
    ''' Returns a float
    Computes the length of the court
    '''
    flag_dist = enemy_flag_pos[0] - self_flag_pos[0]
    return self_pos[0] * 2 + flag_dist

def dist_to_enemy_flag(self_pos, enemy_flag_pos):
    ''' Returns a float
    Computes the distance to enemy flag
    '''
    return np.linalg.norm(enemy_flag_pos - self_pos)

def dist_to_self_flag(self_pos, self_flag_pos):
    ''' Returns a float
    Computes the distance to one's own flag
    '''
    return np.linalg.norm(self_flag_pos - self_pos)

def num_teammate_offense(self_pos, teammate_pos, enemy_flag_pos):
    ''' Returns an integer
    Computes the number of teammates who are closer to the enemy_flag
    '''
    teammate_dist_to_flag = np.linalg.norm(enemy_flag_pos - teammate_pos, axis=1)
    if_teammate_closer = \
        teammate_dist_to_flag < dist_to_enemy_flag(self_pos, enemy_flag_pos)
    return np.sum(if_teammate_closer)

def num_opponent_offense(self_flag_pos, enemy_pos, enemy_flag_pos):
    ''' Returns an integer
    Counts the number of enemy agents in one's own half of the court
    '''
    enemy_dist_to_self_flag = np.linalg.norm(self_flag_pos - enemy_pos, axis=1)
    enemy_dist_to_enemy_flag = np.linalg.norm(enemy_flag_pos - enemy_pos, axis=1)
    return np.sum(enemy_dist_to_self_flag - enemy_dist_to_enemy_flag < 0)

def closest_enemy_to_self_flag(self_flag_pos, enemy_pos):
    ''' Returns a float
    Returns the distance between self flag and the enemy closest to self flag
    '''
    enemy_dist_to_self_flag = np.linalg.norm(self_flag_pos - enemy_pos, axis=1)
    return np.min(enemy_dist_to_self_flag)

def default_feature(observation):
    ''' Returns an 1D numpy array
    Summarize a feature vector from state observations
    '''
    self_pos, self_flag_pos, enemy_flag_pos, teammate_pos, enemy_pos = \
        unpack_observation(observation)
    features = [
        dist_to_enemy_flag(self_pos, enemy_flag_pos),
        dist_to_self_flag(self_pos, self_flag_pos),
        num_teammate_offense(self_pos, teammate_pos, enemy_flag_pos),
        num_opponent_offense(self_flag_pos, enemy_pos, enemy_flag_pos),
        closest_enemy_to_self_flag(self_flag_pos, enemy_pos)
    ]
    return np.array(features)
