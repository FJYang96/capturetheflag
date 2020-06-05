import numpy as np
from .ctfenv import CTFSim

def gen_velocities(num_players, max_vels):
    ''' Returns a 2D array
    Returns a max velocity array that can be passed into the constructor
    for CTF environment
    '''
    vel = [max_vels[0]] * num_players[0] + [max_vels[1]] * num_players[1] 
    return np.array([vel, vel]).T

def default_asymmetric_positions(num_players):
    ''' Returns a 2D array
    Returns a default asymmetric initial position array for all agents
    '''
    N = np.sum(num_players)
    h = 10 + 4 * N / 2
    l = 50 + 1 * N / 2
    lr_margin = 10
    tb_margin = 7
    red_initial_pos = np.array([(lr_margin*2, tb_margin + i*4) \
                                 for i in range(num_players[0])])
    blue_initial_pos = np.array([(l-lr_margin/2, h - tb_margin - i*4) \
                                 for i in range(num_players[1])])
    initial_pos = np.vstack([red_initial_pos, blue_initial_pos])
    return initial_pos

def asymmetric_sim(num_players,
                   max_vels=[1,0.5],
                   noise_var=0,
                   initial_pos=None,
                   render=False):
    if initial_pos is None:
        initial_pos = default_asymmetric_positions(num_players)
    sim = CTFSim(num_players=num_players,
                 max_vels=gen_velocities(num_players, max_vels),
                 noise_var=noise_var,
                 initial_pos = initial_pos,
                 render=render)
    return sim
