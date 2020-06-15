import numpy as np
from agent.attackdefenseagent import DynamicADAgent
from agent.featurizer import default_feature
from environment.ctfenv import CTFSim
import environment.wrapper as wrapper
import utils

np.set_printoptions(precision=2, suppress=True)
RES_DIR = './results/'

# Experiment name
NAME = '4v4'

# Number of players different
num_players = [4,2]
N = np.sum(num_players)

# Asymmetric starting condition
sim = wrapper.asymmetric_sim(num_players, max_vels=[0.75,0.75], render=False)
# Symmetric starting condition
#sim = CTFSim(num_players, noise_var=0, render=False)

#utils.simulate_profile(sim, np.array([2,0,2,1]),num_players)
log = utils.simulate_episode(sim, np.array([4,0,0,2]), num_players)

history_dir = RES_DIR + NAME + '.txt'
gif_dir = RES_DIR + NAME + '.mp4'
log.dump_to_file(history_dir)
log.mp4_from_file(gif_dir)
log.render_from_file(history_dir)

'''
observation, game_ended, _, _ = sim.reset()
N = np.sum(num_players)
while not game_ended:
    controls = np.zeros((N, 2))
    for i, a in enumerate(agents):
        controls[i] = a.get_action(observation[i])
    observation, game_ended, blue_win, red_win = sim.step(controls)
    print('-'*20)
'''
