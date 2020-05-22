import numpy as np
import time

from agent.attackdefenseagent import DynamicADAgent
from agent.featurizer import default_feature
from environment.ctfenv import CTFSim
import utils

from gameanalysis import rsgame, paygame, nash

#---------------------------Simulation script---------------------
class CTFEqFinder:
    def __init__(self, num_players, num_meta_strategies, symm=False):
        self.num_players = num_players
        self.ms = num_meta_strategies
        self.sim = CTFSim(num_players, render=False)

    def find_eq(self):
        players = [self.num_players, self.num_players]
        strats = [self.ms] * len(players)
        eg = rsgame.empty(players, strats)
        profs = eg.all_profiles()
        pays = []
        for p in profs:
            pays.append(utils.estimate_payoff(self.sim, p))
        pays = np.array(pays)
        pays[profs==0] = 0
        pg = paygame.game(players, strats, profs, pays)
        # Compute the Nash for a couple of times
        nashes = []
        for _ in range(5):
            n = nash.replicator_dynamics(pg, pg.random_mixture())
            nashes.append(n)
        return nashes
