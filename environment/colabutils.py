import numpy as np
from .colabenv import ColabSim
from .colabagent import ColabCtrlAgent
from gameanalysis import rsgame, paygame, nash, learning

class ColabGameSolver:
    def __init__(self, num_players=2, num_strat=8):
        self.N = 2
        self.S = 8

    def profile_to_env_strat(self, profile):
        # Convert EGTA (single-player) profile to strategy in env
        num1 = np.argmax(profile[:self.S])
        num2 = np.argmax(profile[self.S:])
        return (self.number_to_strat(num1), self.number_to_strat(num2))

    def number_to_strat(self, num):
        strat = []
        for i in range(3):
            strat.append(num % 2)
            num = num // 2
        return strat

    def estimate_payoff(self, sim, profile, n=10):
        ''' Returns an 1D array
        Wrapper for simulating a profile multiple times and averaging payoff
        '''
        avg_pay = np.zeros(profile.shape)
        for i in range(n):
            avg_pay = avg_pay + self.simulate_profile(sim, profile)
        avg_pay = avg_pay / n
        return avg_pay

    def simulate_profile(self, sim, profile):
        ''' Returns a ctflogger
        Wrapper for the simulating a profile with CTFSim. Returns the payoffs of a
        profile
        '''
        # Write script
        strats = self.profile_to_env_strat(profile)
        s = int(len(profile) / 2)
        payoffs = self.simulate_strat(sim, strats)
        return profile * np.repeat(payoffs, s)

    def simulate_strat(self, sim, strats):
        observation, game_ended = sim.reset()
        agents = [ColabCtrlAgent(strats[0], sim.time_limit),
                ColabCtrlAgent(strats[1], sim.time_limit)]
        # Simulate episode and accumulate penalty
        total_penalty = np.zeros(2)
        last_control = np.array([[0.071, 0.08],[0.071, 0.08]])
        while not game_ended:
            control = []
            for i in range(2):
                control.append( agents[i].control(observation[i], last_control[1-i]) )
            control = np.array(control)
            last_control = control
            observation, penalty, game_ended = sim.step(control)
            total_penalty += penalty
        return total_penalty

    def find_nash(self, sim):
        players = [self.N/2,self.N/2]
        strategies = [self.S, self.S]
        g = rsgame.empty(players, strategies)
        profiles = g.all_profiles()
        payoffs = np.zeros(profiles.shape)
        for i,p in enumerate(profiles):
           payoffs[i] = ( -self.estimate_payoff(sim, p, n=1) )
        payoffs = np.array(payoffs)
        pg = paygame.game(players, strategies, profiles, payoffs)
        ne = nash.fictitious_play(pg, prof=pg.random_mixture())
        return ne
