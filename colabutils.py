import numpy as np
from environment.colabenv import ColabSim
from environment.colabagent import ColabAgent

S = 8

def profile_to_env_strat(profile):
    # Convert EGTA (single-player) profile to strategy in env
    num1 = np.where(profile[:S]==1)[0][0]
    num2 = np.where(profile[S:]==1)[0][0]
    return (number_to_strat(num1), number_to_strat(num2))

def number_to_strat(num):
    strat = []
    for i in range(3):
        strat.append(num % 2)
        num = num // 2
    return strat

def agents_from_profile(profile, symm=True):
    ''' Returns a list of Agents
    '''
    num_players = np.sum(profile)
    if symm:
        assn = assignment_from_symmetric_profile(profile)
        return agents_from_assignment(num_players, assn)
    else:
        # TODO: implement asymmetric profile; remove what is written here
        assn = assignment_from_symmetric_profile(profile)
        return agents_from_assignment(assn)

def estimate_payoff(profile, n=10):
    ''' Returns an 1D array
    Wrapper for simulating a profile multiple times and averaging payoff
    '''
    avg_pay = np.zeros(profile.shape)
    for i in range(n):
        avg_pay = avg_pay + simulate_profile(profile)
    avg_pay = avg_pay / n
    return avg_pay

def simulate_profile(profile, render=False):
    ''' Returns a ctflogger
    Wrapper for the simulating a profile with CTFSim. Returns the payoffs of a
    profile
    '''
    # Write script
    strats = profile_to_env_strat(profile)
    return profile * simulate_strat(strats, render=render)

def simulate_strat(strats, render=False):
    sim = ColabSim(render=render)
    observation, game_ended = sim.reset()
    agents = [ColabAgent(strats[0]), ColabAgent(strats[1])]
    # Simulate episode and accumulate penalty
    total_penalty = 0
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

#c = simulate_strat(([1,0,0],[0,1,0]), render=True)
#print(c)
