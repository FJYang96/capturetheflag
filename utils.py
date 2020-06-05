import numpy as np
from agent.attackdefenseagent import DynamicADAgent
from agent.featurizer import default_feature
from environment.ctfenv import CTFSim

attack_weight = np.array([[1,1,1,1,1],[0,0,0,0,0]])
defense_weight = np.array([[0,0,0,0,0],[1,1,1,1,1]])
attack_inclined = np.array([[-0.8,4,-0.5,-0.5,-0.5],[0.5,-8,0.5,0.5,0.5]])
defense_inclined = np.array([[-2,6,-2,-2,-2],[2,-2,2,2,2]])
weights = [attack_weight, defense_weight, attack_inclined, defense_inclined]

def assignment_from_symmetric_profile(profile):
    ''' Returns a 1D numpy array
    Randomly assign a meta-strategy to each agent so that the overall counts
    is consistent with the given profile
    '''
    N = np.sum(profile)
    assn = np.zeros(N)
    inds = np.arange(N)
    for i in range(len(profile)-1):
        s_i = np.random.choice(np.arange(len(inds)), profile[i], replace=False)
        assn[inds[s_i]] = i
        inds = np.delete(inds, s_i)
    assn[inds] = len(profile) - 1
    return assn.astype(int)

def agents_from_assignment(num_players, assignment):
    ''' Returns a list of Agents
    Creates a list of agents from the assignment of meta-strategies
    '''
    agents = []
    for i in assignment:
        agents.append(DynamicADAgent(num_players, default_feature, weights[i]))
    return agents

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

def simulate_profile(sim, profile, num_players):
    ''' Returns an 1D array
    Wrapper for the simulating a profile with CTFSim. Returns the payoffs of a
    profile
    '''
    S = int(len(profile) / 2)
    red_agents = agents_from_profile(profile[:S])
    blue_agents = agents_from_profile(profile[S:])
    agents = red_agents + blue_agents

    observation, game_ended, _, _ = sim.reset()
    N = np.sum(num_players)
    while not game_ended:
        controls = np.zeros((N, 2))
        for i, a in enumerate(agents):
            controls[i] = a.get_action(observation[i])
        observation, game_ended, blue_win, red_win = sim.step(controls)
    payoffs = np.zeros(profile.shape)
    if red_win:
        payoffs[:S] = 1
        payoffs[S:] = -1
    elif blue_win:
        payoffs[:S] = -1
        payoffs[S:] = 1
    return payoffs

def count_effective_simulation_steps(sim, profile, num_players):
    S = int(len(profile) / 2)
    red_agents = agents_from_profile(profile[:S])
    blue_agents = agents_from_profile(profile[S:])
    agents = red_agents + blue_agents
    observation, game_ended, _, _ = sim.reset()
    count = 0
    N = np.sum(num_players)
    while not game_ended:
        count += 1
        controls = np.zeros((N, 2))
        for i, a in enumerate(agents):
            controls[i] = a.get_action(observation[i])
        observation, game_ended, blue_win, red_win = sim.step(controls)
    return count

def estimate_payoff(sim, profile, num_players, n=10):
    ''' Returns an 1D array
    Wrapper for simulating a profile multiple times and averaging payoff
    '''
    avg_pay = np.zeros(profile.shape)
    for i in range(n):
        avg_pay = avg_pay + simulate_profile(sim, profile, num_players)
    avg_pay = avg_pay / n
    return avg_pay

def simulate_episode(sim, profile, num_players):
    ''' Returns a ctflogger
    Wrapper for the simulating a profile with CTFSim. Returns the payoffs of a
    profile
    '''
    S = int(len(profile) / 2)
    red_agents = agents_from_profile(profile[:S])
    blue_agents = agents_from_profile(profile[S:])
    agents = red_agents + blue_agents

    observation, game_ended, _, _ = sim.reset()
    N = np.sum(num_players)
    while not game_ended:
        controls = np.zeros((N, 2))
        for i, a in enumerate(agents):
            controls[i] = a.get_action(observation[i])
        observation, game_ended, blue_win, red_win = sim.step(controls)
    return sim.get_logger()
