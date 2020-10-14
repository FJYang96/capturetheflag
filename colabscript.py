import numpy as np
from environment.colabenv import ColabSim
from environment.colabagent import ColabCtrlAgent, ColabMPCAgent
from environment.colabutils import ColabGameSolver

# Parameters
time_limit = 100
sim = ColabSim(2,2,render=False, time_limit=time_limit,
        penalty_weight=np.array([[2,50,1.5],[2,50,1.5]]))

# Define game-theoretic solver
solver = ColabGameSolver()

# Define strat to simulate
#strats = [[0,0,1],[0,1,0]]

# Or simulate open-loop NE strategy
#ne = solver.find_nash(sim)
#strats = solver.profile_to_env_strat(ne)
#agents = [ColabCtrlAgent(strats[0], sim.time_limit),
#        ColabCtrlAgent(strats[1], sim.time_limit)]

# Or simulate MPC agents
agents = [ColabMPCAgent(solver), ColabMPCAgent(solver)]

def simulate_strat(sim, agents):
    observation, game_ended = sim.reset()
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

sim.render=True
print(simulate_strat(sim, agents))
