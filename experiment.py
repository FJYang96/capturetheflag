from CTFEnv import CTFSim
from agent import PotentialAgent, CollisionAgent
import numpy as np

#---------------------------Simulation script---------------------
num_players = 3
sim = CTFSim(num_players, render=True)
observation, game_ended, _, _ = sim.reset()

#blue_assignment = np.random.choice(num_players, size=num_players)
agents = [CollisionAgent(np.random.randint(num_players)), 
          CollisionAgent(1),
          PotentialAgent(),
          PotentialAgent(),
          PotentialAgent(),
          CollisionAgent(np.random.randint(num_players))]

while not game_ended:
    controls = np.zeros((num_players * 2, 2))
    for i, a in enumerate(agents):
        controls[i] = a.get_action(*observation[i])
    # Corrupt the control with tiny bits of noise
    controls = controls + (np.random.random((num_players * 2, 2)) - 0.5) / 8
    observation, game_ended, _, _ = sim.step(controls)

