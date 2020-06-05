import numpy as np
from .ctfvisualizer import CTFRenderer
from .io import ctflogger

class CTFSim:
    def __init__(self,
                 num_players, 
                 max_vels = None,
                 initial_pos=None,
                 initial_vels=None,
                 time_limit = 150,
                 noise_var = 0.25,
                 render=False):
        '''
        Initialize the simulator; place everyone in position etc.
        ----------------------
        Params:
            N - (2-tuple) number of players on each team
            max_vels - (1D array) max velocity of each robot
            initial_position - (2D array) initial position for each robot
            initial_velocity - (2D array) initial velocity for each robot
            render - (bool) whether to render the animation for this simulation
        '''
        # Field configuration
        self.num_players = num_players
        self.N = np.sum(num_players)
        self.height_margin = 10      # starting height of the field
        self.length_margin = 50      # starting length of the field
        self.height_expansion = 4    # height gain of the field per additional agent
        self.length_expansion = 1    # length gain of the field per additional agent
        self.flag_margin_dist = 5    # distance from flag to left/right sides
        self.h = self.height_margin + self.height_expansion * self.N / 2
        self.l = self.length_margin + self.length_expansion * self.N / 2
        self.red_flag_pos = np.array([self.flag_margin_dist, self.h/2])
        self.blue_flag_pos = np.array([self.l - self.flag_margin_dist, self.h/2])
        self.flag_positions = (self.red_flag_pos, self.blue_flag_pos)
        self.radius = 0.5

        # Set the initial position and velocity of the agents
        self.agent_lr_margin = 10    # distance from agent to left/right sides
        self.agent_tb_margin = 7     # distance from agent to top/bottom sides

        if initial_pos is None:
            red_distance = (self.h - self.agent_tb_margin) / num_players[0]
            blue_distance = (self.h - self.agent_tb_margin) / num_players[1]
            red_positions = np.array(
                [(self.agent_lr_margin, self.agent_tb_margin + \
                  i * red_distance) for i in range(self.num_players[0])])
            blue_positions = np.array(
                [(self.l-self.agent_lr_margin, self.agent_tb_margin \
                  + i * blue_distance) for i in range(self.num_players[1])])
            self.initial_pos = np.vstack([red_positions, blue_positions])
        else:
            self.initial_pos = initial_pos

        if initial_vels is None:
            self.initial_vels = np.zeros((self.N, 2))
        else:
            self.initial_vels = initial_vels

        # Set the maximum velocity of the agents
        if max_vels is None:
            self.max_vels = 0.5
        else:
            self.max_vels = max_vels

        # Set simulation step limit
        self.time_limit = time_limit

        # Set control noise
        self.noise_var = noise_var

        # Set whether to render the simulation or not
        self.render = render
        if self.render:
            self.renderer = CTFRenderer(num_players, self.l, self.h)

        # initialize a logger
        self.logger = ctflogger(num_players, self.l, self.h, self.flag_positions)

    def reset(self):
        ''' Returns a tuple
        Resets the environment to its initial condition
        '''
        self.agent_positions = self.initial_pos.copy()
        self.agent_velocity = self.initial_vels.copy()
        self.step_count = 0
        self.logger.clear_history()
        if self.render:
            self.renderer.render_step(self.agent_positions, 
                                      (self.red_flag_pos, self.blue_flag_pos))

        return (self.pack_observation(), False, False, False)


    def apply_control(self, controls):
        ''' Returns a 2D numpy array
        Computes the position of agents given the current control
        Returns the projected position of the agents (ignoring collisions)
        '''
        self.agent_velocity = self.agent_velocity + controls
        self.agent_velocity = \
            self.agent_velocity.clip(-self.max_vels, self.max_vels)
        positions = self.agent_positions + self.agent_velocity
        return positions

    def pack_observation(self):
        ''' Returns a list of tuples
        Pack the observations for all agents
        Return:
            observation - an array where each element is the observation of
            the corresponding agent
        Note:
            for each agent, the observation is a 5-tuple of
            (self_pos, self_flag_pos, enemy_flag_pos, teammate_pos, enemy_pos)
        '''
        # TODO: vectorize the code
        observation = []
        red_positions = self.agent_positions[:self.num_players[0]]
        blue_positions = self.agent_positions[self.num_players[0]:]
        # Append red agents' observations
        for i in range(self.num_players[0]):
            observation.append(
                (self.agent_positions[i], self.red_flag_pos, self.blue_flag_pos,
                 red_positions, blue_positions)
            )
        # Append blue agents' observations
        for i in range(self.num_players[0], self.N):
            observation.append(
                (self.agent_positions[i], self.blue_flag_pos, self.red_flag_pos,
                 blue_positions, red_positions)
            )
        return observation


    def step(self, controls):
        ''' Returns a tuple
        One discrete step of the simulation; moves the robot according to their
        control
        -----------------------
        Params:
        -controls:  an array of shape (num_robot * 2, 2). This is maintained
                    under the convention that red control is followed by
                    blue control

        '''
        # Corrupt the controls with a small amount of noise
        noise = np.random.normal(0, self.noise_var, size=(self.N, 2))
        controls = controls + noise
        # Clip positions to be inside the walls
        for i in range(int(self.N/2)):
            pos = self.apply_control(controls)
            pos[:, 0] = pos[:, 0].clip(0, self.l)
            pos[:, 1] = pos[:, 1].clip(0, self.h)

        # Compute the distance between each pair of agents
        dists_table = np.linalg.norm(pos[:,None,:] - pos, axis=2)

        # Check for collision and apply game dynamics
        # TODO: vectorize the code here
        for i in range(self.num_players[0]):
            for j in range(self.num_players[0], self.N):
                if dists_table[i,j] < self.radius * 2:
                    #print('collision!')
                    if self.agent_positions[i,0] < self.l / 2:
                        # Collision is in the red half of the court
                        pos[j] = \
                            (self.l - self.agent_lr_margin, self.h / 2)
                        self.agent_velocity[i] = \
                            (self.agent_velocity[i] + self.agent_velocity[j]) / 2
                        self.agent_velocity[j] = (0, 0)
                    else:
                        # Collision in blue half of the court
                        pos[i] = \
                            (self.agent_lr_margin, self.h / 2)
                        self.agent_velocity[j] = \
                            (self.agent_velocity[i] + self.agent_velocity[j]) / 2
                        self.agent_velocity[i] = (0, 0)

        self.agent_positions = pos
        observation = self.pack_observation()                     

        # Render if asked
        if self.render:
            self.renderer.render_step(self.agent_positions, 
                                      (self.red_flag_pos, self.blue_flag_pos))

        # Check for whether the game has ended
        dist_to_red_flag = np.linalg.norm(
            self.agent_positions[self.num_players[0]:, :] - np.array(self.red_flag_pos),
            axis=1)
        red_flag_captured = np.any(dist_to_red_flag < self.radius)

        dist_to_blue_flag = np.linalg.norm(
            self.agent_positions[:self.num_players[0], :] - np.array(self.blue_flag_pos),
            axis=1)
        blue_flag_captured = np.any(dist_to_blue_flag < self.radius)

        self.step_count += 1
        game_ended = red_flag_captured or blue_flag_captured or \
            self.step_count >= self.time_limit

        self.logger.log_state(self.agent_positions)

        return observation, game_ended, \
            red_flag_captured, blue_flag_captured

    def get_logger(self):
        return self.logger
