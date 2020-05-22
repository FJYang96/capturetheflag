import numpy as np
from .ctfvisualizer import CTFRenderer

class CTFSim:
    def __init__(self, N, render=False):
        '''
        Initialize the simulator; place everyone in position etc.
        ----------------------
        Params:
            N - number of players on each team
        '''
        # Field configuration
        self.height_margin = 10      # starting height of the field
        self.length_margin = 50      # starting length of the field
        self.height_expansion = 4    # height gain of the field per additional agent
        self.length_expansion = 1    # length gain of the field per additional agent
        self.flag_margin_dist = 5    # distance from flag to left/right sides
        self.agent_lr_margin = 10    # distance from agent to left/right sides
        self.agent_tb_margin = 7     # distance from agent to top/bottom sides
        self.agent_distance = 4      # distance between agents
        self.max_vel = 0.5        # max velocity of the robots
        self.radius = 0.5
        self.render = render
        self.time_limit = 125

        self.N = N
        self.h = self.height_margin + self.height_expansion * N
        self.l = self.length_margin + self.length_expansion * N
        self.red_flag_pos = np.array([self.flag_margin_dist, self.h/2])
        self.blue_flag_pos = np.array([self.l - self.flag_margin_dist, self.h/2])

        if render:
            self.renderer = CTFRenderer(N, self.l, self.h)

    def reset(self):
        ''' Returns a tuple
        Resets the environment to its initial condition
        '''
        # Reset the position and velocity of agents
        red_positions = np.array(
            [(self.agent_lr_margin, self.agent_tb_margin + \
              i * self.agent_distance) for i in range(self.N)])
        blue_positions = np.array(
            [(self.l-self.agent_lr_margin, self.agent_tb_margin \
              + i * self.agent_distance) for i in range(self.N)])
        self.agent_positions = np.vstack([red_positions, blue_positions])
        # Initialize agent velocities to be zero
        self.agent_velocity = np.zeros((self.N*2, 2))
        self.step_count = 0
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
            self.agent_velocity.clip(-self.max_vel, self.max_vel)
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
        red_positions = self.agent_positions[:self.N]
        blue_positions = self.agent_positions[self.N:]
        # Append red agents' observations
        for i in range(self.N):
            observation.append(
                (self.agent_positions[i], self.red_flag_pos, self.blue_flag_pos,
                 red_positions, blue_positions)
            )
        # Append blue agents' observations
        for i in range(self.N, self.N*2):
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
        controls = controls + (np.random.random((self.N * 2, 2)) - 0.5) / 20
        # Clip positions to be inside the walls
        for i in range(self.N):
            pos = self.apply_control(controls)
            pos[:, 0] = pos[:, 0].clip(0, self.l)
            pos[:, 1] = pos[:, 1].clip(0, self.h)

        # Compute the distance between each pair of agents
        dists_table = np.linalg.norm(pos[:,None,:] - pos, axis=2)

        # Check for collision and apply game dynamics
        # TODO: vectorize the code here
        for i in range(self.N):
            for j in range(self.N, self.N*2):
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
            self.agent_positions[self.N:, :] - np.array(self.red_flag_pos),
            axis=1)
        red_flag_captured = np.any(dist_to_red_flag < self.radius)

        dist_to_blue_flag = np.linalg.norm(
            self.agent_positions[:self.N, :] - np.array(self.blue_flag_pos),
            axis=1)
        blue_flag_captured = np.any(dist_to_blue_flag < self.radius)

        self.step_count += 1
        game_ended = red_flag_captured or blue_flag_captured or \
            self.step_count >= self.time_limit

        return observation, game_ended, \
            red_flag_captured, blue_flag_captured