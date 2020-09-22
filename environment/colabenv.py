import numpy as np
from .colabvisualizer import ColabRenderer
#from .io import ctflogger

EPS = 1e-8

class ColabSim:
    def __init__(self,
                 max_vels=None,
                 time_limit=150,
                 noise_var=0.25,
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
        self.num_players = np.array([2,1])
        self.N = 3
        self.h = 100
        self.l = 100
        self.radius = 0.8
        self.ob_limit = 20
        self.dt = 1
        #How do adversary weight goal and collision avoidance
        self.adversary_control_weights = (5, 0.4)

        self.red_initial_pos = np.array([[10,10],[30,10]])
        self.red_goal_pos = np.array([[70,80],[90,80]])
        self.blue_initial_pos = np.array([80,15])
        self.blue_goal_pos = np.array([20,80])
        self.goal_pos = np.vstack([self.red_goal_pos, self.blue_goal_pos[None, :]])
        # Set initial_pos and initial_vel to be a 5-tuple that corresponds to
        # (bar midpoint x, bar midpoint y, bar angle, adversary x, adversary y)
        # The internal states of the system will always be represented this way
        self.initial_x = np.array([20,10,0,80,15])
        self.initial_dx = np.zeros(5)

        # Set the maximum velocity of the agents
        if max_vels is None:
            self.max_vels = np.array([1, 1, 0.05, 1, 1])
        else:
            self.max_vels = max_vels

        # Set simulation step limit
        self.time_limit = time_limit

        # Set control noise
        self.noise_var = noise_var

        ##########################################################################
        # TODO: implement logger for the colab transport environment 
        ##########################################################################

        # Set whether to render the simulation or not
        self.render = render
        if self.render:
            self.renderer = ColabRenderer(self.num_players, self.l, self.h)

        # initialize a logger
        #self.logger = ctflogger(num_players, self.l, self.h, self.flag_positions)
        self.reset()

    def reset(self):
        ''' Returns a tuple
        Resets the environment to its initial condition
        '''
        self.x = self.initial_x
        self.dx = self.initial_dx
        self.agent_positions = np.array([[1,1],[3,1],[8,1.5]]) * 10
        self.step_count = 0
        # TODO
        self.agent_vel = np.zeros((2,2))
        #self.logger.clear_history()
        if self.render:
            self.renderer.render_step(self.agent_positions)

        return (self.pack_observation(), False)


    def apply_control(self, controls):
        ''' Does not return; By-product: set self.dx
        Computes the position of agents given the current control
        Returns the projected position of the agents (ignoring collisions)
        '''
        # Bar midpoint velocity
        self.dx[0:2] = self.dx[0:2] + self.dt * 1/2 * (controls[0] + controls[1])
        # Bar angular velocity
        bar_direction = self.agent_positions[1] - self.agent_positions[0]
        torque = float(np.cross(controls[0], bar_direction)) + \
                float(np.cross(controls[1], -bar_direction))
        self.dx[2] = self.dx[2] + self.dt * torque

    def x_to_positions(self, x):
        x_mid, y_mid, theta, x_ad, y_ad = x
        displacement = np.array([np.cos(theta), np.sin(theta)]) * 10
        positions = np.array([[x_mid-displacement[0], y_mid-displacement[1]],
                              [x_mid+displacement[0], y_mid+displacement[1]],
                              [x_ad, y_ad]])
        return positions

    def positions_to_x(self, positions):
        midpoint = positions.mean(0)
        diff = positions[1] - positions[0]
        angle = np.arctan(diff[1]/diff[0])
        return np.array([midpoint[0], midpoint[1], angle])

    def pack_observation(self):
        ''' Returns a list of tuples
        Pack the observations for all agents
        Return:
            observation - an array where each element is the observation of
            the corresponding agent
        Note:
            for red agent, the observation is a 4-tuple of
            (self_pos, teammate_pos, goal_pos, enemy_pos)
            for blue agent, the observation is also a 3-tuple of
            (self_pos, goal_pos, red_pos)
        '''
        observation = []
        positions = self.x_to_positions(self.x)
        # Append red agents' observations
        for i in range(self.num_players[0]):
            dist_to_adversary = np.linalg.norm(positions[i] - positions[2])
            enemy_observation = positions[2] \
                    if dist_to_adversary <= self.ob_limit else None
            observation.append((positions[i], positions[1-i], \
                        self.red_goal_pos[i], self.red_goal_pos[1-i], \
                        enemy_observation)
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
        #print('-'*15 + 'Stepping' + '-'*15)
        #print('Current adversary position:', self.x[3:])

        # Apply controls to agents
        #self.apply_control(controls)
        positions = self.x_to_positions(self.x)[0:2]
        self.agent_vel = self.agent_vel + self.dt * controls
        self.agent_vel = self.agent_vel.clip(-self.max_vels[0], self.max_vels[0])
        new_pos = positions + self.dt * self.agent_vel
        agent_x = self.positions_to_x(new_pos)

        # Adversary control
        goal_diff = self.blue_goal_pos - self.x[3:]
        goal_attraction = goal_diff / (np.linalg.norm(goal_diff)**2 + EPS)
        agent_diff = self.x[3:] - self.x_to_positions(self.x)[:2]
        agent_avoidance = agent_diff / (np.linalg.norm(agent_diff, axis=1)**2)[:, None]
        adv_control = goal_attraction * self.adversary_control_weights[0] +\
                agent_avoidance.sum(0) * self.adversary_control_weights[1]
        self.dx[3:] = self.dx[3:] + self.dt * adv_control
        self.dx = self.dx.clip(-self.max_vels, self.max_vels)

        # Apply dx to x
        self.x = self.x + self.dt * self.dx
        self.x[:3] = agent_x

        # Recompute positions after normalizing for bar position
        positions = self.x_to_positions(self.x)

        # Compute penalty
        time_penalty = 1
        bar_points = np.array([positions[0], positions[1], self.x[:2]])
        dist_ad_bar = np.min( np.linalg.norm(bar_points - self.x[3:], axis=1) )
        distance_penalty = 50 / (dist_ad_bar + EPS)
        penalty = time_penalty + distance_penalty

        # Pack observations
        observation = self.pack_observation()

        # Render if asked
        if self.render:
            self.renderer.render_step(self.x_to_positions(self.x))

        # Check for whether the game has ended
        positions = self.x_to_positions(self.x)
        dist_to_goal = np.linalg.norm(positions - self.goal_pos, axis=1)[:1]
        self.step_count += 1
        game_ended = np.all(dist_to_goal <= self.radius) or \
            self.step_count >= self.time_limit

        #self.logger.log_state(self.agent_positions)

        return observation, penalty, game_ended

    def collision_dynamics(self):
        '''
        Just storing these lines of code here; to use them copy/paste into self.step()
        '''
        # Check for collision
        v = positions[1] - positions[0]
        u = positions[1] - positions[2]
        dist_ad_bar = np.dot(u, u) - np.dot(u, v) ** 2 / np.dot(v, v)
        if dist_ad_bar <= 2*self.radius:
            # Elastic Collision Dynamics
            new_v_bar = 1/3 * self.dx[3:]
            new_v_ad = 3 * self.dx[:2]
            self.dx[:2] = new_v_bar
            self.dx[3:] = new_v_ad
        # Check for boundary
        if np.any(positions[:2,0] < 0) or np.any(positions[:2,0] > self.l):
            self.dx[0] = 0
            self.dx[2] = 0
        if np.any(positions[:2,1] < 0) or np.any(positions[:2,1] > self.h):
            self.dx[1] = 0
            self.dx[2] = 0
        x_ad, y_ad = self.x[3:]
        if x_ad < 0 or x_ad > self.l:
            self.dx[3] = 0
        if y_ad < 0 or y_ad > self.h:
            self.dx[4] = 0

    def get_logger(self):
        return self.logger
