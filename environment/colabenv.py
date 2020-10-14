import numpy as np
from .colabvisualizer import ColabRenderer
#from .io import ctflogger

EPS = 1e-8

class ColabSim:
    def __init__(self,
                 n_agent,
                 n_adversary,
                 ob_type='full',
                 penalty_weight=np.array([[2,50,1.5],[2,50,1.5]]),
                 max_vels=None,
                 initial_pos=np.array([[10,10],[30,10],[80,15],[10,50]]),
                 initial_dx=np.zeros(7),
                 time_limit=100,
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
        # Basic info
        self.num_players = np.array([n_agent,n_adversary])
        self.N = np.sum(self.num_players)
        self.ob_limit = 20
        self.ob_type = ob_type
        self.penalty_weight = penalty_weight

        # Field configuration
        self.h = 100
        self.l = 100
        self.radius = 0.8

        self.red_initial_pos = initial_pos[:n_agent]
        self.blue_initial_pos = initial_pos[n_agent:]
        self.red_goal_pos = np.array([[70,80],[90,80]])
        self.blue_goal_pos = np.array([[20,80], [90,50]])
        self.goal_pos = np.vstack([self.red_goal_pos, self.blue_goal_pos])

        # Set initial_pos and initial_vel to be a 5-tuple that corresponds to
        # (bar midpoint x, bar midpoint y, bar angle, adversary x, adversary y)
        # The internal states of the system will always be represented this way
        x = self.positions_to_x(self.red_initial_pos)
        self.initial_x = np.concatenate([x, self.blue_initial_pos.flatten()])
        self.initial_dx = initial_dx

        # Set the maximum velocity of the agents
        if max_vels is None:
            self.max_vels = np.array([1, 1, 0.05, 1, 1, 1, 1])
        else:
            self.max_vels = max_vels

        # Set simulation step limit
        self.dt = 1
        self.time_limit = time_limit
        self.adversary_control_weights = (5, 0.4) # adversary goal pull vs collision push

        # Set control noise
        self.noise_var = noise_var

        ##########################################################################
        # TODO: implement logger for the colab transport environment 
        ##########################################################################

        # Set whether to render the simulation or not
        self.render = render
        self.renderer = ColabRenderer(self.num_players, self.l, self.h)

        # initialize a logger
        #self.logger = ctflogger(num_players, self.l, self.h, self.flag_positions)
        self.reset()

    def reset(self):
        ''' Returns a tuple
        Resets the environment to its initial condition
        '''
        self.x = self.initial_x.copy()
        self.dx = self.initial_dx.copy()
        self.agent_positions = np.array([[1,1],[3,1],[8,1.5],[1,5]]) * 10
        self.step_count = 0
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
        positions = self.x_to_positions(self.x)
        bar_direction = positions[1] - positions[0]
        torque = float(np.cross(controls[0], bar_direction)) + \
                float(np.cross(controls[1], -bar_direction))
        self.dx[2] = self.dx[2] + self.dt * torque

    def x_to_positions(self, x):
        x_mid, y_mid, theta, x_ad1, y_ad1, x_ad2, y_ad2 = x
        displacement = np.array([np.cos(theta), np.sin(theta)]) * 10
        positions = np.array([[x_mid-displacement[0], y_mid-displacement[1]],
                              [x_mid+displacement[0], y_mid+displacement[1]],
                              [x_ad1, y_ad1],
                              [x_ad2, y_ad2]])
        return positions

    def positions_to_x(self, positions):
        midpoint = positions.mean(0)
        diff = positions[1] - positions[0]
        angle = np.arctan(diff[1]/diff[0])
        return np.array([midpoint[0], midpoint[1], angle])

    def partial_observation(self):
        observation = []
        positions = self.x_to_positions(self.x)
        # Append red agents' observations
        for i in range(self.num_players[0]):
            enemy_observation = np.array([positions[3-i]]) #0 observes 3, 1 observes 2
            observation.append((positions[i], positions[1-i], \
                        self.red_goal_pos[i], self.red_goal_pos[1-i], \
                        enemy_observation)
            )
        return observation

    def full_observation(self):
        observation = []
        positions = self.x_to_positions(self.x)
        # Append red agents' observations
        for i in range(self.num_players[0]):
            sim_params = (positions, self.x.copy(), self.dx.copy(), self.penalty_weight, i)
            enemy_observation = positions[2:] #0 observes 3, 1 observes 2
            observation.append((positions[i], positions[1-i], \
                        self.red_goal_pos[i], self.red_goal_pos[1-i], \
                        enemy_observation, sim_params)
            )
        return observation

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
        if self.ob_type=='partial':
            observation = self.partial_observation()
        else:
            observation = self.full_observation()
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
        self.apply_control(controls)
        '''
        positions = self.x_to_positions(self.x)[0:2]
        self.agent_vel = self.agent_vel + self.dt * controls
        self.agent_vel = self.agent_vel.clip(-self.max_vels[0], self.max_vels[0])
        new_pos = positions + self.dt * self.agent_vel
        agent_x = self.positions_to_x(new_pos)
        '''

        # Adversary control
        for i in range(2):
            pos = self.x[3+i*2:3+i*2+2]
            goal_diff = self.blue_goal_pos[i] - pos
            goal_attraction = goal_diff / (np.linalg.norm(goal_diff)**2 + EPS)
            agent_diff = pos - self.x_to_positions(self.x)[:2]
            agent_avoidance = agent_diff / \
                    (np.linalg.norm(agent_diff, axis=1)**2)[:, None]
            adv_control = goal_attraction * self.adversary_control_weights[0] +\
                    agent_avoidance.sum(0) * self.adversary_control_weights[1]
            self.dx[3+i*2:3+i*2+2] = self.dx[3+i*2:3+i*2+2] + self.dt * adv_control
        self.dx = self.dx.clip(-self.max_vels, self.max_vels)

        # Apply dx to x
        self.x = self.x + self.dt * self.dx
        #self.x[:3] = agent_x

        # Recompute positions after normalizing for bar position
        positions = self.x_to_positions(self.x)

        # Compute penalty
        def distance(p1, p2, x):
            rhs = np.dot(x, p2-p1) - np.dot(p2, p2-p1)
            lhs = np.dot(p1, p2-p1) - np.dot(p2, p2-p1)
            alfa = rhs / lhs
            alfa = np.clip(alfa, 0, 1)
            proj = alfa * p1 + (1-alfa) * p2
            return np.linalg.norm(x-proj)
        dist_ad1_bar = distance(positions[0], positions[1], self.x[3:5])
        dist_ad2_bar = distance(positions[0], positions[1], self.x[5:7])
        dist_ad_bar = min(dist_ad1_bar, dist_ad2_bar)
        distance_penalty = 1 / (dist_ad_bar + EPS)
        penalty = np.dot(self.penalty_weight[:,:2], (1, distance_penalty))
        control_cost = np.linalg.norm(controls, axis=1)
        penalty += control_cost * self.penalty_weight[:,2]

        # Pack observations
        observation = self.pack_observation()

        # Render if asked
        if self.render:
            self.renderer.render_step(self.x_to_positions(self.x), controls)

        # Check for whether the game has ended
        positions = self.x_to_positions(self.x)
        dist_to_goal = np.linalg.norm(positions[:2] - self.goal_pos[:2], axis=1)[:2]
        self.step_count += 1
        game_ended = False
        if np.all(dist_to_goal <= self.radius * 1.5):
            game_ended = True
        elif self.step_count >= self.time_limit:
            game_ended = True
            penalty += 10 * dist_to_goal
            #penalty += 1000

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
