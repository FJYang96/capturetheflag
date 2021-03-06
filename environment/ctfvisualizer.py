from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from celluloid import Camera

class CTFRenderer:
    def __init__(self, num_players, l, h, frame_duration=0.01):
        self.num_players = num_players
        self.l = l
        self.h = h
        self.frame_duration = frame_duration

    def create_canvas(self):
        pass

    def plot_agents(self, agent_positions):
        ''' No return
        Plot the players as colored circles on the axis
        '''
        plt.scatter(agent_positions[:self.num_players[0],0],
                    agent_positions[:self.num_players[0],1], \
                    marker='o', color='r')
        plt.scatter(agent_positions[self.num_players[0]:,0],
                    agent_positions[self.num_players[0]:,1], \
                    marker='o', color='b')

    def plot_flags(self, flag_positions):
        ''' No return
        Plot the flags as colored crosses on the axis
        '''
        plt.scatter(flag_positions[0][0], flag_positions[0][1], \
                    marker='x', color='r')
        plt.scatter(flag_positions[1][0], flag_positions[1][1], \
                    marker='x', color='b')

    def plot_step(self, agent_positions, flag_positions, clearfig=True):
        ''' No return
        Plots the field and the agents in the field
        '''
        if clearfig:
            plt.clf()
        plt.axis([0, self.l, 0, self.h])
        self.plot_agents(agent_positions)
        self.plot_flags(flag_positions)

    def render_step(self, agent_positions, flag_positions):
        ''' No return
        Renders the field by calling plt.pause for the specified frame duration
        '''
        self.plot_step(agent_positions, flag_positions)
        plt.pause(self.frame_duration)

    def mp4_from_history(self, history, flag_positions, gif_dir):
        '''
        Note: can use "ffmpeg -i [input].mp4 [out].gif" to convert this into
        a gif file
        '''
        camera = Camera(plt.figure())
        for s in history:
            self.plot_step(np.array(s), flag_positions, clearfig=False)
            camera.snap()
        anim = camera.animate()
        anim.save(gif_dir)

    def gif_from_history(self, history, flag_positions, gif_dir):
        pass

