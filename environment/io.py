import json
import numpy as np
from .ctfvisualizer import CTFRenderer

class ctflogger:
    def __init__(self, num_players, l, h, flag_positions):
        self.num_players = num_players
        self.l = l
        self.h = h
        self.flag_positions = flag_positions
        self.position_history = []
        self.renderer = None

    def log_state(self, state):
        self.position_history.append(state.tolist())

    def clear_history(self):
        self.position_history = []

    def dump(self):
        info = {
            'num_players': self.num_players,
            'states': self.position_history
        }
        return json.dumps(info)

    def dump_to_file(self, filename):
        with open(filename, 'w') as f:
            f.write(self.dump())

    def load_from_file(self, filename):
        info = None
        with open(filename, 'r') as f:
            info = json.loads(f.read())
        self.num_players = info['num_players']
        self.position_history = info['states']

    def render_history(self):
        if self.renderer is None:
            self.renderer = CTFRenderer(self.num_players, self.l, self.h)
        for s in self.position_history:
            self.renderer.render_step(np.array(s), self.flag_positions)

    def render_from_file(self, filename):
        self.load_from_file(filename)
        self.render_history()

    def mp4_from_file(self, gif_dir):
        if self.renderer is None:
            self.renderer = CTFRenderer(self.num_players, self.l, self.h)
        self.renderer.mp4_from_history(
            self.position_history, self.flag_positions, gif_dir)
