class Agent:
    '''
    Abstract class of agents.
    '''
    def __init__(self, num_players):
        self.num_players = None

    def get_action(self, observation):
        '''
        This is the general interface for agents. They take an observation and
        return a control output
        '''
        raise NotImplementedError

class TwoTierAgent(Agent):
    '''
    Abstract class of two-tier agents.
    Two tier agents make decisions with a two-step process:
        1. decide on a meta-strategy
        2. given the meta-strategy, decide on a specific action
    '''
    def __init__(self, num_players):
        super().__init__(num_players)
        self.meta_strategy = -1
        self.meta_strategy_target = -1

    def get_meta_strategy(self, observation):
        raise NotImplementedError

    def get_action(self, observation):
        raise NotImplementedError
