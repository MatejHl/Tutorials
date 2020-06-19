import numpy as np

class User:
    """
    """
    def __init__(self, name, state_space_size, action_space_size):
        self.name = name
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size


    def act(self, state, **kwargs):
        allowed = False
        while not allowed:
            action = int(input('Enter action:    '))
            if action in state.allowedActions:
                allowed = True
        pi = np.zeros(self.action_space_size, dtype = np.integer)
        pi[action] = 1.0
        
        return action, pi
