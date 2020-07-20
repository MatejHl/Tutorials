from collections import deque
import pickle


class Memory:
    """
    For different types of games few modifications will be probablly needed.
    Here I assume two player and symmetric win/loose rewards.
    """
    def __init__(self, config):
        self.memory_size = config.MEMORY_SIZE
        # deque is list-like but with more efficient appending.
        self.shortMemory = deque(maxlen = self.memory_size)
        self.longMemory = deque(maxlen = self.memory_size)

    def commit_shortMemory(self, state, pi):
        # for i in identities(state, pi):
        self.shortMemory.append({'state' : state,
                                 'id' : state.id,
                                 'policy' : pi,
                                 'playerTurn' : state.playerTurn})


    def add_value_shortMemory(self, value, looser):
        for move in self.shortMemory:
            if move['playerTurn'] == looser:
                move['value'] = value
            else:
                move['value'] = -value


    def short_to_longMemory(self):
        for row in self.shortMemory:
            self.longMemory.append(row)
        self.clear_shortMemory()


    def clear_shortMemory(self):
        self.shortMemory = deque(maxlen = self.memory_size)


    def save_longMemory(self, pklfile):
        with open(pklfile, 'wb') as file:
            pickle.dump(self.longMemory, file)


    def load_longMemory(self, pklfile):
        with open(pklfile, 'rb') as file:
            self.longMemory = pickle.load(file)