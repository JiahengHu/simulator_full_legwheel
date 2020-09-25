
from collections import namedtuple
import random

## Replay Memory
# We use experience replay memory for training the DQN.
Transition = namedtuple('Transition',
                        ('state', 'terrain', 'action', 'next_state', 'reward', 'non_final'))
class replay_buffer(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        transitions= random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        return batch
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
            
    def __len__(self):
        return len(self.memory)

