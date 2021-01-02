class Buffer:

    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.done = []
        self.state_values = []

    def clear_buffer(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.done[:]
        del self.state_values[:]  # needed only for ActorCritic
