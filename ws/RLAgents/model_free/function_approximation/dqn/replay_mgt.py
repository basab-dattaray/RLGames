from collections import deque
import random


def fn_replay_mgr(mem_size=1000000, mini_batch_size=64):
    _memory = deque(maxlen=mem_size)

    _mini_batch_size = mini_batch_size

    def fn_remember(state, action, reward, next_state, done):
        nonlocal _memory
        _memory.append((state, action, reward, next_state, done))

    def fn_get_mini_batch():
        if len(_memory) < _mini_batch_size:
            return None

        minibatch = random.sample(_memory, _mini_batch_size)
        return minibatch

    return fn_remember, fn_get_mini_batch
