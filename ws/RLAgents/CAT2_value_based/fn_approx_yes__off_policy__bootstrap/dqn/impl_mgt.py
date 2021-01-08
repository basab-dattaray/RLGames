
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.activations import relu, linear
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizer_v2.adam import Adam

from .replay_mgt import fn_replay_mgt

import random
import numpy as np
import os


def impl_mgt(app_info, state_size, action_size):
    _epsilon = app_info['EPSILON']
    _gamma = app_info['GAMMA']
    _batch_size = app_info['BATCH_SIZE']
    _epsilon_min = app_info['EPSILON_MIN']
    _learning_rate = app_info['LEARNING_RATE']
    _epsilon_decay = app_info['EPSILON_DECAY']
    # _memory = None
    _model = None

    fn_remember_for_replay = None
    fn_get_mini_batch = None

    def fn_reset_env():
        nonlocal _model, fn_remember_for_replay, fn_get_mini_batch

        fn_remember_for_replay, fn_get_mini_batch = fn_replay_mgt()
        _model = fn_build_model()

    def fn_build_model():
        model = Sequential()
        model.add(Dense(150, input_dim=state_size, activation=relu))
        model.add(Dense(120, activation=relu))
        model.add(Dense(action_size, activation=linear))
        model.compile(loss='mse', optimizer=Adam(lr=_learning_rate))
        return model

    def fnAct(state):

        if np.random.rand() <= _epsilon:
            return random.randrange(action_size)
        act_values = _model.predict(state)
        return np.argmax(act_values[0])

    def fnReplay():
        nonlocal _epsilon, fn_get_mini_batch

        minibatch = fn_get_mini_batch()
        if minibatch is None:
            return

        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + _gamma * (np.amax(_model.predict_on_batch(next_states), axis=1)) * (1 - dones)
        targets_full = _model.predict_on_batch(states)
        ind = np.array([i for i in range(_batch_size)])
        targets_full[[ind], [actions]] = targets

        _model.fit(states, targets_full, epochs=1, verbose=0)
        if _epsilon > _epsilon_min:
            _epsilon *= _epsilon_decay

    def fnSaveWeights(model_dir):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        path = os.path.join(model_dir, 'model.h5')
        _model.save_weights(path)

    def fnLoadWeights(model_dir):
        nonlocal _model

        path = os.path.join(model_dir, 'model.h5')
        if os.path.exists(path):
            _model.load_weights(path)
        j1 = 1

    fn_reset_env()

    return fn_reset_env, fn_remember_for_replay, fnAct, fnReplay, fnSaveWeights, fnLoadWeights
