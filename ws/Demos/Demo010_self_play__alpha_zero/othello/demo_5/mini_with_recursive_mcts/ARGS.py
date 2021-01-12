from ws.RLUtils.common.DotDict import DotDict

args = DotDict({
    'STRATEGY': 'CAT4_self_play.alpha_zero',
    'NUM_TRAINING_ITERATIONS': 3,
    'NUM_TRAINING_EPISODES': 3,
    'PROBABILITY_SPREAD_THRESHOLD': 15,
    'SCORE_BASED_MODEL_UPDATE_THRESHOLD': 0.5,
    'SAMPLE_BUFFER_SIZE': 200000,
    'NUM_MC_SIMULATIONS': 27,
    'NUM_GAMES_FOR_MODEL_COMPARISON': 4,
    'EXPLORE_EXPLOIT_FACTOR': 1,

    'MODEL_NAME': 'model.tar',

    'DO_LOAD_MODEL': True,

    'SAMPLE_HISTORY_BUFFER_SIZE': 20,

    'NUM_EPOCHS': 4,
    'BOARD_SIZE': 5,
    'NUM_TEST_GAMES': 8,

    'UCB_USE_LOG_IN_NUMERATOR': True,
    'UCB_USE_POLICY_FOR_EXPLORATION': True,

})

def fn_get_args():
    return args