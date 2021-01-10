from ws.RLUtils.common.DotDict import DotDict

args = DotDict({
    'STRATEGY': 'CAT4_self_play.alpha_zero',
    'NUM_TRAINING_ITERATIONS': 3,
    'NUM_TRAINING_EPISODES': 3,              # Number of complete self-play games to simulate during a new iteration.
    'PROBABILITY_SPREAD_THRESHOLD': 15,        #
    'SCORE_BASED_MODEL_UPDATE_THRESHOLD': 0.5,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'SAMPLE_BUFFER_SIZE': 200000,    # Number of game examples to train the neural networks.
    'NUM_MC_SIMULATIONS': 27,          # Number of games moves for MCTS to simulate.
    'NUM_GAMES_FOR_MODEL_COMPARISON': 4,         # Number of games to play during arena play to determine if new net will be accepted.
    'EXPLORE_EXPLOIT_FACTOR': 1,

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