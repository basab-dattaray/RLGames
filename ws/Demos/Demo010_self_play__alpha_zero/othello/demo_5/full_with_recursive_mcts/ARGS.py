from ws.RLUtils.common.DotDict import DotDict

args = DotDict({
    'NUM_TRAINING_ITERATIONS': 5,
    'NUM_TRAINING_EPISODES': 50,              # Number of complete self-play games to simulate during a new iteration.
    'PROBABILITY_SPREAD_THRESHOLD': 0,        #
    'SCORE_BASED_MODEL_UPDATE_THRESHOLD': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'SAMPLE_BUFFER_SIZE': 200000,    # Number of game examples to train the neural networks.
    'NUM_MC_SIMULATIONS': 25,          # Number of games moves for MCTS to simulate.
    'NUM_GAMES_FOR_MODEL_COMPARISON': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'EXPLORE_EXPLOIT_FACTOR': 1,

    'MODEL_NAME': 'model.tar',

    'DO_LOAD_MODEL': True,

    'SAMPLE_HISTORY_BUFFER_SIZE': 20,

    'NUM_EPOCHS': 20,
    'BOARD_SIZE': 5,
    'NUM_TEST_GAMES': 300,

    'UCB_USE_LOG_IN_NUMERATOR': True,
    'UCB_USE_POLICY_FOR_EXPLORATION': True,
})

def fn_get_args():
    return args