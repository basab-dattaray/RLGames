from ws.RLAgents.self_play.alpha_zero.misc.utils import dotdict

args = dotdict({
    'numIters': 5,
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 35,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': 'tmp.1/',
    'load_model': False,
    'load_folder_file': ('tmp.1/','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

    'epochs': 10,
    'board_size': 5,
    'num_of_test_games': 500,
    'mcts_recursive': False,
})