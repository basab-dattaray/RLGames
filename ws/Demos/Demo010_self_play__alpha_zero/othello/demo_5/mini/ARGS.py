from ws.RLAgents.self_play.alpha_zero.misc.utils import dotdict

args = dotdict({
    'numIters': 3,
    'numEps': 3,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.0,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 4,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': 'tmp/',
    'load_model': False,
    'load_folder_file': ('tmp/','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

    'epochs': 3,
    'board_size': 5,
    'num_of_test_games': 8,
    # 'mcts_recursive': 0,

})