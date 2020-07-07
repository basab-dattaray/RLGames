from ws.RLAgents.self_play.alpha_0.misc.utils import dotdict

args = dotdict({
    'numIters': 2,
    'numEps': 2,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.0,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 15,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 6,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/5x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 4,

    'board_size': 5,
    'mcts_recursive': 1,
    'test_num_games': 8

})