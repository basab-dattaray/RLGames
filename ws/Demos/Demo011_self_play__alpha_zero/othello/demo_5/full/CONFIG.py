from ws.RLAgents.self_play.alpha_0.misc.utils import dotdict

args = dotdict({
    'numIters': 7,
    'numEps': 10,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/5x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

    'board_size': 5,

    'test_num_games': 500

})