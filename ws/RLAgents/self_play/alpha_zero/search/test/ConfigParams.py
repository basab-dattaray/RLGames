

from ws.RLAgents.self_play.alpha_zero.othello.pytorch.utils import dotdict


class ConfigParams():

    args = dotdict({
        'num_iterations': 2,
        'num_episodes': 6,              # Number of complete self-play games to simulate during a new iteration.
        'tempThreshold': 15,        #
        'model_acceptance_win_ratio': 0.2,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
        'sample_buffer_size': 200000,    # Number of game examples to train the neural networks.
        'num_of_mcts_simulations': 10,          # Number of games moves for MCTS to simulate.
        'num_training_eval_games': 8,         # Number of games to play during arena play to determine if new net will be accepted.
        'cpuct': 1,

        'num_sample_buffers': 4,

        'board_size': 5,

        'skip_log_interval': 1,
        'mean_log_interval' :1,

        'debug_board': 1,

        'epochs': 5,

        'model_loss_val_fraction': 0.5,

        #  PolicyAlgos: MCTS, DIRECT_MODEL, MCTS_RECURSIVE
        'POLICY_RULES': {
            'POLICY_TRAINING_EXECUTION': 'MCTS',
            'POLICY_TRAINING_EVAL': 'DIRECT_MODEL',
            'POLICY_TESTING_EVAL': 'MCTS_RECURSIVE'
        }

    })