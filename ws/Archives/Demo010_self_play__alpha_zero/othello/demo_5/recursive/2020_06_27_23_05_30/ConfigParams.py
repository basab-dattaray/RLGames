

from ws.RLAgents.self_play.alpha_zero_old.othello.pytorch.utils import dotdict


class ConfigParams():

    args = dotdict({
        'num_iterations': 4,
        'num_model_upgrades': 6,
        'num_episodes': 10,              # Number of complete self-play games to simulate during a new iteration.
        'model_acceptance_win_ratio': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
        'win_ratio_reduction_rate': 11,
        'sample_buffer_size': 200000,    # Number of game examples to train the neural networks.
        'num_of_mcts_simulations': 25,          # Number of games moves for MCTS to simulate.
        'num_training_eval_games': 10,         # Number of games to play during arena play to determine if new net will be accepted.
        'num_testing_eval_games': 20,
        'cpuct': 1,

        'num_sample_buffers': 3,

        'board_size': 5,

        'skip_log_interval': 3,
        'mean_log_interval' :1,
        'debug_board': 1,

        'epochs': 3,

        'model_loss_val_fraction': 0.5,

        #  PolicyAlgos: MCTS, DIRECT_MODEL, MCTS_RECURSIVE
        'POLICY_RULES': {
            'POLICY_TRAINING_EXECUTION': 'MCTS_RECURSIVE',
            'POLICY_TRAINING_EVAL': 'MCTS_RECURSIVE',
            'POLICY_TESTING_EVAL': 'MCTS_RECURSIVE'
        }
    })