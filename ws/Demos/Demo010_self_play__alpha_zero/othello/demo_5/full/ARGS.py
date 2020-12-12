from ws.RLUtils.common.DotDict import DotDict

args = DotDict({
    'num_of_training_iterations': 5,
    'num_of_training_episodes': 50,              # Number of complete self-play games to simulate during a new iteration.
    'probability_spread_threshold': 0,        #
    'score_based_model_update_threshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'sample_buffer_size': 200000,    # Number of game examples to train the neural networks.
    'num_of_mc_simulations': 35,          # Number of games moves for MCTS to simulate.
    'number_of_games_for_model_comarison': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct_exploration_exploitation_factor': 1,

    'do_load_model': True,
    # 'rel_model_path': 'tmp/',
    # 'do_load_samples': False,
    # 'model_name':'model.tar',
    # 'load_folder_file': ('tmp/','model.tar'),
    'sample_history_buffer_size': 20,

    'epochs': 20,
    'board_size': 5,
    'num_of_test_games': 300,

    'mcts_ucb_use_log_in_numerator': True,
    'mcts_ucb_use_action_prob_for_exploration': False,

    'run_recursive_search': False,
})