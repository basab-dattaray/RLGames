from ws.Demos.Demo010_self_play__alpha_zero.othello.demo_5.recursive.ConfigParams import ConfigParams
from ws.RLAgents.self_play.alpha_zero.Agent import Agent

if __name__ == "__main__":
    Agent.fn_init(__file__, ConfigParams()) \
 \
        .fn_clean() \
        .fn_create_new_session() \
        .fn_change_configs(configs=
    {
        'model_acceptance_win_ratio': 0.95,
        'num_training_eval_games': 4,
        'POLICY_RULES': {  # PolicyAlgos: MCTS, DIRECT_MODEL, MCTS_RECURSIVE
            'POLICY_TRAINING_EXECUTION': 'MCTS_RECURSIVE',
            'POLICY_TRAINING_EVAL': 'MCTS_RECURSIVE'},

        'num_episodes': 3,
        'num_iterations': 2,
        'epochs': 2,
        'num_of_mcts_simulations': 12,
    }) \
        .fn_train() \
 \
        .fn_archive_it()