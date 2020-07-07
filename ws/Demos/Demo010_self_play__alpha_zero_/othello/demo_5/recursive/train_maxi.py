from ws.Demos.Demo010_self_play__alpha_zero_.othello.demo_5.recursive.ConfigParams import ConfigParams
from ws.RLAgents.self_play.alpha_zero.Agent import Agent

if __name__ == "__main__":
    Agent.fn_init(__file__, ConfigParams()) \
 \
        .fn_clean() \
        .fn_create_new_session() \
        .fn_change_configs(configs=
    {
        'model_acceptance_win_ratio': 0.55,
        'num_training_eval_games': 14,
        'POLICY_RULES': {  # PolicyAlgos: MCTS, DIRECT_MODEL, MCTS_RECURSIVE
            'POLICY_TRAINING_EXECUTION': 'MCTS_RECURSIVE',
            'POLICY_TRAINING_EVAL': 'MCTS_RECURSIVE'},

        'num_episodes': 10,
        'num_iterations': 8,
        'epochs': 10,
        'num_of_mcts_simulations': 30,
    }) \
        .fn_train() \
 \
        .fn_archive_it()