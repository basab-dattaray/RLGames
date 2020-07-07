from ws.Demos.Demo010_self_play__alpha_zero.othello.demo_5.non_recursive.ConfigParams import ConfigParams
from ws.RLAgents.self_play.alpha_zero.Agent import  Agent
#  PolicyAlgos: MCTS, DIRECT_MODEL, MCTS_RECURSIVE
if __name__ == "__main__":
    Agent.fn_init( __file__, ConfigParams())\
        .fn_create_new_session()\
        .fn_change_configs(configs=
                {
                    'num_testing_eval_games': 200,
                      'POLICY_RULES': {
                        'POLICY_TESTING_EVAL': 'MCTS'}})\
        .fn_pit_against_human()