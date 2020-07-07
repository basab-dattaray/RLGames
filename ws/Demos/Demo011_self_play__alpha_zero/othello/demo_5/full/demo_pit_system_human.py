from ws.RLAgents.self_play.alpha_0.misc.Agent import Agent
from ws.Demos.Demo011_self_play__alpha_zero.othello.demo_5.full.CONFIG import args


if __name__ == "__main__":
    Agent.fn_init(args, __file__).\
        fn_change_configs({
            'mcts_recursive': 0,
            'numMCTSSims': 50,
            'test_num_games': 2,
        }).\
        fn_pit_system_human()
