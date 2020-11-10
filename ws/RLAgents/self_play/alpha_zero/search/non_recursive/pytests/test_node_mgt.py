import pytest

# from ws.RLEnvironments.self_play_games.othello.game_mgt import game_mgt
from ws.RLAgents.model_free.policy_gradient.agent_mgt import agent_mgt
from ..node_mgt import node_mgt
from ...mcts_adapter import mcts_adapter
from ws.RLEnvironments.self_play_games.othello.game_mgt import game_mgt

GAME_SIZE= 5

def init():
    from ws.RLUtils.monitoring.tracing.trace_example.agent_caller import args
    agent = agent_mgt(args, __file__)
    game = game_mgt(GAME_SIZE)
    pmcts = mcts_adapter(nn_mgr_P, args)

def test_create_root_node():

    root_node = node_mgt(
        state,
        fn_get_normalized_predictions,
        max_num_actions,
        explore_exploit_ratio,

        parent_action=-1,
        val=0.0,
        parent_node=None
    )




