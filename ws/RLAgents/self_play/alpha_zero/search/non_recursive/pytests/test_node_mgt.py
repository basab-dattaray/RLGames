import pytest

from ws.RLAgents.self_play.alpha_zero.misc.agent_mgt import agent_mgt
from ws.RLAgents.self_play.alpha_zero.search.non_recursive.pytests.ARGS import args
# from ..node_mgt import node_mgt
# from ...mcts_adapter import mcts_adapter
# from ws.RLEnvironments.self_play_games.othello.game_mgt import game_mgt

GAME_SIZE= 5

@pytest.fixture()
def init_agent():

    agent = agent_mgt(args, __file__)
    return agent


def test_create_root_node(init_agent):
    # agent = init_agent(args, __file__)
    args = init_agent.arguments

    x = 3

    # root_node = node_mgt(
    #     state,
    #     fn_get_normalized_predictions,
    #     max_num_actions,
    #     explore_exploit_ratio,
    #
    #     parent_action=-1,
    #     val=0.0,
    #     parent_node=None
    # )




