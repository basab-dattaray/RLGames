import pytest

from ws.RLAgents.self_play.alpha_zero.misc.agent_mgt import agent_mgt, fn_init_arg_with_default_val
from ws.RLAgents.self_play.alpha_zero.search.mcts_adapter import mcts_adapter
from ws.RLAgents.self_play.alpha_zero.search.non_recursive.node_mgt import node
from ws.RLAgents.self_play.alpha_zero.search.non_recursive.pytests.ARGS import args
# from ..node import node
# from ...mcts_adapter import mcts_adapter
# from ws.RLEnvironments.self_play_games.othello.game_mgt import game_mgt

GAME_SIZE= 5

@pytest.fixture()
def setup():

    agent = agent_mgt(args, __file__)

    mcts = mcts_adapter(agent.arguments.neural_net_mgr, agent.arguments)
    fn_get_normalized_predictions = mcts.fn_get_normalized_predictions
    arguments = fn_init_arg_with_default_val(agent.arguments, 'fn_get_normalized_predictions', fn_get_normalized_predictions)

    return arguments

def fn_get_state():
    board_size = GAME_SIZE ** 2
    pieces = [None] * board_size
    for i in range(board_size):
        pieces[i] = [0] * board_size
    return pieces

def test_create_root_node(setup):
    # agent = init_agent(args, __file__)
    args = setup
    state = fn_get_state()

    root_node = node(
        state,
        args.fn_get_normalized_predictions,
        args.game_mgr.fn_get_action_size(),
        args.cpuct_exploration_exploitation_factor,

        parent_action=-1,
        val=0.0,
        parent_node=None
    )

    assert root_node is not None




