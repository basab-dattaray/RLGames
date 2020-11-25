import pytest

from ws.RLAgents.self_play.alpha_zero.misc.agent_mgt import agent_mgt, fn_init_arg_with_default_val
from ws.RLAgents.self_play.alpha_zero.search.mcts_adapter import mcts_adapter
from ws.RLAgents.self_play.alpha_zero.search.non_recursive.mcts_mgt import mcts_mgt
from ws.RLAgents.self_play.alpha_zero.search.non_recursive.pytests.ARGS import args
# from ..node import node
# from ...mcts_adapter import mcts_adapter
# from ws.RLEnvironments.self_play_games.othello.game_mgt import game_mgt
from ws.RLEnvironments.self_play_games.othello.board_mgt import board_mgt
from ws.RLEnvironments.self_play_games.othello.game_mgt import game_mgt

GAME_SIZE= 5

@pytest.fixture()
def setup():

    agent = agent_mgt(args, __file__)

    return agent

def fn_get_state():
    board_size = GAME_SIZE ** 2
    pieces = [None] * board_size
    for i in range(board_size):
        pieces[i] = [0] * board_size
    return pieces

def test_fn_execute_monte_carlo_tree_search(setup):
    # agent = init_agent(args, __file__)
    args = setup.arguments

    # state = args.game.fn_init_board()
    state = game_mgt(GAME_SIZE).fn_get_init_board()

    mcts = mcts_adapter(args.neural_net_mgr, args)
    # fn_get_prediction_info = mcts.fn_get_prediction_info
    # arguments = fn_init_arg_with_default_val(agent.arguments, 'mcts', mcts)

    mcts_mgr = mcts_mgt(
        mcts.fn_get_prediction_info,
        args.game_mgr.fn_get_state_key,
        args.game_mgr.fn_get_next_state,
        args.game_mgr.fn_get_canonical_form,
        mcts.fn_terminal_value,
        args.num_of_mc_simulations,
        args.cpuct_exploration_exploitation_factor,
        args.game_mgr.fn_get_action_size()
    )

    assert mcts_mgr.fn_execute_monte_carlo_tree_search is not None
    assert mcts_mgr.fn_get_policy is not None

    qval = mcts_mgr.fn_execute_monte_carlo_tree_search(state)

    assert qval == 1 or qval == -1






