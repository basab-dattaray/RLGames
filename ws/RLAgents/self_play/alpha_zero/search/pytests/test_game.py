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
    game = game_mgt(GAME_SIZE)
    return game

def fn_get_state():
    board_size = GAME_SIZE ** 2
    pieces = [None] * board_size
    for i in range(board_size):
        pieces[i] = [0] * board_size
    return pieces



    # assert qval == 1 or qval == -1






