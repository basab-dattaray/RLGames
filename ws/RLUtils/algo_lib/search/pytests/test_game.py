import pytest
#

from ws.RLEnvironments.self_play_games.othello.game_mgt import game_mgt

GAME_SIZE= 5

@pytest.fixture()
def setup():
    game = game_mgt(GAME_SIZE)
    return game

def fn_get_state():
    BOARD_SIZE = GAME_SIZE ** 2
    pieces = [None] * BOARD_SIZE
    for i in range(BOARD_SIZE):
        pieces[i] = [0] * BOARD_SIZE
    return pieces

# def test_fn_next_state_given_action():
#     game = game_mgt(GAME_SIZE)
#     state1 = game.fn_get_init_board()
#     action = 5
#     state2 = game.fn_next_state_given_action(state1, action)
#     list_state2 = state2.tolist()
#     expected_state2 = [[0, 0, 0, 0, 0], [-1, -1, -1, 0, 0], [0, -1, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
#
#     assert list_state2 == expected_state2
#
#     pass







