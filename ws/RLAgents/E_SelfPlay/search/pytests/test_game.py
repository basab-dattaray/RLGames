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









