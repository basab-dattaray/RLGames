import copy
from collections import namedtuple

import numpy

from ws.RLEnvironments.self_play_games.othello.flip_mgt import flip_mgt


def board_mgt(BOARD_SIZE):

    BOARD_SIZE = BOARD_SIZE
    # Create the empty board_pieces array.
    # board_pieces =  fn_init_board()

    flip_mgr = flip_mgt(BOARD_SIZE)

    def fn_init_board():
        pieces = [None] * BOARD_SIZE
        for i in range(BOARD_SIZE):
            pieces[i] = [0] * BOARD_SIZE
        # Set up the initial 4 board_pieces.
        pieces[int(BOARD_SIZE / 2) - 1][int(BOARD_SIZE / 2)] = 1
        pieces[int(BOARD_SIZE / 2)][int(BOARD_SIZE / 2) - 1] = 1
        pieces[int(BOARD_SIZE / 2) - 1][int(BOARD_SIZE / 2) - 1] = -1;
        pieces[int(BOARD_SIZE / 2)][int(BOARD_SIZE / 2)] = -1;
        return pieces

    def fn_get_advantage_count(pieces, color):
        BOARD_SIZE = len(pieces[0])
        count = 0
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if pieces[x][y]==color:
                    count += 1
                if pieces[x][y]==-color:
                    count -= 1
        return count

    def fn_find_legal_moves(pieces, color):

        all_allowed_moves = flip_mgr.fn_get_all_allowable_moves(pieces, color)

        return all_allowed_moves

    def fn_are_any_legal_moves_available(pieces, color):
        atleast_one_legal_move_exists = flip_mgr.fn_any_legal_moves_exist(pieces, color)
        return atleast_one_legal_move_exists


    def fn_execute_flips(pieces, move, color):
        copied_pieces = copy.deepcopy(pieces)
        flip_trails = flip_mgr.fn_get_flippables(copied_pieces, color, move)

        if flip_trails is None or len(list(flip_trails))==0:
            return False, pieces

        for x, y in flip_trails:
            copied_pieces[x][y] = color
        return True, copied_pieces

    board_mgr = namedtuple('_', [
        'fn_init_board',
        'fn_get_advantage_count',
        'fn_find_legal_moves',
        'fn_are_any_legal_moves_available',
        'fn_execute_flips',
        ]
    )

    board_mgr.fn_init_board = fn_init_board
    board_mgr.fn_get_advantage_count = fn_get_advantage_count
    board_mgr.fn_find_legal_moves = fn_find_legal_moves
    board_mgr.fn_are_any_legal_moves_available = fn_are_any_legal_moves_available
    board_mgr.fn_execute_flips = fn_execute_flips

    return board_mgr

