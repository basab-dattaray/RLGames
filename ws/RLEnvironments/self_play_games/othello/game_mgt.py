from __future__ import print_function
import sys
from collections import namedtuple

# from .board_mgt import board_mgt

sys.path.append('..')

from .Board import Board
import numpy as np

EXISTING = True

def game_mgt(board_size):

    square_content = {
        -1: "X",
        +0: "-",
        +1: "O"
    }

    def fn_get_init_board():
        # return initial board_pieces (numpy board_pieces)
        # b = Board(board_size)
        # if not EXISTING:
        #     b = board_mgt(board_size)
        return np.array(Board(board_size).fn_init_board())

    def fn_get_board_size():
        # (a,b) tuple
        return board_size

    def fn_get_action_size():
        return board_size * board_size + 1

    def fn_get_next_state(pieces, player, action):
        if action == board_size*board_size:
            return (pieces, -player)
        b = Board(board_size)
        move = (int(action / board_size), action % board_size)
        success, pieces = b.fn_execute_flips(pieces, move, player)
        if not success:
            return (pieces, player)
        return (pieces, -player)

    def fn_get_valid_moves(pieces, player):
        valids = [0]*fn_get_action_size()
        b = Board(board_size)

        legalMoves =  b.fn_find_legal_moves(pieces, player)
        if len(legalMoves)==0:
            valids[-1]=1
            return np.array(valids)
        for x, y in legalMoves:
            valids[board_size * x + y]=1
        return np.array(valids)

    def fn_get_game_progress_status(pieces, player):
        if player is None:
            return fn_game_status(pieces)

        b = Board(board_size)

        if b.fn_are_any_legal_moves_available(pieces, player):
            return 0
        if b.fn_are_any_legal_moves_available(pieces, -player):
            return 0
        if b.fn_get_advantage_count(pieces, player) > 0:
            return 1
        return -1

    def fn_game_status(pieces):
        val = sum(pieces.flatten())
        status = 0 if val == 0 else -1 if val < 0 else 1
        return status

    def fn_get_canonical_form(pieces, player):
        canonical_pieces =  player * pieces
        return canonical_pieces

    def fn_get_symetric_samples(pieces, action_probs):
        pi_board = np.reshape(action_probs[:-1], (board_size, board_size))
        list_of_symetries = []

        for i in range(1, 5):
                rotated_board = np.rot90(pieces, i)
                rotated_actions_rel_to_board = np.rot90(pi_board, i)
                list_of_symetries += [(rotated_board, list(rotated_actions_rel_to_board.ravel()) + [action_probs[-1]])]

                rotated_board_flipped = np.fliplr(rotated_board)
                rotated_actions_rel_to_board_flipped = np.fliplr(rotated_actions_rel_to_board)
                list_of_symetries += [(rotated_board_flipped, list(rotated_actions_rel_to_board_flipped.ravel()) + [action_probs[-1]])]
        return list_of_symetries

    def fn_get_string_representation(pieces):
        return pieces.tostring()


    def fn_get_score(pieces, player):
        b = Board(board_size)
        return b.fn_get_advantage_count(pieces, player)

    def fn_display(pieces):
        n = pieces.shape[0]
        print("   ", end="")
        for y in range(n):
            print(y, end=" ")
        print("")
        print("-----------------------")
        for y in range(n):
            print(y, "|", end="")    # print the row #
            for x in range(n):
                piece = pieces[y][x]    # get the piece to print
                print(square_content[piece], end=" ")
            print("|")

        print("-----------------------")

    ret_refs = namedtuple('_', [
        'fn_get_init_board',
        'fn_get_board_size',
        'fn_get_action_size',
        'fn_get_next_state',

        'fn_get_valid_moves',
        'fn_get_game_progress_status',
        'fn_game_status',
        'fn_get_canonical_form',

        'fn_get_symetric_samples',
        'fn_get_string_representation',
        'fn_get_score' ,
        'fn_display'
        ]
    )

    ret_refs.fn_get_init_board = fn_get_init_board
    ret_refs.fn_get_board_size = fn_get_board_size
    ret_refs.fn_get_action_size = fn_get_action_size
    ret_refs.fn_get_next_state = fn_get_next_state

    ret_refs.fn_get_valid_moves = fn_get_valid_moves
    ret_refs.fn_get_game_progress_status = fn_get_game_progress_status
    ret_refs.fn_game_status = fn_game_status
    ret_refs.fn_get_canonical_form = fn_get_canonical_form

    ret_refs.fn_get_symetric_samples = fn_get_symetric_samples
    ret_refs.fn_get_string_representation = fn_get_string_representation
    ret_refs.fn_get_score = fn_get_score
    ret_refs.fn_display = fn_display

    return ret_refs