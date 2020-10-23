from __future__ import print_function
import sys
from collections import namedtuple

sys.path.append('..')

from .Board import Board
import numpy as np

def game_mgt(n):
    square_content = {
        -1: "X",
        +0: "-",
        +1: "O"
    }

    # @staticmethod
    # def getSquarePiece(self, piece):
    #     return game_mgt.square_content[piece]

    # def __init__(self, n):
    #     self.n = n

    def fn_get_init_board():
        # return initial board (numpy board)
        b = Board(n)
        return np.array(b.pieces)

    def fn_get_board_size():
        # (a,b) tuple
        return n

    def fn_get_action_size():
        # return number of actions
        return n*n + 1

    def fn_get_next_state(board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        if action == n*n:
            return (board, -player)
        b = Board(n)
        b.pieces = np.copy(board)
        move = (int(action/n), action%n)
        if not b.execute_move(move, player):
            return (b.pieces, None)
        return (b.pieces, -player)

    def fn_get_valid_moves(board, player):
        # return a fixed size binary vector
        valids = [0]*fn_get_action_size()
        b = Board(n)
        b.pieces = np.copy(board)
        legalMoves =  b.get_legal_moves(player)
        if len(legalMoves)==0:
            valids[-1]=1
            return np.array(valids)
        for x, y in legalMoves:
            valids[n*x+y]=1
        return np.array(valids)

    def fn_get_game_progress_status(board, player):
        if player is None:
            return fn_game_status(board)

        b = Board(n)
        b.pieces = np.copy(board)
        if b.has_legal_moves(player):
            return 0
        if b.has_legal_moves(-player):
            return 0
        if b.countDiff(player) > 0:
            return 1
        return -1

    def fn_game_status(board):
        val = sum(board.flatten())
        status = 0 if val == 0 else -1 if val < 0 else 1
        return status

    def fn_get_canonical_form(board, player):
        # return state if player==1, else return -state if player==-1
        return player*board

    def fn_get_symetric_samples(board, action_probs):
        # mirror, rotational
        assert(len(action_probs) == n ** 2 + 1)  # 1 for pass
        pi_board = np.reshape(action_probs[:-1], (n, n))
        list_of_symetries = []

        for i in range(1, 5):
                rotated_board = np.rot90(board, i)
                rotated_actions_rel_to_board = np.rot90(pi_board, i)
                list_of_symetries += [(rotated_board, list(rotated_actions_rel_to_board.ravel()) + [action_probs[-1]])]

                rotated_board_flipped = np.fliplr(rotated_board)
                rotated_actions_rel_to_board_flipped = np.fliplr(rotated_actions_rel_to_board)
                list_of_symetries += [(rotated_board_flipped, list(rotated_actions_rel_to_board_flipped.ravel()) + [action_probs[-1]])]
        return list_of_symetries

    def fn_get_string_representation(board):
        return board.tostring()

    def fn_get_string_representationReadable(board):
        board_s = "".join(square_content[square] for row in board for square in row)
        return board_s

    def fn_get_score(board, player):
        b = Board(n)
        b.pieces = np.copy(board)
        return b.countDiff(player)

    # @staticmethod
    def fn_fn_display(board):
        n = board.shape[0]
        print("   ", end="")
        for y in range(n):
            print(y, end=" ")
        print("")
        print("-----------------------")
        for y in range(n):
            print(y, "|", end="")    # print the row #
            for x in range(n):
                piece = board[y][x]    # get the piece to print
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
    ret_refs.fn_display = fn_fn_display

    return ret_refs