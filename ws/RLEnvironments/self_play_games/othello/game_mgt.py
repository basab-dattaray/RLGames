from __future__ import print_function
import sys
from collections import namedtuple

sys.path.append('..')

from .OthelloLogic import Board
import numpy as np

def OthelloGame(n):
    square_content = {
        -1: "X",
        +0: "-",
        +1: "O"
    }

    # @staticmethod
    # def getSquarePiece(self, piece):
    #     return OthelloGame.square_content[piece]

    # def __init__(self, n):
    #     self.n = n

    def getInitBoard():
        # return initial board (numpy board)
        b = Board(n)
        return np.array(b.pieces)

    def getBoardSize():
        # (a,b) tuple
        return n

    def getActionSize():
        # return number of actions
        return n*n + 1

    def getNextState(board, player, action):
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

    def getValidMoves(board, player):
        # return a fixed size binary vector
        valids = [0]*getActionSize()
        b = Board(n)
        b.pieces = np.copy(board)
        legalMoves =  b.get_legal_moves(player)
        if len(legalMoves)==0:
            valids[-1]=1
            return np.array(valids)
        for x, y in legalMoves:
            valids[n*x+y]=1
        return np.array(valids)

    def getGameEnded(board, player):
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

    def getCanonicalForm(board, player):
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

    def stringRepresentation(board):
        return board.tostring()

    def stringRepresentationReadable(board):
        board_s = "".join(square_content[square] for row in board for square in row)
        return board_s

    def getScore(board, player):
        b = Board(n)
        b.pieces = np.copy(board)
        return b.countDiff(player)

    # @staticmethod
    def display(board):
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
        'getInitBoard',
        'getBoardSize',
        'getActionSize',
        'getNextState',

        'getValidMoves',
        'getGameEnded',
        'fn_game_status',
        'getCanonicalForm',

        'fn_get_symetric_samples',
        'stringRepresentation',
        'getScore' ,
        'display'
        ]
    )

    ret_refs.getInitBoard = getInitBoard
    ret_refs.getBoardSize = getBoardSize
    ret_refs.getActionSize = getActionSize
    ret_refs.getNextState = getNextState

    ret_refs.getValidMoves = getValidMoves
    ret_refs.getGameEnded = getGameEnded
    ret_refs.fn_game_status = fn_game_status
    ret_refs.getCanonicalForm = getCanonicalForm

    ret_refs.fn_get_symetric_samples = fn_get_symetric_samples
    ret_refs.stringRepresentation = stringRepresentation
    ret_refs.getScore = getScore
    ret_refs.display = display

    return ret_refs