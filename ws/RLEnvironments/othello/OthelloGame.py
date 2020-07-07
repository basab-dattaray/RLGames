from __future__ import print_function
import sys
sys.path.append('..')
from ws.RLInterfaces.Game_old import Game
from .OthelloLogic import Board
import numpy as np

class OthelloGame(Game):
    square_content = {
        -1: "X",
        +0: "-",
        +1: "O"
    }

    @staticmethod
    def getSquarePiece(piece):
        return OthelloGame.square_content[piece]

    def __init__(self, n):
        self.n = n

    def getInitBoard(self):
        b = Board(self.n)
        return np.array(b.pieces)

    def getBoardSize(self):
        return (self.n, self.n)

    def getActionSize(self):
        return self.n*self.n

    def getNextState(self, board, player, action):
        b = Board(self.n)
        b.pieces = np.copy(board)
        move = (int(action/self.n), action%self.n)
        game_to_continue = b.execute_move(move, player)
        if not game_to_continue:
            early_completeion_score = b.countDiff(player)
            early_completion_result = 0 if early_completeion_score == 0 else (-1 if early_completeion_score < 0 else 1)
            return b.pieces, player, early_completion_result
        return (b.pieces, -player, None)

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        valids = [0]*self.getActionSize()
        b = Board(self.n)
        b.pieces = np.copy(board)
        legalMoves =  b.get_legal_moves(player)
        for x, y in legalMoves:
            valids[self.n*x+y]=1
        return np.array(valids)

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = Board(self.n)
        b.pieces = np.copy(board)

        if b.has_legal_moves(player):
            return 0
        if b.has_legal_moves(-player):
            return 0
        if b.countDiff(player) > 0:
            return 1
        return -1

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        return player*board

    def getSymmetries(self, board, pi):
        # mirror, rotational

        #! assert(len(pi) == self.n**2+1)  # 1 for pass
        #! pi_board = np.reshape(pi[:-1], (self.n, self.n))
        assert (len(pi) == self.n ** 2)  # 1 for pass
        pi_board = np.reshape(pi, (self.n, self.n))

        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                #! l += [(newB, list(newPi.ravel()) + [pi[-1]])]
                l += [(newB, list(newPi.ravel()))]
        return l

    def stringRepresentation(self, board):
        return board.tostring()

    def stringRepresentationReadable(self, board):
        board_s = "".join(self.square_content[square] for row in board for square in row)
        return board_s

    def getScore(self, board, player):
        b = Board(self.n)
        b.pieces = np.copy(board)
        return b.countDiff(player)


    @staticmethod
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
                print(OthelloGame.square_content[piece], end=" ")
            print("|")

        print("-----------------------")
