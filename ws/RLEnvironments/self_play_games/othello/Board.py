from copy import copy

import numpy

from ws.RLEnvironments.self_play_games.othello.flip_mgt import flip_mgt


class Board():

    # list of all 8 directions on the board_pieces, as (x,y) offsets
    __directions = [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1)]

    def __init__(self, n):

        self.n = n
        # Create the empty board_pieces array.
        self.board_pieces = [None] * self.n
        for i in range(self.n):
            self.board_pieces[i] = [0] * self.n

        # Set up the initial 4 board_pieces.
        self.board_pieces[int(self.n / 2) - 1][int(self.n / 2)] = 1
        self.board_pieces[int(self.n / 2)][int(self.n / 2) - 1] = 1
        self.board_pieces[int(self.n / 2) - 1][int(self.n / 2) - 1] = -1;
        self.board_pieces[int(self.n / 2)][int(self.n / 2)] = -1;

        self.flip_mgr = flip_mgt(self.n)

    def fn_get_pieces(self):
        return self.board_pieces

    def fn_set_pieces(self, pieces):
        self.board_pieces = numpy.copy(pieces)

    def countDiff(self, color):
        """Counts the # board_pieces of the given color
        (1 for white, -1 for black, 0 for empty spaces)"""
        count = 0
        for y in range(self.n):
            for x in range(self.n):
                if self.board_pieces[x][y]==color:
                    count += 1
                if self.board_pieces[x][y]==-color:
                    count -= 1
        return count

    def get_legal_moves(self, color):

        all_allowed_moves = self.flip_mgr.fn_get_all_allowable_moves(self.board_pieces, color)

        # moves = set()  # stores the legal moves.
        #
        # # Get all the squares with board_pieces of the given color.
        # for y in range(self.n):
        #     for x in range(self.n):
        #         if self.board_pieces[x][y]==color:
        #             newmoves = self.get_moves_for_square((x,y))
        #             moves.update(newmoves)
        #
        # lst2 = list(moves)
        #
        # _is_equal = self.___compare(lst, lst2)
        # assert _is_equal is True

        return all_allowed_moves

    def ___compare(self, lst1, lst2):
        if lst1 is None:
            if lst2 is not None:
                return False

        if lst2 is None:
            if lst1 is not None:
                return False

        return copy(lst1).sort() == copy(lst2).sort()

    def has_legal_moves(self, color):
        # old =  self.old_has_legal_moves(color)
        atleast_one_legal_move_exists = self.flip_mgr.fn_any_legal_moves_exist(self.board_pieces, color)
        # _is_Equal = old == new


        return atleast_one_legal_move_exists

    # def old_has_legal_moves(self, color):
    #     for y in range(self.n):
    #         for x in range(self.n):
    #             if self.board_pieces[x][y] == color:
    #                 newmoves = self.get_moves_for_square((x, y))
    #                 if len(newmoves) > 0:
    #                     return True
    #     return False

    # def get_moves_for_square(self, square):
    #
    #     (x,y) = square
    #
    #     # determine the color of the piece.
    #     color = self.board_pieces[x][y]
    #
    #     # skip empty source squares.
    #     if color==0:
    #         return None
    #
    #     # search all possible directions.
    #     moves = []
    #     for direction in self.__directions:
    #         move = self._discover_move(square, direction)
    #         if move:
    #             # print(square,move,direction)
    #             moves.append(move)
    #
    #     return moves

    def execute_move(self, move, color):

        # flips = [flip for direction in self.__directions
        #               for flip in self._get_flips(move, direction, color)]
        #
        # flips2 = list(set(flips))

        flip_trails = self.flip_mgr.fn_get_flippables(self.board_pieces, color, move)

        # _is_equal = self.___compare(flip_trails, flips2)

        if len(list(flip_trails))==0:
            return False


        for x, y in flip_trails:
            self.board_pieces[x][y] = color
        return True

    # def _discover_move(self, origin, direction):
    #     x, y = origin
    #     color = self.board_pieces[x][y]
    #     flips = []
    #
    #     for x, y in self._increment_move(origin, direction, self.n):
    #         if self.board_pieces[x][y] == 0:
    #             if flips:
    #                 # print("Found", x,y)
    #                 return (x, y)
    #             else:
    #                 return None
    #         elif self.board_pieces[x][y] == color:
    #             return None
    #         elif self.board_pieces[x][y] == -color:
    #             # print("Flip",x,y)
    #             flips.append((x, y))

    # def _get_flips(self, origin, direction, color):
    #     old_flips = self.old_get_flips(color, direction, origin)
    #     return old_flips

    # def old_get_flips(self, color, direction, origin):
    #     flips = [origin]
    #     for x, y in self._increment_move(origin, direction, self.n):
    #
    #         if self.board_pieces[x][y] == 0:
    #             return []
    #         if self.board_pieces[x][y] == -color:
    #             flips.append((x, y))
    #         elif self.board_pieces[x][y] == color and len(flips) > 0:
    #             return flips
    #     return []

    # def _increment_move(self, move, direction, n):
    #     move = list(map(sum, zip(move, direction)))
    #     while all(map(lambda x: 0 <= x < n, move)):
    #         yield move
    #         move=list(map(sum,zip(move,direction)))


