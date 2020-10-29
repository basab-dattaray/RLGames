from copy import copy

import numpy

from ws.RLEnvironments.self_play_games.othello.flip_mgt import flip_mgt


class Board():

    def __init__(self, board_size):

        self.board_size = board_size
        # Create the empty board_pieces array.
        self.board_pieces =  self.fn_init_board()

        self.flip_mgr = flip_mgt(self.board_size)

    def fn_init_board(self):
        pieces = [None] * self.board_size
        for i in range(self.board_size):
            pieces[i] = [0] * self.board_size
        # Set up the initial 4 board_pieces.
        pieces[int(self.board_size / 2) - 1][int(self.board_size / 2)] = 1
        pieces[int(self.board_size / 2)][int(self.board_size / 2) - 1] = 1
        pieces[int(self.board_size / 2) - 1][int(self.board_size / 2) - 1] = -1;
        pieces[int(self.board_size / 2)][int(self.board_size / 2)] = -1;
        return pieces

    def fn_get_pieces(self):
        return self.board_pieces

    def fn_set_pieces(self, pieces):
        self.board_pieces = numpy.copy(pieces)

    def count_diff(self, color):
        count = 0
        for y in range(self.board_size):
            for x in range(self.board_size):
                if self.board_pieces[x][y]==color:
                    count += 1
                if self.board_pieces[x][y]==-color:
                    count -= 1
        return count

    def get_legal_moves(self, color):

        all_allowed_moves = self.flip_mgr.fn_get_all_allowable_moves(self.board_pieces, color)

        return all_allowed_moves

    def has_legal_moves(self, color):
        atleast_one_legal_move_exists = self.flip_mgr.fn_any_legal_moves_exist(self.board_pieces, color)
        return atleast_one_legal_move_exists


    def execute_move(self, move, color):
        flip_trails = self.flip_mgr.fn_get_flippables(self.board_pieces, color, move)

        if len(list(flip_trails))==0:
            return False

        for x, y in flip_trails:
            self.board_pieces[x][y] = color
        return True



