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

    def count_diff(self, pieces, color):
        board_size = len(pieces[0])
        count = 0
        for y in range(board_size):
            for x in range(board_size):
                if pieces[x][y]==color:
                    count += 1
                if pieces[x][y]==-color:
                    count -= 1
        return count

    def get_legal_moves(self, pieces, color):

        all_allowed_moves = self.flip_mgr.fn_get_all_allowable_moves(pieces, color)

        return all_allowed_moves

    def has_legal_moves(self, pieces, color):
        atleast_one_legal_move_exists = self.flip_mgr.fn_any_legal_moves_exist(pieces, color)
        return atleast_one_legal_move_exists


    def execute_move(self, pieces, move, color):
        copied_pieces = copy(pieces)
        flip_trails = self.flip_mgr.fn_get_flippables(copied_pieces, color, move)

        if len(list(flip_trails))==0:
            return False, pieces

        for x, y in flip_trails:
            copied_pieces[x][y] = color
        return True, copied_pieces



