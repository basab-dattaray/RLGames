from collections import namedtuple
from enum import Enum

import numpy as np



def flip_mgt(board_size):
    directionS = [(1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)]
    # board_size = len(pieceS[0])

    def _fn_get_next_valid_position(direction, pos):
        def fn_pos_valid(pos___):
            return ((pos___[0] >= 0) and (pos___[0] < board_size) and (pos___[1] >= 0) and (pos___[1] < board_size))

        def fn_get_next_position(pos___):
            new_pos = pos___[0] + direction[0], pos___[1] + direction[1]
            return new_pos

        next_pos = fn_get_next_position(pos)
        if not fn_pos_valid(next_pos):
            return None
        return next_pos

    # end def _fn_get_next_valid_position


    def fn_get_all_allowable_moves(pieceS, color):
        def fn_find_moves_given_position(origin_positioN):

            # directionS = [(-1, 0)]
            origin_color = pieceS[origin_positioN[0]][origin_positioN[1]]

            def fn_find_directional_open_positions(direction_):

                def fn_find_open_position(pos_, first_time=True, opponent_is_engulfed=False):

                    if pos_ is None:
                        return None
                    travel_color = pieceS[pos_[0]][pos_[1]]
                    if first_time:  # Step 1
                        next_pos = _fn_get_next_valid_position(direction_, pos_)
                        return fn_find_open_position(next_pos, first_time=False)
                    else:
                        if not opponent_is_engulfed:  # Step 2
                            if travel_color == -origin_color:
                                next_pos = _fn_get_next_valid_position(direction_, pos_)
                                return fn_find_open_position(next_pos, first_time, opponent_is_engulfed=True)
                            if travel_color == origin_color:
                                return None  # No hope!
                            if travel_color == 0:
                                return None  # No Hope
                        else:
                            if travel_color == -origin_color:
                                next_pos = _fn_get_next_valid_position(direction_, pos_)
                                return fn_find_open_position(next_pos, first_time, opponent_is_engulfed)
                            if travel_color == origin_color:
                                return None  # Already taken, sorry no hope
                            if travel_color == 0:
                                return pos_

                # end def fn_find_flip
                return fn_find_open_position(origin_positioN)

            # end def fn_find_directional_flips

            # flips = [flip for direction in directionS
            #     for flip in fn_find_directional_flips(direction) if flip is not None]
            flips = []
            for direction in directionS:
                flip = fn_find_directional_open_positions(direction)
                if flip is not None:
                    flips.append(flip)

            return flips

        # end def fn_find_moves_given_position

        def fn_get_list_of_starting_coordinates():
            coord_list = []
            for row in range(board_size):
                for col in range(board_size):
                    if pieceS[row][col] == color:
                        coord_list.append((row, col))

            return coord_list

        lst_starting_coords = fn_get_list_of_starting_coordinates()

        all_flips = []
        for coords in lst_starting_coords:
            flippables = fn_find_moves_given_position(coords)
            for flip in flippables:
                if flip not in all_flips:
                    all_flips.append(flip)

        return all_flips

    def fn_get_flippables(pieceS, color, target):
        def _fn_flip_em(direction_, pos, first_time= True, seen_adjacent_same_color= False):
            if pos is None:
                return None


            next_pos = _fn_get_next_valid_position(direction_, pos)
            if first_time:
                 flips = _fn_flip_em(direction_, next_pos, first_time=False, seen_adjacent_same_color=False)
                 if flips is not None:
                     flips.append(pos)
                 return flips
            else:
                travel_color = pieceS [pos[0]] [pos[1]]

                if travel_color == - color:
                    # flip_list.append(pos)
                    flips = _fn_flip_em(direction_, next_pos, first_time=False, seen_adjacent_same_color= True)
                    if flips is not None:
                        flips.append(pos)
                    return flips
                elif travel_color == color:
                    if seen_adjacent_same_color:
                        return []
                    else:
                        return None
                else:
                    return None
        flip_trails = None
        for direction in directionS:
            flip_trails = _fn_flip_em(direction, target)
            if flip_trails is not None:
                break

        return flip_trails


    fn_any_legal_moves_exist = lambda pieces, color: True if len(fn_get_all_allowable_moves(pieces, color)) > 0 else False

    ret_refs = namedtuple('_', [
        'fn_get_all_allowable_moves',
        'fn_any_legal_moves_exist',
        'fn_get_flippables'
        ]
    )

    ret_refs.fn_get_all_allowable_moves = fn_get_all_allowable_moves
    ret_refs.fn_any_legal_moves_exist = fn_any_legal_moves_exist
    ret_refs.fn_get_flippables = fn_get_flippables

    return ret_refs


def fn_scaffold_create_flip_trails_testing_pieces_flip_2():
    global pieces
    pieces = [None] * board_size
    for i in range(board_size):
        pieces[i] = [0] * board_size
    pieces[0][0] = pieces[1][0] = pieces[2][0] = pieces[1][1] = -1
    pieces[1][2] = pieces[2][1] = pieces[2][2] = pieces[2][3] = pieces[3][1] = 1

    return pieces

def fn_scaffold_create_flip_trails_testing_pieces_flip_4():
    global pieces
    pieces = [None] * board_size
    for i in range(board_size):
        pieces[i] = [0] * board_size
    pieces[0][0]  = -1
    pieces[1][1] = pieces[2][2] = pieces[3][3]  = 1

    return np.array(pieces)


def fn_scaffold_init_pieces(board_size):
    # Create the empty board_pieces array.

    pieces = [None] * board_size
    for i in range(board_size):
        pieces[i] = [0] * board_size

    # Set up the initial 4 board_pieces.
    pieces[int(board_size / 2) - 1][int(board_size / 2)] = 1
    pieces[int(board_size / 2)][int(board_size / 2) - 1] = 1
    pieces[int(board_size / 2) - 1][int(board_size / 2) - 1] = -1
    pieces[int(board_size / 2)][int(board_size / 2)] = -1
    return np.array(pieces)

def fn_scaffold_display_board(pieceS, board_size):
    for i in range(0, board_size):
        print(pieceS[i])
if __name__ == '__main__':
    board_size = 5
    pieces = fn_scaffold_init_pieces(5)
    flip_mgr = flip_mgt(board_size)

    fn_scaffold_display_board(pieces, board_size)

    #--------------------1
    # origin = (1, 2)
    #
    # flips = flip_mgr.fn_find_moves_given_position(
    #     pieces,
    #     origin
    # )
    # assert flips == [(3, 2), (1, 0)]
    #
    # print()
    # print('origin: {}, flips: {}'.format(origin, flips))

    #--------------------2
    # origin = (2, 1)
    # flips = flip_mgr.fn_find_moves_given_position(
    #     pieces,
    #     origin
    # )
    # assert flips == [(0, 1), (2, 3)]
    #
    # print()
    # print('origin: {}, flips: {}'.format(origin, flips))

    allowable_moves = flip_mgr.fn_get_all_allowable_moves(pieces, color= 1)
    assert allowable_moves == [(3, 2), (1, 0), (0, 1), (2, 3)]
    print()

    print('allowable_moves: {}'.format(allowable_moves))

    moves_exist = flip_mgr.fn_any_legal_moves_exist(pieces, color= 1)
    assert moves_exist == True
    print('moves_exist: {}'.format(moves_exist))

    target = (1,0)
    # flip_trails = flip_mgr.fn_get_flippables(1, target)
    # print('flip_trails: {}'.format(flip_trails))


###################
    pieces = fn_scaffold_create_flip_trails_testing_pieces_flip_2()

    fn_scaffold_display_board(pieces, board_size)
    target = (4,2)
    color = -1
    flip_trails = flip_mgr.fn_get_flippables(pieces, color, target)
    assert flip_trails == [(3, 1), (4, 2)]
    print()
    print('flip_trails: {}'.format(flip_trails))


###################
    pieces = fn_scaffold_create_flip_trails_testing_pieces_flip_4()

    fn_scaffold_display_board(pieces, board_size)
    target = (4,4)
    color = -1
    flip_trails = flip_mgt(board_size).fn_get_flippables(pieces, color, target)
    assert flip_trails == [(1, 1), (2, 2), (3, 3), (4, 4)]
    print()
    print('flip_trails: {}'.format(flip_trails))



