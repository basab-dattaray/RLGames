from enum import Enum


def fn_scaffold_init_pieces(size):
    # Create the empty board_pieces array.

    pieces = [None] * size
    for i in range(size):
        pieces[i] = [0] * size

    # Set up the initial 4 board_pieces.
    pieces[int(size / 2) - 1][int(size / 2)] = 1
    pieces[int(size / 2)][int(size / 2) - 1] = 1
    pieces[int(size / 2) - 1][int(size / 2) - 1] = -1
    pieces[int(size / 2)][int(size / 2)] = -1
    return pieces

def fn_scaffold_display_board(pieceS, sizE):
    for i in range(0, sizE):
        print(pieceS[i])

def flip_mgt(pieceS, sizE):
    def fn_find_flippables(origin_positioN):

        directionS = [(1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)]
        # directionS = [(-1, 0)]
        origin_color = pieceS[origin_positioN[0]][origin_positioN[1]]

        def fn_find_directional_flips(direction_):

            def fn_find_flip(pos_, first_time= True, opponent_is_engulfed= False):

                def fn_get_next_valid_position(pos__):
                    def fn_pos_valid(pos___):
                        return ((pos___[0] >= 0) and (pos___[0] < sizE) or (pos___[1] >= 0) and (pos___[1] < sizE))

                    def fn_get_next_position(pos___):
                        new_pos = pos___[0] + direction_[0], pos___[1] + direction_[1]
                        return new_pos

                    next_pos = fn_get_next_position(pos__)
                    if not fn_pos_valid(next_pos):
                        return None
                    return next_pos
                # end def fn_get_next_valid_position

                if pos_ is None:
                    return None
                travel_color = pieceS[pos_[0]][pos_[1]]
                if first_time: # Step 1
                    next_pos = fn_get_next_valid_position(pos_)
                    return fn_find_flip(next_pos, first_time= False)
                else:
                    if not opponent_is_engulfed: # Step 2
                        if travel_color == -origin_color:
                            next_pos = fn_get_next_valid_position(pos_)
                            return fn_find_flip(next_pos, first_time, opponent_is_engulfed= True)
                        if travel_color == origin_color:
                            return None # No hope!
                        if travel_color == 0:
                            return None # No Hope
                    else:
                        if travel_color == -origin_color:
                            next_pos = fn_get_next_valid_position(pos_)
                            return fn_find_flip(next_pos, first_time, opponent_is_engulfed)
                        if travel_color == origin_color:
                            return None # Already taken, sorry no hope
                        if travel_color == 0:
                            return pos_

            # end def fn_find_flip
            return fn_find_flip(origin_positioN)

        # end def fn_find_directional_flips

        # flips = [flip for direction in directionS
        #     for flip in fn_find_directional_flips(direction) if flip is not None]
        flips = []
        for direction in directionS:
            flip = fn_find_directional_flips(direction)
            if flip is not None:
                flips.append(flip)

        return flips
    return fn_find_flippables


if __name__ == '__main__':
    size = 5
    pieces = fn_scaffold_init_pieces(5)
    fn_find_flippables = flip_mgt(pieces, size)

    fn_scaffold_display_board(pieces, size)

    #--------------------1
    origin = (1, 2)

    flips = fn_find_flippables(
        origin
    )
    assert flips == [(3, 2), (1, 0)]

    print()
    print('origin: {}, flips: {}'.format(origin, flips))

    #--------------------2
    origin = (2, 1)
    flips = fn_find_flippables(
        origin
    )
    assert flips == [(0, 1), (2, 3)]

    print()
    print('origin: {}, flips: {}'.format(origin, flips))





