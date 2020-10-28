from enum import Enum


def fn_scaffold_init_pieces(n):
    # Create the empty board_pieces array.

    pieces = [None] * n
    for i in range(n):
        pieces[i] = [0] * n

    # Set up the initial 4 board_pieces.
    pieces[int(n / 2) - 1][int(n / 2)] = 1
    pieces[int(n / 2)][int(n / 2) - 1] = 1
    pieces[int(n / 2) - 1][int(n / 2) - 1] = -1
    pieces[int(n / 2)][int(n / 2)] = -1
    return pieces

def fn_scaffold_display_board(pieceS, sizE):
    for i in range(0, sizE):
        print(pieceS[i])

def fn_find_flippables(pieceS, sizE, origin_positioN):

    directionS = [(1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)]
    directionS = [(1, 0)]
    origin_color = pieceS[origin_positioN[0]][origin_positioN[1]]
        
    def fn_find_directional_flips(direction_):

        def fn_find_next_flip(pos_, direction_):
            class Match(Enum):
                found = 1
                failed = 2
                search = 3

            def fn_find_flip(pos__, opponent_is_engulfed= False):
                if pos__ is None:
                    return Match.failed
                travel_color = pieceS[pos__[0]][pos__[1]]
                if not opponent_is_engulfed:
                    next_pos = fn_get_next_valid_position(pos__)
                    return fn_find_flip(next_pos, opponent_is_engulfed= True)
                else:
                    if travel_color == -origin_color:
                        next_pos = fn_get_next_valid_position(pos__)
                        return fn_find_flip(next_pos, opponent_is_engulfed)
                    if travel_color == origin_color:
                        return Match.failed # already taken, sorry!
                    if travel_color == 0:
                        return Match.found

            def fn_pos_valid(pos__):
                return ((pos__[0] >= 0) and (pos__[0] < sizE) or (pos__[1] >= 0) and (pos__[1] < sizE))

            def fn_get_next_valid_position(pos__):
                def fn_get_next_position(p):
                    new_pos = p[0] + direction_[0], p[1] + direction_[1]
                    return new_pos

                next_pos = fn_get_next_position(pos__)
                if not fn_pos_valid(next_pos):
                    return None
                return next_pos

            next_pos = pos_
            match = fn_find_flip(next_pos)

            while match is Match.search:
                next_pos = fn_get_next_valid_position(next_pos)
                match = fn_find_flip(next_pos)

            if match is Match.failed:
                return None
            if match is Match.found:
                return next_pos

        pos = origin_positioN
        flip_position = fn_find_next_flip(pos, direction_)
        return flip_position

    # flips = [flip for direction in directionS
    #     for flip in fn_find_directional_flips(direction) if flip is not None]

    flips = []
    for direction in directionS:
        flip = fn_find_directional_flips(direction)
        if flip is not None:
            flips.append(flip)

    return flips


if __name__ == '__main__':
    size = 5
    pieces = fn_scaffold_init_pieces(5)
    flips = fn_find_flippables(
        pieces,
        size,
        (1, 2)
    )
    fn_scaffold_display_board(pieces, size)
    print()
    print('flips: {}'.format(flips))




