from enum import Enum


def init_pieces(n):
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


def fn_find_flippables(pieceS, sizE, current_positioN, coloR):
    directionS = [(-1, -1),(1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)]
    class Match(Enum):
        found = 1
        failed = 2
        search = 3
        
    def fn_find_directional_flips(direction_):

        def fn_get_next_valid_position(pos_, direction_):
            
            def fn_match_opposite_color(pos__):
                if pieceS[pos__[0]][pos__[1]] == -coloR:
                    return Match.search
                if pieceS[pos__[0]][pos__[1]] == coloR:
                    return Match.failed
                return Match.found

            def fn_pos_valid(pos__):
                return ((pos__[0] >= 0) and (pos__[0] < sizE) or (pos__[1] >= 0) and (pos__[1] < sizE))

            def fn_seek_match(pos__):                  
                if fn_pos_valid(pos__) is False:
                    return Match.failed 
                return fn_match_opposite_color(pos__)
            
            def fn_get_next_position(p):
                return p[0] + direction_[0], p[1] + direction_[1]

            next_pos = fn_get_next_position(pos_)
            match = fn_seek_match(next_pos)

            while match is Match.search:
                next_pos = fn_get_next_position(next_pos)
                match = fn_seek_match(next_pos)

            if match is Match.failed:
                return None
            if match is Match.found:
                return next_pos

        pos = current_positioN
        flip_position = fn_get_next_valid_position(pos, direction_)
        return flip_position

    flips = [flip for direction in directionS
        for flip in fn_find_directional_flips(direction)]

    return flips

if __name__ is '__main__':
    pieces = init_pieces(5)
    flips = fn_find_flippables(
        pieces,
        5,
        (0, 0),
        1
    )



