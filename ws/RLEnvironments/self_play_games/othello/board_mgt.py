from collections import namedtuple

import numpy


def board_mgt(n):

    # list of all 8 directions on the pieces, as (x,y) offsets
    __directions = [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1)]

    pieces = [None]*n
    for i in range(n):
        pieces[i] = [0]*n

    # Set up the initial 4 pieces.
    pieces[int(n/2)-1][int(n/2)] = 1
    pieces[int(n/2)][int(n/2)-1] = 1
    pieces[int(n/2)-1][int(n/2)-1] = -1;
    pieces[int(n/2)][int(n/2)] = -1;

    def fn_get_pieces():
        return pieces

    def fn_set_pieces(pieces):
        pieces = numpy.copy(pieces)

    def countDiff(color):
        """Counts the # pieces of the given color
        (1 for white, -1 for black, 0 for empty spaces)"""
        count = 0
        for y in range(n):
            for x in range(n):
                if pieces[x][y]==color:
                    count += 1
                if pieces[x][y]==-color:
                    count -= 1
        return count

    def get_legal_moves(color):

        moves = set()  # stores the legal moves.

        # Get all the squares with pieces of the given color.
        for y in range(n):
            for x in range(n):
                if pieces[x][y]==color:
                    newmoves = get_moves_for_square((x,y))
                    moves.update(newmoves)
        return list(moves)

    def has_legal_moves(color):
        list = get_legal_moves(color)
        if list is None:
            return False
        return True if len(list) > 0 else False
        # for y in range(n):
        #     for x in range(n):
        #         if pieces[x][y]==color:
        #             newmoves = get_moves_for_square((x,y))
        #             if len(newmoves)>0:
        #                 return True
        # return False

    def get_moves_for_square(square):

        (x,y) = square

        # determine the color of the piece.
        color = pieces[x][y]

        # skip empty source squares.
        if color==0:
            return None

        # search all possible directions.
        moves = []
        for direction in __directions:
            move = _discover_move(square, direction)
            if move:
                # print(square,move,direction)
                moves.append(move)

        return moves

    def execute_move(move, color):

        flips = [flip for direction in __directions
                      for flip in _get_flips(move, direction, color)]

        if len(list(flips))==0:
            return False
        for x, y in flips:
            pieces[x][y] = color
        return True

    def _discover_move(origin, direction):
        x, y = origin
        color = pieces[x][y]
        flips = []

        for x, y in _increment_move(origin, direction, n):
            if pieces[x][y] == 0:
                if flips:
                    # print("Found", x,y)
                    return (x, y)
                else:
                    return None
            elif pieces[x][y] == color:
                return None
            elif pieces[x][y] == -color:
                # print("Flip",x,y)
                flips.append((x, y))

    def _get_flips(origin, direction, color):
        flips = [origin]

        for x, y in _increment_move(origin, direction, n):

            if pieces[x][y] == 0:
                return []
            if pieces[x][y] == -color:
                flips.append((x, y))
            elif pieces[x][y] == color and len(flips) > 0:
                return flips

        return []

    def _increment_move(move, direction, n):
        move = list(map(sum, zip(move, direction)))
        while all(map(lambda x: 0 <= x < n, move)):
            yield move
            move=list(map(sum,zip(move,direction)))


    ret_refs = namedtuple('_', [
        'fn_get_pieces',
        'fn_set_pieces',
        'countDiff',
        'get_legal_moves',

        'has_legal_moves',
        'get_moves_for_square',
        'execute_move'
        ]
    )

    ret_refs.fn_get_pieces = fn_get_pieces
    ret_refs.fn_set_pieces = fn_set_pieces
    ret_refs.countDiff = countDiff
    ret_refs.get_legal_moves = get_legal_moves

    ret_refs.has_legal_moves = has_legal_moves
    ret_refs.get_moves_for_square = get_moves_for_square
    ret_refs.execute_move = execute_move

    return ret_refs