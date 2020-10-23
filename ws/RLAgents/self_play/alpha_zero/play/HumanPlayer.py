
class HumanPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # fn_display(board)
        valid = self.game.fn_get_valid_moves(board, 1)
        for i in range(len(valid)):
            if valid[i]:
                print("[", int(i/self.game.fn_get_board_size()), int(i%self.game.fn_get_board_size()), end="] ")
        while True:
            input_move = input()
            input_a = input_move.split(" ")
            if len(input_a) == 2:
                try:
                    x,y = [int(i) for i in input_a]
                    if ((0 <= x) and (x < self.game.fn_get_board_size()) and (0 <= y) and (y < self.game.fn_get_board_size())) or \
                            ((x == self.game.n) and (y == 0)):
                        a = self.game.fn_get_board_size() * x + y if x != -1 else self.game.fn_get_board_size() ** 2
                        if valid[a]:
                            break
                except ValueError:
                    # Input needs to be an integer
                    'Invalid integer'
            print('Invalid move')
        return a

