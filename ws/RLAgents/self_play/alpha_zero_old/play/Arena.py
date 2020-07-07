# import logging
import gc
import numpy
from tqdm import tqdm

# from ws.RLEnvironments.othello import OthelloGame
from ws.RLEnvironments.othello.OthelloGame import OthelloGame
# log = logging.getLogger(__name__)


class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, services, game, display=OthelloGame.display):
        self.services = services
        self.game = game
        self.display = display

        self.show_board = False
        if 'debug_board' in services.nnet_params.keys():
            if services.nnet_params['debug_board'] == 1:
                self.show_board = True

        self.game_count = 0

        self.zero_runs = 0
        self.winning_runs = 0
        self.losing_runs = 0


    def playGame(self, first_player, fn_policy_player_A, fn_policy_player_B, verbose=False):

        player_policies = [fn_policy_player_B, None, fn_policy_player_A]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0

        continue_game = True
        while (self.game.getGameEnded(board, curPlayer) == 0) and continue_game:
            it += 1
            if verbose:
                assert self.display
                self.display(board)
                print("Turn ", str(it), "Player ", str(curPlayer))

            action = player_policies[curPlayer + 1](self.game.getCanonicalForm(board, curPlayer))
            if action == None:
                it += -1
                continue_game = False
            else:
                valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer), 1)
                if valids[action] == 0:
                    continue_game = False
                    it += -1
                else:
                    board, curPlayer, _ = self.game.getNextState(board, curPlayer, action)
            pass

        if verbose:
            assert self.display
            self.display(board)

        score = numpy.sum(board)
        plusses = numpy.sum([x for x in board.ravel() if x == 1])
        minusses = numpy.sum([x for x in board.ravel() if x == -1])
        zeros = numpy.sum([x for x in board.ravel() if x == 0])
        assert score == plusses + minusses

        tally = 0 if score == 0 else (1 if score > 1 else -1)

        absolute_result = tally * first_player

        last_player = curPlayer * first_player * -1

        if absolute_result == 1:
            self.winning_runs += 1
        if absolute_result == 0:
            self.zero_runs += 1
        if absolute_result == -1:
            self.losing_runs += 1

        total_runs = self.winning_runs + self.zero_runs + self.losing_runs

        plusses_trending = (self.winning_runs + self.zero_runs/2)/total_runs  * 100

        if self.show_board:
            self.game_count += 1
            self.services.fn_record(f'@@@ play game{self.game_count:4d}: result= {absolute_result:2d},  first_player= {first_player:2d}, last_player={last_player: 2d} turns= {it:2d} ,   score= {int(score):3d},      [{int(plusses):3d} / {int(minusses):3d} / {int(zeros):3d}]  ==>  plusses trending: {plusses_trending}')

        return absolute_result

    def playGames(self, fn_policy_player1, fn_policy_player2, num_of_games, verbose=False):
        num_game_pairs = int(num_of_games / 2)
        oneWon = None
        twoWon = None
        draws = None

        fn_tally_games = self.game_counter()

        for _ in range(num_game_pairs):
            gameResult = self.playGame(first_player= 1, fn_policy_player_A= fn_policy_player1, fn_policy_player_B= fn_policy_player2, verbose= verbose)
            oneWon, twoWon, draws = fn_tally_games(gameResult)

            gc.collect()

        for _ in range(num_game_pairs):
            gameResult = self.playGame(first_player= -1, fn_policy_player_A= fn_policy_player2, fn_policy_player_B= fn_policy_player1, verbose= verbose)
            oneWon, twoWon, draws = fn_tally_games(gameResult)

            gc.collect()

        return oneWon, twoWon, draws

    @staticmethod
    def game_counter():
        _oneWon = 0
        _twoWon = 0
        _draws = 0
        def fn_tally_games(new_result):
            nonlocal _oneWon, _twoWon, _draws
            if new_result == 1:
                _oneWon += 1
            elif new_result == -1:
                _twoWon += 1
            else:
                _draws += 1
            return _oneWon, _twoWon, _draws

        return fn_tally_games