import numpy
import pytest

from ws.RLAgents.self_play.alpha_zero_old.Services import Services

from ws.RLAgents.self_play.alpha_zero_old.othello.pytorch.test.ConfigParams import ConfigParams
# from ws.RLInterfaces.Game import Game
from ws.RLEnvironments.othello.OthelloGame import OthelloGame as Game, OthelloGame
from ws.RLAgents.self_play.alpha_zero_old.othello.pytorch.NNetWrapper import NNetWrapper as NNet

@pytest.fixture(scope='module')
def ref_services():
    config_params = ConfigParams()

    services = Services(config_params, __file__)

    yield services


def test_fn_get_best_predicted_action(ref_support):

    # SETUP
    services = ref_support
    max_num_action = services.args.board_size ** 2
    game = Game(services.args.board_size)
    board = game.getInitBoard()
    nnet = NNet(services)

    best_action = nnet.fn_get_best_predicted_action(board)

    assert best_action >= 0

    pass


def test_fn_get_best_action_policy(ref_support):

    # SETUP
    services = ref_support

    game = Game(services.args.board_size)
    board = game.getInitBoard()
    nnet = NNet(services)



    fn_get_best_action_policy = nnet.fn_get_best_action_policy_func(nnet.fn_predict_action)

    best_action = fn_get_best_action_policy(board)

    assert best_action >= 0

    pass
