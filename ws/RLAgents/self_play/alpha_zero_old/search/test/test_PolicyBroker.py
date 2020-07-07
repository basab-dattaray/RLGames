
import pytest

from ws.RLAgents.self_play.alpha_zero_old.Services import Services
from ws.RLAgents.self_play.alpha_zero_old.othello.pytorch.NNetWrapper import NNetWrapper
from ws.RLAgents.self_play.alpha_zero_old.search.test.ConfigParams import ConfigParams

from ws.RLInterfaces.PolicyTypes import PolicyTypes

@pytest.fixture(scope='module')
def ref_support():
    config_params = ConfigParams()

    services = Services(config_params, __file__)
    neural_net = NNetWrapper(services)
    state = services.game.getInitBoard()


    yield services, neural_net, state

def test_policy_broken_basics(ref_support):

    services, neural_net, state = ref_support

    policy_broker = services.policy_broker

    fn_policy = policy_broker.fn_get_best_action_policy(PolicyTypes.POLICY_TRAINING_EVAL, neural_net) # DIRECT_MODEL
    assert fn_policy is not None

    fn_policy = policy_broker.fn_get_best_action_policy(PolicyTypes.POLICY_TRAINING_EXECUTION, neural_net) # MCTS
    assert fn_policy is not None

    fn_policy = policy_broker.fn_get_best_action_policy(PolicyTypes.POLICY_TESTING_EVAL, neural_net)
    assert fn_policy is not None

    pass

def test_policy_broken_NEW_MCTS(ref_support):

    services, neural_net, state = ref_support

    policy_broker = services.policy_broker

    fn_policy = policy_broker.fn_get_best_action_policy(PolicyTypes.POLICY_TRAINING_EXECUTION, neural_net) # MCTS
    assert fn_policy is not None


    pass


