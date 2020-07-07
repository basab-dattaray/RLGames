import numpy

from ws.RLAgents.self_play.alpha_zero_old.othello.pytorch.NNetWrapper import NNetWrapper
from ws.RLInterfaces.PolicyAlgos import PolicyAlgos


class PolicyBroker():
    def __init__(self, services):
        self.services = services
        self.default_policy_algo = PolicyAlgos.MCTS.name
        self.policy_dict = None
        # if 'POLICY_RULES' in self.services.args:
        #     self.policy_dict = {}
        #     for k, v in self.services.args.POLICY_RULES.items():
        #         self.policy_dict[k] = v

        pass

    def fn_get_best_action_policy(self, policy_type, neural_net, actor= None):
        if 'POLICY_RULES' in self.services.args:
            self.policy_dict = {}
            for k, v in self.services.args.POLICY_RULES.items():
                self.policy_dict[k] = v

        policy_algo = self.default_policy_algo

        if policy_type.name in self.policy_dict.keys():
            policy_algo = self.policy_dict[policy_type.name]

        fn_best_policy_func = None
        if policy_algo == PolicyAlgos.DIRECT_MODEL.name:
            fn_best_policy_func = neural_net.fn_get_best_action_policy_func(neural_net.fn_predict_action)
            return fn_best_policy_func

        if policy_algo == PolicyAlgos.MCTS.name:
            mcts = self.services.mcts_framework_cls(neural_net, is_new_mcts=True)
            fn_best_policy_func = lambda state: numpy.argmax(mcts.getActionProb(state))
            return fn_best_policy_func

        if policy_algo == PolicyAlgos.MCTS_RECURSIVE.name:
            mcts = self.services.mcts_framework_cls(neural_net, is_new_mcts=False)
            fn_best_policy_func = lambda state : numpy.argmax(mcts.getActionProb(state))
            return fn_best_policy_func

        return fn_best_policy_func