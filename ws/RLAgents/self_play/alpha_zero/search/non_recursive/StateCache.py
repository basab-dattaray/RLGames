import numpy

class StateCache():
    def __init__(self, ref_mcts,  state):
        self.state = state
        self.ref_mcts = ref_mcts
        self.valid_norm_action_probabilities = None
        self.action_probabilities = None
        self.state_value = None


    def fn_get_valid_normalized_action_probabilities_(self, action_probabilities, distribute_if_none):
        valid_moves = self.ref_mcts.fn_get_valid_actions(self.state)
        if valid_moves is None:
            return None
        valid_action_probabilities = action_probabilities * valid_moves
        sum_action_probs = numpy.sum(valid_action_probabilities)
        if sum_action_probs > 0:
            return valid_action_probabilities / sum_action_probs
        else:
            if not distribute_if_none:
                return None
            else:
                distributed_action_probabilities = [1/len(action_probabilities)] * len(action_probabilities)
                return distributed_action_probabilities


    def fn_get_valid_normalized_action_probabilities(self, action_probabilities, distribute_if_none= False):
        if self.valid_norm_action_probabilities is None:
            if action_probabilities is None:
                action_probabilities, _ = self.fn_get_predictions()
            self.valid_norm_action_probabilities = self.fn_get_valid_normalized_action_probabilities_(action_probabilities, distribute_if_none)
        return self.valid_norm_action_probabilities

    def fn_get_predictions(self):
        if self.action_probabilities is None or self.state_value is None:
            self.action_probabilities, self.state_value = self.ref_mcts.fn_predict_action_probablities(self.state)
        return self.action_probabilities, self.state_value
