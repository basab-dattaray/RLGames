import numpy

class StateCache():
    def __init__(self, fn_get_valid_actions, fn_predict_action_probablities, state):
        self.state = state
        # self.ref_mcts = ref_mcts
        self.fn_get_valid_actions = fn_get_valid_actions
        self.fn_predict_action_probablities = fn_predict_action_probablities

        self.valid_norm_action_probabilities = None
        self.action_probabilities = None
        self.value = None

    def __fn_get_valid_normalized_action_probabilities_impl(self, action_probabilities):
        valid_moves = self.fn_get_valid_actions(self.state)
        if valid_moves is None:
            return None
        valid_action_probabilities = action_probabilities * valid_moves
        sum_action_probs = numpy.sum(valid_action_probabilities)
        if sum_action_probs > 0:
            return valid_action_probabilities / sum_action_probs
        else:
            distributed_action_probabilities = [1/len(action_probabilities)] * len(action_probabilities)
            return distributed_action_probabilities


    def fn_get_valid_normalized_action_probabilities(self, action_probabilities):
        if self.valid_norm_action_probabilities is None:
            if action_probabilities is None:
                action_probabilities, _ = self.fn_get_predictions()
            self.valid_norm_action_probabilities = self.__fn_get_valid_normalized_action_probabilities_impl(action_probabilities)
        return self.valid_norm_action_probabilities

    def fn_get_predictions(self):
        if self.action_probabilities is None or self.value is None:
            self.action_probabilities, self.value = self.fn_predict_action_probablities(self.state)
        return self.action_probabilities, self.value
