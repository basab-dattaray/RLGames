TRAINING_BASED_ON_PREDICTION = False
PLAY_IT_OUT = False


class Rollout():

    def __init__(self,
                 fn_predict_action_probablities):
        self.fn_predict_action_probablities = fn_predict_action_probablities

        self.result = None

    # def fn_set_result(self, result):
    #     self.result = result

    def fn_get_result(self):
        return self.result

    def fn_get_value(self,
                        state):
        action_probabilities, state_value = self.fn_predict_action_probablities(state)
        ret_value =  state_value[0]
        return  ret_value, action_probabilities

    def fn_get_rollout_value(self, fn_terminal_state_status, state):
        ret_val = None

        terminal_state = False
        if fn_terminal_state_status is not None:
            ret_val = fn_terminal_state_status(state)
            if ret_val != 0:
                terminal_state = True
                return ret_val, None, terminal_state

        val, action_probs = self.fn_get_value(state)

        ret_val = val
        return ret_val, action_probs, terminal_state



