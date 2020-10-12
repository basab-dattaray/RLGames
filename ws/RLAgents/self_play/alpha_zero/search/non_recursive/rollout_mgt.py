import numpy

def rollout_mgt(state_cache, fn_predict_action_probablities, fn_terminal_state_status, fn_find_next_state, multirun=False):

    def __fn_get_value(state):
        action_probabilities, state_value = fn_predict_action_probablities(state)
        ret_value =  state_value[0]
        return  ret_value, action_probabilities

    def __fn_get_rollout_value(fn_terminal_state_status, state):
        ret_val = None

        terminal_state = False
        if fn_terminal_state_status is not None:
            ret_val = fn_terminal_state_status(state)
            if ret_val != 0:
                terminal_state = True
                return ret_val, None, terminal_state

        val, action_probs = __fn_get_value(state)

        ret_val = val
        return ret_val, action_probs, terminal_state

    def fn_rollout(state):
        opponent_val, action_probs, is_terminal_state = __fn_get_rollout_value(
            fn_terminal_state_status, state
        )

        while not is_terminal_state and multirun:
            normalized_valid_action_probabilities = state_cache.fn_get_valid_normalized_action_probabilities(
                action_probabilities=action_probs
            )
            if normalized_valid_action_probabilities is None:
                is_terminal_state = True
            else:
                action = numpy.random.choice(len(normalized_valid_action_probabilities),
                                             p=normalized_valid_action_probabilities)
                new_state = fn_find_next_state(state, action)
                if new_state is None:
                    is_terminal_state = True
                else:
                    opponent_val, action_probs, is_terminal_state =__fn_get_rollout_value(
                        fn_terminal_state_status, new_state
                    )

                    state = new_state
        val = -opponent_val
        return val, is_terminal_state

    return fn_rollout