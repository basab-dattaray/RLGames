import numpy

def rollout_mgt(
        fn_terminal_state_status,
        fn_get_normalized_predictions,
        multirun=False):

    def _fn_get_rollout_value(fn_terminal_state_status, state):
        ret_val = None

        terminal_state = False
        if fn_terminal_state_status is not None:
            ret_val = fn_terminal_state_status(state)
            if ret_val != 0:
                terminal_state = True
                return ret_val, None, terminal_state

        action_probabilities, state_value = fn_get_normalized_predictions(state)[:-1]

        return state_value[0], action_probabilities, terminal_state

    def fn_rollout(state):
        opponent_val, action_probs, is_terminal_state = _fn_get_rollout_value(
            fn_terminal_state_status, state
        )

        # while not is_terminal_state and multirun:
        #     normalized_valid_action_probabilities = state_cache.fn_get_valid_normalized_action_probabilities(
        #         action_probabilities=action_probs
        #     )
        #     if normalized_valid_action_probabilities is None:
        #         is_terminal_state = True
        #     else:
        #         action = numpy.random.choice(len(normalized_valid_action_probabilities),
        #                                      p=normalized_valid_action_probabilities)
        #         new_state = fn_find_next_state(state, action)
        #         if new_state is None:
        #             is_terminal_state = True
        #         else:
        #             opponent_val, action_probs, is_terminal_state =_fn_get_rollout_value(
        #                 fn_terminal_state_status, new_state
        #             )
        #
        #             state = new_state
        return -opponent_val, is_terminal_state

    return fn_rollout