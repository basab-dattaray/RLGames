import numpy


def fn_rollout(
        mcts_cache_mgr, fn_get_normalized_predictions, fn_get_next_state, fn_get_canonical_form, fn_terminal_value,
        state):
    EPS = 1e-8

    def _fn_get_state_info(fn_terminal_value, state):
        qval = None
        terminal_state = False
        if fn_terminal_value is not None:
            qval = mcts_cache_mgr.fn_get_progress_status(state)  # fn_terminal_value(state)
            if qval != 0:
                terminal_state = True
                return -qval, None, terminal_state

        action_probabilities, state_value = fn_get_normalized_predictions(state)[:-1]

        return state_value[0], action_probabilities, terminal_state

    def _fn_get_best_action(state, action_probs):
        best_action = numpy.random.choice(len(action_probs), p=action_probs)

        next_state, next_player = fn_get_next_state(state, 1, best_action)
        next_state_canonical = fn_get_canonical_form(next_state, next_player)
        return next_state_canonical

    q_val, action_probs, is_terminal_state = _fn_get_state_info(
        fn_terminal_value, state
    )

    while not is_terminal_state:
        next_state = _fn_get_best_action(state, action_probs)
        q_val, action_probs, is_terminal_state = _fn_get_state_info(
            fn_terminal_value, next_state)
        state = next_state

    return q_val, is_terminal_state

