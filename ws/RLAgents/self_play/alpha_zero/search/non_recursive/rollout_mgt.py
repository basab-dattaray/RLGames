import numpy

from ws.RLAgents.self_play.alpha_zero.search.non_recursive.Rollout import Rollout


def rollout_mgt(state_cache, fn_predict_action_probablities, fn_terminal_state_status, fn_find_next_state, multirun=False):

    def fn_rollout(state ):
        # state = self.state

        rollout_impl = Rollout(fn_predict_action_probablities)

        opponent_val, action_probs, is_terminal_state = rollout_impl.fn_get_rollout_value(
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
                    opponent_val, action_probs, is_terminal_state = rollout_impl.fn_get_rollout_value(
                        fn_terminal_state_status, new_state
                    )

                    state = new_state
        val = -opponent_val
        return val, is_terminal_state
    return fn_rollout