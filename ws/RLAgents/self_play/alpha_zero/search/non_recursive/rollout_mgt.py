import numpy

from ws.RLAgents.self_play.alpha_zero.search.non_recursive.Rollout import Rollout


def rollout_mgt(fn_predict_action_probablities, fn_terminal_state_status, fn_find_next_state, multirun=False):

    def fn_get_valid_normalized_action_probabilities_(self, action_probabilities):
        valid_moves = self.ref_mcts.fn_get_valid_actions(self.state)
        if valid_moves is None:
            return None
        valid_action_probabilities = action_probabilities * valid_moves
        sum_action_probs = numpy.sum(valid_action_probabilities)
        if sum_action_probs > 0:
            return valid_action_probabilities / sum_action_probs
        else:
            distributed_action_probabilities = [1/len(action_probabilities)] * len(action_probabilities)
            return distributed_action_probabilities

    def fn_get_valid_normalized_action_probabilities( action_probabilities):
        if self.valid_norm_action_probabilities is None:
            if action_probabilities is None:
                action_probabilities, _ = self.fn_get_predictions()
            self.valid_norm_action_probabilities = self.fn_get_valid_normalized_action_probabilities_(action_probabilities)
        return self.valid_norm_action_probabilities
    def fn_rollout(state ):
        # state = self.state

        rollout_impl = Rollout(fn_predict_action_probablities)

        opponent_val, action_probs, is_terminal_state = rollout_impl.fn_get_rollout_value(
            fn_terminal_state_status, state
        )

        while not is_terminal_state and multirun:
            normalized_valid_action_probabilities = self.ref_mcts.state_cache.fn_get_valid_normalized_action_probabilities(
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