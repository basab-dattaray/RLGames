import numpy


def action_mgt(USE_SMART_PREDICTOR_FOR_ROLLOUT, fn_get_valid_moves, fn_get_prediction_info):
    def _fn_get_possible_actions_from_valid_moves(state):
        valid_moves = fn_get_valid_moves(state, 1)
        sum_policy = numpy.sum(valid_moves)
        normalized_valid_moves = valid_moves / sum_policy
        return normalized_valid_moves

    def _fn_get_possible_actions_from_predictions(state):
        prediction_info = fn_get_prediction_info(state, 1)
        policy = prediction_info[0]
        return policy

    def fn_generate_action_getter(fn_get_possible_actions):

        def fn_get_action_given_state(state):
            normalized_valid_moves = fn_get_possible_actions(state)

            action = numpy.random.choice(len(normalized_valid_moves), p=normalized_valid_moves)
            return action

        return fn_get_action_given_state

    fn_get_possible_actions = None
    if USE_SMART_PREDICTOR_FOR_ROLLOUT:
        fn_get_possible_actions = _fn_get_possible_actions_from_predictions
    else:
        fn_get_possible_actions = _fn_get_possible_actions_from_valid_moves
    fn_get_action_given_state = fn_generate_action_getter(fn_get_possible_actions)
    return fn_get_action_given_state
