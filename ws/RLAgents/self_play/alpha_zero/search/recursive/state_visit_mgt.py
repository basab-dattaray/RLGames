from collections import namedtuple


def state_visit_mgt():
    Nsa = {}  # stores #times edge state_key,action was visited
    Ns = {}  # stores #times board_pieces state_key was visited
    fn_get_state_visits = lambda s: Ns[s] if s in Ns else 0
    fn_get_child_state_visits = lambda sa: Nsa[sa] if sa in Nsa else 0
    fn_does_child_state_visits_exist = lambda sa: sa in Nsa

    def fn_set_state_visits(state_key, visits):
        nonlocal Ns
        Ns[state_key] = visits

    def fn_incr_state_visits(state_key):
        nonlocal Ns
        if state_key not in Ns:
            Ns[state_key] = 0
        Ns[state_key] += 1

    def fn_incr_child_state_visits(state_action_key):
        nonlocal Nsa
        if state_action_key not in Nsa:
            Nsa[state_action_key] = 0
        Nsa[state_action_key] += 1

    def fn_set_child_state_visits(state_action_key, visits):
        nonlocal Nsa
        Nsa[state_action_key] = visits


    ret_functions = namedtuple('_', [
        'fn_get_state_visits',
        'fn_get_child_state_visits',
        'fn_does_child_state_visits_exist',
        'fn_set_state_visits',
        'fn_incr_state_visits',
        'fn_set_child_state_visits',
        'fn_incr_child_state_visits',
    ])
    ret_functions.fn_get_state_visits = fn_get_state_visits
    ret_functions.fn_get_child_state_visits = fn_get_child_state_visits
    ret_functions.fn_does_child_state_visits_exist = fn_does_child_state_visits_exist
    ret_functions.fn_set_state_visits = fn_set_state_visits
    ret_functions.fn_incr_state_visits = fn_incr_state_visits
    ret_functions.fn_set_child_state_visits = fn_set_child_state_visits
    ret_functions.fn_incr_child_state_visits = fn_incr_child_state_visits

    return ret_functions