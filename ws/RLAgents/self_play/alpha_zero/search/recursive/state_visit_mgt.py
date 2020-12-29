from collections import namedtuple


def state_visit_mgt():
    Nsa = {}  # stores #times edge state_key,action was visited
    Ns = {}  # stores #times board_pieces state_key was visited
    fn_get_Ns = lambda s: Ns[s] if s in Ns else 0
    fn_get_Nsa = lambda sa: Nsa[sa] if sa in Nsa else 0
    fn_does_key_exist_in_Nsa = lambda sa: sa in Nsa

    def fn_set_Ns(state_key, visits):
        nonlocal Ns
        Ns[state_key] = visits

    def fn_incr_Ns(state_key):
        nonlocal Ns
        if state_key not in Ns:
            Ns[state_key] = 0
        Ns[state_key] += 1

    def fn_incr_Nsa(state_action_key):
        nonlocal Nsa
        if state_action_key not in Nsa:
            Nsa[state_action_key] = 0
        Nsa[state_action_key] += 1

    def fn_set_Nsa(state_action_key, visits):
        nonlocal Nsa
        Nsa[state_action_key] = visits


    ret_obj = namedtuple('_', [
        'fn_get_Ns',
        'fn_get_Nsa',
        'fn_does_key_exist_in_Nsa',
        'fn_set_Ns',
        'fn_incr_Ns',
        'fn_set_Nsa',
        'fn_incr_Nsa',
    ])
    ret_obj.fn_get_Ns = fn_get_Ns
    ret_obj.fn_get_Nsa = fn_get_Nsa
    ret_obj.fn_does_key_exist_in_Nsa = fn_does_key_exist_in_Nsa
    ret_obj.fn_set_Ns = fn_set_Ns
    ret_obj.fn_incr_Ns = fn_incr_Ns
    ret_obj.fn_set_Nsa = fn_set_Nsa
    ret_obj.fn_incr_Nsa = fn_incr_Nsa

    return ret_obj