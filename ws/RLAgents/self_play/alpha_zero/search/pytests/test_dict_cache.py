import pytest

from ws.RLAgents.self_play.alpha_zero.search.recursive.dict_cache import dict_cache


def test_fn_get_state():
    dict = dict_cache()
    dict.fn_set_data('k1', {'y1': 1, 'y2': 2})
    val = dict.fn_get_data('k1')
    assert val == {'y1': 1, 'y2': 2}
    pass

def test_fn_get_state_2():
    dict = dict_cache()
    dict.fn_set_data('k1', {'y1': 1, 'y2': 2})

    dict.fn_set_data('k1', {'y1': 11})
    val = dict.fn_get_data('k1')
    assert val == {'y1': 11, 'y2': 2}
    pass

def test_fn_get_state_3():
    dict = dict_cache()
    dict.fn_set_data('k1', 5)
    val = dict.fn_get_data('k1')
    assert val == 5
    pass


def test_fn_does_attr_key_exist():
    dict = dict_cache()
    dict.fn_set_data('k1', {'y1': 1, 'y2': 2})
    attr_exists = dict.fn_does_attr_key_exist('k1', 'y1')
    assert attr_exists == True
    attr_exists = dict.fn_does_attr_key_exist('k1_not', 'y1')
    assert attr_exists == False
    attr_exists = dict.fn_does_attr_key_exist('k1', 'y1_not')
    assert attr_exists == False
    attr_exists = dict.fn_does_attr_key_exist('k1_not', 'y1_not')
    assert attr_exists == False
    pass