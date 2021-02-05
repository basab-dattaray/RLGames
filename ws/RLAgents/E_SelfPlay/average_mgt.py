from collections import namedtuple


def average_mgt():

    _sum = 0
    _count = 0

    def fn_update(val, n=1):
        nonlocal _count, _sum
        val = val
        _sum += val * n
        _count += n
        average = _sum / _count
        return average

    ret_obj = namedtuple('_', [
        'fn_update',
    ])

    ret_obj.fn_update = fn_update

    return ret_obj
