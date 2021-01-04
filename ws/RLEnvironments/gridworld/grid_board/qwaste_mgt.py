def qwaste_mgt():
    _dict_waste = {}

    def fn_pop(key):
        nonlocal _dict_waste

        if key in _dict_waste:
            val = _dict_waste.pop(key, None)
            return val
        else:
            return None

    def fn_push(key, val):
        nonlocal _dict_waste

        if key in _dict_waste:
            return False
        else:
            _dict_waste[key] = val
            return True

    return fn_pop, fn_push
