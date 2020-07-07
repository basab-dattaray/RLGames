def qwaste_mgr():
    _dict_waste = {}

    def fnDestructiveGet(key):
        nonlocal _dict_waste

        if key in _dict_waste:
            val = _dict_waste.pop(key, None)
            return val
        else:
            return None

    def fnPushIfEmpty(key, val):
        nonlocal _dict_waste

        if key in _dict_waste:
            return False
        else:
            _dict_waste[key] = val
            return True

    return fnDestructiveGet, fnPushIfEmpty
