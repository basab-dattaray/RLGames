def record_mgt():
    count = 0
    def fn_recorder():
        nonlocal count
        count += 1
        return count
    return fn_recorder