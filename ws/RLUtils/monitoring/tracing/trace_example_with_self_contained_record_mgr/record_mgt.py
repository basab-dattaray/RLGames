def record_mgt():
    count = 0
    def fn_loger():
        nonlocal count
        count += 1
        return count
    return fn_loger