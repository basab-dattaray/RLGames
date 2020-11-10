from functools import wraps

def encapsulate(func, fn_log= None):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if fn_log is None:
            print()
            print(f'<<<<<< {func.__name__} >>>>>>')
        else:
            fn_log()
            fn_log(f'<<<<<< {func.__name__} >>>>>>')
        return func(*args, **kwargs)
    return wrapper