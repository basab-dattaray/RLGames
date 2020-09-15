from functools import wraps

def encapsulate(func, fn_record= None):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if fn_record is None:
            print()
            print(f'<<<<<< {func.__name__} >>>>>>')
        else:
            fn_record()
            fn_record(f'<<<<<< {func.__name__} >>>>>>')
        return func(*args, **kwargs)
    return wrapper