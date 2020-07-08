from functools import wraps

def encapsulate(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print()
        print(f'<<<<<< {func.__name__} >>>>>>')
        return func(*args, **kwargs)

    return wrapper