from functools import wraps


def trace(fn):

    @wraps(fn)
    def wrapper(*args, **kwargs):
        fn2 = fn
        print('*** ' + fn2.__name__)
        return fn(*args, ** kwargs)
    return wrapper






