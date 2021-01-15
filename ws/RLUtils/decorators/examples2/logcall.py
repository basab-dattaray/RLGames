from functools import wraps


def trace(fn):

    @wraps(fn)
    def wrapper(*app_info, **kwargs):
        fn2 = fn
        print('*** ' + fn2.__name__)
        return fn(*app_info, ** kwargs)
    return wrapper






