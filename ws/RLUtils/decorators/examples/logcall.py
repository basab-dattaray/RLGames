from functools import wraps

def logformat(fmt):
    def logged(fn):

        @wraps(fn)
        def wrapper(*args, **kwargs):
            print(fmt.format(funct=fn))
            return fn(*args, ** kwargs)
        return wrapper
    return logged

logged_ = logformat('Youall logged {funct.__name__}')