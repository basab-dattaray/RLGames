from functools import wraps

def logformat(fmt):
    def logged(fn):

        @wraps(fn)
        def wrapper(*app_info, **kwargs):
            print(fmt.format(funct=fn))
            return fn(*app_info, ** kwargs)
        return wrapper
    return logged

logged_ = logformat('Youall logged {funct.__name__}')