import functools

def tracer(recorder):
    def wrapper_maker(fn):
        @functools.wraps(fn)
        def fn_wrapper(*args, **kwargs):
            print('START: ' + str(recorder()))
            ret_value = fn(*args, **kwargs)
            print('END: ' + str(recorder()))
            return ret_value
        return fn_wrapper
    return wrapper_maker

