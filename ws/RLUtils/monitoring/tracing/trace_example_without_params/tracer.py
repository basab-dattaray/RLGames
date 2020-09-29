import functools

def tracer():
    def wrapper_maker(fn):

        @functools.wraps(fn)
        def fn_wrapper(*args, **kwargs):
            print('START: ')
            ret_value = fn(*args, **kwargs)
            print('END: ')
            return ret_value
        return fn_wrapper
    return wrapper_maker

