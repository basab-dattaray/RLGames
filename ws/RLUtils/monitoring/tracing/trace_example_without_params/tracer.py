import functools

def tracer():
    def wrapper_maker(fn):

        @functools.wraps(fn)
        def fn_wrapper(*app_info, **kwargs):
            print('START: ')
            ret_value = fn(*app_info, **kwargs)
            print('END: ')
            return ret_value
        return fn_wrapper
    return wrapper_maker

