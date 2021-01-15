import functools

def tracer(recorder):
    def function_wrapper_maker(fn):
        @functools.wraps(fn)
        def fn_wrapper(*app_info, **kwargs):
            print('START: ' + str(recorder()))
            ret_value = fn(*app_info, **kwargs)
            print('END: ' + str(recorder()))
            return ret_value
        return fn_wrapper
    return function_wrapper_maker

