import functools

def trace(recoreder):
    def wrapper_maker(fn):
        @functools.wraps(fn)
        def fn_wrapper(*args, **kwargs):
            print('START')
            ret_value = fn(*args, **kwargs)
            # print(ret_value)
            print('END')
            return ret_value
        return fn_wrapper
    return wrapper_maker

