import functools

def tracer(args):
    recorder = args['fn_loger']
    def function_wrapper_maker(fn):
        @functools.wraps(fn)
        def fn_wrapper(*args, **kwargs):
            print('START: ' + str(recorder()))
            ret_value = fn(*args, **kwargs)
            print('END: ' + str(recorder()))
            print()
            return ret_value
        return fn_wrapper
    return function_wrapper_maker

