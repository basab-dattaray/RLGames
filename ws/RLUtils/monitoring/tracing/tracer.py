import functools
import inspect


def tracer(args):
    # recorder = args['fn_recorder']

    recorder = args.recorder
    def function_wrapper_maker(fn):
        @functools.wraps(fn)
        def fn_wrapper(*args, **kwargs):
            recorder.fn_record_func_title_begin(fn.__qualname__)

            ret_value = fn(*args, **kwargs)
            recorder.fn_record_func_title_end()

            return ret_value
        return fn_wrapper
    return function_wrapper_maker
