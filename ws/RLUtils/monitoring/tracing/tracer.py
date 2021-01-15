import functools
import inspect


def tracer(app_info, verboscity = 1):
    # recorder = nn_args['fn_loger']

    recorder = app_info.CALL_TRACER_
    def function_wrapper_maker(fn):
        @functools.wraps(fn)
        def fn_wrapper(*app_info, **kwargs):
            if verboscity >= 3:
                recorder.fn_enter_function(fn.__qualname__)

            ret_value = fn(*app_info, **kwargs)
            recorder.fn_leave_function()

            return ret_value
        return fn_wrapper
    return function_wrapper_maker
