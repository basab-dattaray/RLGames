from functools import wraps

from ws.RLUtils.monitoring.tracing.call_trace_mgt import call_trace_mgt

def trace_container():
    def trace(fn):

        @wraps(fn)
        def wrapper(*args, **kwargs):
            print('*** ')
            return fn(*args, ** kwargs)
        return wrapper
    return trace


