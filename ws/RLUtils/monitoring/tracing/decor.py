from functools import wraps

from ws.RLUtils.monitoring.tracing.Recorder import Recorder


def trace(fn):

    @wraps(fn)
    def wrapper(*args, **kwargs):
        print('*** ')
        return fn(*args, ** kwargs)
    return wrapper

