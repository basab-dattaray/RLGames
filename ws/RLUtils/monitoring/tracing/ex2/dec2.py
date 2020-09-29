import functools

def trace(tag):
    def dec(f0):
        @functools.wraps(f0)
        def wrapper(*args, **kwargs):
            print('START')
            ret_value = f0(*args, **kwargs)
            print(ret_value)
            print('END')
            return 'self'
        return wrapper
    return dec

def container(recorder):
    @trace(recorder)
    def test2():
        return 'test1XML'
    return test2