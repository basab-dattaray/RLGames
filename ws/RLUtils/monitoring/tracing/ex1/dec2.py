import functools

def decAny(tag):
    def dec(f0):
        @functools.wraps(f0)
        def wrapper(*args, **kwargs):
            ret_value = f0(*args, **kwargs)
            return "%s: %s " % (tag, ret_value )
        return wrapper
    return dec

def container(par1):
    @decAny( par1 )
    def test2():
        return 'test1XML'
    return test2