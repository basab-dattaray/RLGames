import functools

def decAny(tag):
    def dec(f0):
        @functools.wraps(f0)
        def wrapper(*args, **kwargs):
            return "<%s> %s </%s>" % (tag, f0(*args, **kwargs), tag)
        return wrapper
    return dec

def container(par1):
    @decAny( par1 )
    def test2():
        return 'test1XML'
    return test2