
# from .logcall import logformat, logged


# logged = logformat('CALLING) {func.__name__}')
# from ws.RLUtils.decorators.examples.logcall import logged_, logformat

# logged = logformat(fn111)
# from ws.RLUtils.decorators.examples2.logcall import logged_
from ws.RLUtils.decorators.examples2.logcall import trace


@trace
def add(x,y):
    return x + y


if __name__ == '__main__':
    print(add(4,2))
