
# from .logcall import logformat, logged


# logged = logformat('CALLING) {func.__name__}')
from ws.RLUtils.decorators.examples.logcall import logged_, logformat

# logged = logformat('YOU ARE CALLING {fn1.__name__} {d}')

@logged_
def add(x,y):
    return x + y


if __name__ == '__main__':
    print(add(4,2))
