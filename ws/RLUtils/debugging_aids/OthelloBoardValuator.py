import numpy

from ws.RLUtils.debugging_aids.ArrayValuator import ArrayValuator


class OthelloBoardValuator(ArrayValuator):

    def __init__(self):
        super().__init__(numpy.array([-1, 0, 1]))

if __name__ == '__main__':

    valuator = OthelloBoardValuator()
    array = numpy. array([1, 0, -1, 1], dtype=int)
    val = valuator.fn_get_value(array)
    assert (val == 59)