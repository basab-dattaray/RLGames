import numpy


class ArrayValuator():

    def __init__(self, ref_array):
        self.ref_array = ref_array.flatten()
        self.val_dict = {}
        self.new_val_counter = 0
        self._fn_create_val_map()

    def _fn_create_val_map(self):
        for i in range(0, len(self.ref_array)):
            key = self.ref_array[i]
            if key not in self.val_dict.keys():
                self.val_dict[key] = self.new_val_counter
                self.new_val_counter += 1

    def fn_get_value(self, target_array):
        flattened_array = target_array.flatten()
        sum = 0
        for i in range(0, len(flattened_array)):
            key = flattened_array[i]
            val = self.val_dict[key]
            sum += val * self.new_val_counter ** i
            pass
        return sum

if __name__ == '__main__':

    valuator = ArrayValuator(numpy.array([-1, 0, 1]))
    # array = numpy. array([[-1, 0, -1, 1], [ 0, 1, -1, -1]], dtype=int)
    array = numpy. array([1, 0, -1, 1], dtype=int)
    val = valuator.fn_get_value(array)
    assert (val == 59)