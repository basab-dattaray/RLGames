from collections import deque


class Compactor():
    def __init__(self, size, average_interval, skip_interval):
        self.q = deque(maxlen=size)
        self.average_interval = average_interval
        self.skip_interval = skip_interval

    # def fn_add(self, *args, **kwargs):
    #     pass

    def fn_gen_skip_filter(self, *args, **kwargs):
        new_stuff = None
        i = 0
        mid_index = int(self.skip_interval/2)
        while i < self.skip_interval:
            if (i % self.skip_interval) == mid_index:
                new_index = i
                if kwargs['index'] is not None:
                    new_index = kwargs['index']
                new_stuff = {'index': new_index, 'y_vals': kwargs['y_vals']}
            i += 1
        yield new_stuff  # new_index, kwargs['y_vals']