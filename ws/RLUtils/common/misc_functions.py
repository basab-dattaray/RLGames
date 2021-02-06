import math
import random
from time import time


def calc_pixels(unit, x, y):
    return int((x + 0.5) * unit), int((y + 0.5) * unit)


def arg_max(next_state):
    max_index_list = []
    max_value = next_state[0]
    for index, value in enumerate(next_state):
        if value > max_value:
            max_index_list.clear()
            max_value = value
            max_index_list.append(index)
        elif value == max_value:
            max_index_list.append(index)
    return random.choice(max_index_list)


# def fn_get_elapsed_time(start_time):
#     end_time = time()
#     time_diff = int(end_time - start_time)
#     mins = math.floor(time_diff / 60)
#     secs = time_diff % 60
#     time_stats = f'Time elapsed:    minutes: {mins}    seconds: {secs}  -----  (start_time:{start_time}, end_time:{end_time})'
#     # if fn_record is not None:
#     #     fn_record(f'start_time:{start_time}')
#     #     fn_record(f'end_time:{end_time}')
#     #     fn_record(f'Time elapsed:    minutes: {mins}    seconds: {secs}')
#
#     return time_stats
