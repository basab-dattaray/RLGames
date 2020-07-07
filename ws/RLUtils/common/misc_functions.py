import random


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
