import sys


def inplace_print_mgt(draw_bar=True, max_num_bars=60, indent=8):

    def fn_print_in_place(bar_max_value=0, bar_actual_value=max_num_bars, text_line=''):

        prefix = ' ' * indent
        bar_line = ''
        if draw_bar:
            num_bars = int(((bar_actual_value / bar_max_value) * max_num_bars))
            bar_line = '#' * num_bars

        sys.stdout.write("\r" + prefix + text_line + ' ' + bar_line)
        sys.stdout.flush()

    def fn_print_end():
        print()

    return fn_print_in_place, fn_print_end