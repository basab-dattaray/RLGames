import sys

from pip._vendor.colorama import Fore


def progress_count_mgt(title, size):
    count = 1

    def fn_count_event():
        nonlocal count
        msg = '{}::: {} of {}'.format(title, count, size)
        count += 1
        sys.stdout.write("\r" + Fore.BLUE + msg)
        sys.stdout.flush()

    def fn_stop_counting():
        print(Fore.BLACK)


    return fn_count_event, fn_stop_counting