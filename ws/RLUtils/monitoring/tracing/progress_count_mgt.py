import sys

from pip._vendor.colorama import Fore


def progress_count_mgt(title, size):
    count = 1

    def count_episode():
        nonlocal count
        msg = '{}::: {} of {}'.format(title, count, size)
        count += 1
        sys.stdout.write("\r" + Fore.BLUE + msg)
        sys.stdout.flush()

    def end_count():
        print(Fore.BLACK)


    return count_episode, end_count