import sys

from pip._vendor.colorama import Fore


def progress_count_mgt(title, size):
    count = 1

    def count_episode():
        nonlocal count
        msg = '{}::: {} of {}'.format(title, count, size)
        count += 1
        # sys.stdout.write(Fore.BLUE + msg)
        # sys.stdout.flush()
        print(Fore.BLUE + msg, end='\r', flush= True)
        print(flush=True)

    def end_couunt():
        print(Fore.BLACK)


    return count_episode, end_couunt