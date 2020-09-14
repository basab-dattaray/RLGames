import os

from pip._vendor.colorama import Fore, Back

from ws.RLUtils.monitoring.tracing.log_mgt import log_mgr

if __name__ == '__main__':
    cwd = os.path.curdir

    acwd = os.path.join(cwd, '_tests')
    log_dir = os.path.join(acwd, "logs")

    fn_record2 = log_mgr(log_dir, show_debug=True, fixed_log_file=False)
    color_black_background = Fore.YELLOW + Back.BLACK
    fn_record2('3. show_debug = True, debug=True', color=color_black_background, debug=True)
    fn_record2('4. show_debug = True, debug=False', debug=False)
