import os

from pip._vendor.colorama import Fore

from ws.RLUtils.monitoring.tracing.log_mgt import log_mgr

if __name__ == '__main__':
    cwd = os.path.curdir
    acwd = os.path.join(cwd, '_tests')
    log_dir = os.path.join(acwd, "logs")

    fn_record = log_mgr(log_dir, show_debug=False, fixed_log_file=False)
    color_red_foreground = Fore.RED
    fn_record('1.  show_debug = False, debug=True', color=color_red_foreground, debug=True)
    fn_record('2. show_debug = False, debug=False', color=color_red_foreground, debug=False)
