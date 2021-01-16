import os

from ws.RLUtils.monitoring.tracing.log_mgt import log_mgt

if __name__ == '__main__':
    cwd = os.path.curdir
    acwd = os.path.join(cwd, '_tests')
    log_dir = os.path.join(acwd, "logs")

    log_dir_show_debug_False = os.path.join(log_dir, "show_debug_False")
    fn_log = log_mgt(log_dir=log_dir_show_debug_False)[0]
    fn_log('default debug=False')
    fn_log('debug=True', debug=True)

    log_dir_show_debug_True = os.path.join(log_dir, "show_debug_True")
    fn_log2 = log_mgt(log_dir=log_dir_show_debug_True, show_debug=True)[0]
    fn_log2('default debug=False')
    fn_log2('debug=True', debug=True)
