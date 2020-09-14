import os

from ws.RLUtils.monitoring.tracing.log_mgt import log_mgr

if __name__ == '__main__':
    cwd = os.path.curdir
    acwd = os.path.join(cwd, '_tests')
    log_dir = os.path.join(acwd, "logs")

    log_dir_show_debug_False = os.path.join(log_dir, "show_debug_False")
    fn_record = log_mgr(log_dir=log_dir_show_debug_False)
    fn_record('default debug=False')
    fn_record('debug=True', debug=True)

    log_dir_show_debug_True = os.path.join(log_dir, "show_debug_True")
    fn_record2 = log_mgr(log_dir=log_dir_show_debug_True, show_debug=True)
    fn_record2('default debug=False')
    fn_record2('debug=True', debug=True)
