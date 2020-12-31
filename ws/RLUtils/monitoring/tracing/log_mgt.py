import logging, coloredlogs
import logging.handlers
import os
from datetime import datetime as dt

# from ws.RLUtils.monitoring.tracing.CustomFormatter import CustomFormatter


def log_mgt(log_dir, show_debug=False, log_file_name = 'log.txt',  fresh_logfile_content=True, fixed_log_file=True):
    _log = None
    _log_file_name = log_file_name

    _log_level = logging.INFO
    if show_debug:
        _log_level = logging.DEBUG

    def setup():
        nonlocal _log_file_name
        nonlocal _log, _log_level
        logger = logging.getLogger()
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        logging.getLogger().setLevel(_log_level)

        if os.path.exists(log_dir) is False:
            os.makedirs(log_dir)

        if not fixed_log_file:
            _log_file_name = dt.now().strftime("%Y_%m_%d_%H_%M_%S")
        logfile = os.path.join(log_dir, _log_file_name)
        if fresh_logfile_content:
            if os.path.exists(logfile):
                os.remove(logfile)

        handler = logging.handlers.RotatingFileHandler(filename=logfile, maxBytes=1000000, backupCount=5)
        handler.setLevel(logging.DEBUG)
        # formatter = logging.Formatter('%(asctime)state - %(message)state')
        # formatter = logging.Formatter(CustomFormatter)
        # handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)
        _log = logging.getLogger("app." + __name__)

    def fn_log(msg="", color="", debug=False):

        # PRINT
        if show_debug or not debug:
            print_msg = color + msg
            print(print_msg)

        # LOG
        log_op = _log.info
        if debug:
            log_op = _log.debug
        log_op(msg)

    setup()

    return fn_log
