from collections import namedtuple

from ws.RLUtils.common.misc_functions import fn_get_elapsed_time
from ws.RLUtils.common.module_loader import load_function, load_mgt_function

from ws.RLUtils.setup.startup_mgt import startup_mgt

def agent_dispatcher(file_path):
    app_info = startup_mgt(file_path, __file__)

    def fn_change_args(change_args, verbose= False):
        if change_args is not None:
            for k, v in change_args.items():
                app_info[k] = v
                if verbose:
                    app_info.trace_mgr.fn_write(f'  app_info[{k}] = {v}')
        agent_mgr.app_info = app_info
        return agent_mgr

    def fn_show_args():
        for k, v in app_info.items():
            app_info.trace_mgr.fn_write(f'  app_info[{k}] = {v}')
        return agent_mgr

    def fn_measure_time_elapsed(start_time):

        start_time = fn_get_elapsed_time(start_time, app_info.trace_mgr.fn_write)
        return agent_mgr

    def fn_archive_log_file():
        archive_msg = app_info.fn_archive(archive_folder_path=app_info.FULL_ARCHIVE_PATH_,
                                          fn_save_model=app_info.neural_net_mgr.fn_save_model)
        app_info.fn_log(archive_msg)

    common_funcs = namedtuple('_',
                                  [
                                      'fn_change_args',
                                      'fn_show_args',
                                      'fn_measure_time_elapsed',
                                      'fn_archive_log_file',
                                  ])
    common_funcs.fn_change_args = fn_change_args
    common_funcs.fn_show_args = fn_show_args
    common_funcs.fn_measure_time_elapsed = fn_measure_time_elapsed
    common_funcs.fn_archive_log_file = fn_archive_log_file

    agent_mgt = load_mgt_function(loc_dotpath=app_info.AGENT_DOT_PATH, module_name='agent_mgt')

    agent_mgr = agent_mgt(app_info, common_funcs)

    return agent_mgr