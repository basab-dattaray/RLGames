from collections import namedtuple

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

    def fn_archive_log_file():
        archive_msg = app_info.fn_archive()
        app_info.fn_log(archive_msg)

    common_funcs = namedtuple('_',
                                  [
                                      'fn_change_args',
                                      'fn_show_args',
                                      'fn_archive_log_file',
                                  ])
    common_funcs.fn_change_args = fn_change_args
    common_funcs.fn_show_args = fn_show_args
    common_funcs.fn_archive_log_file = fn_archive_log_file

    agent_mgt = load_mgt_function(loc_dotpath=app_info.AGENT_DOT_PATH, module_name='agent_mgt')

    agent_mgr = agent_mgt(app_info, common_funcs)

    return agent_mgr