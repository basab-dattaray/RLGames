from collections import OrderedDict, namedtuple

from ws.RLAgents.A_ModelBased_Planning.Planning.agent_mgt import agent_mgt
from ws.RLAgents.A_ModelBased_Planning.Planning.impl_mgt import impl_mgt
from ws.RLUtils.common.misc_functions import fn_get_elapsed_time

from ws.RLUtils.monitoring.tracing.tracer import tracer
from ws.RLUtils.setup.startup_mgt import startup_mgt

def agent_mgt_wrapper(file_path):
    app_info = startup_mgt(file_path, __file__)

    @tracer(app_info, verboscity=4)
    def fn_change_args(change_args):
        if change_args is not None:
            for k, v in change_args.items():
                app_info[k] = v
                app_info.trace_mgr.fn_write(f'  app_info[{k}] = {v}')
        agent_mgr.app_info = app_info
        return agent_mgr

    @tracer(app_info, verboscity=4)
    def fn_show_args():
        for k, v in app_info.items():
            app_info.trace_mgr.fn_write(f'  app_info[{k}] = {v}')
        return agent_mgr

    @tracer(app_info,  verboscity=4)
    def fn_measure_time_elapsed(start_time):

        start_time = fn_get_elapsed_time(start_time, app_info.trace_mgr.fn_write)
        return agent_mgr

    @tracer(app_info, verboscity=4)
    def fn_archive_log_file():
        archive_msg = app_info.fn_archive(archive_folder_path=app_info.FULL_ARCHIVE_PATH_,
                                          fn_save_to_neural_net=app_info.neural_net_mgr.fn_save_model)
        app_info.fn_log(archive_msg)

    export_functions = namedtuple('_',
                                  [
                                      'fn_change_args',
                                      'fn_show_args',
                                      'fn_measure_time_elapsed',
                                      'fn_archive_log_file',
                                  ])
    export_functions.fn_change_args = fn_change_args
    export_functions.fn_show_args = fn_show_args
    export_functions.fn_measure_time_elapsed = fn_measure_time_elapsed
    export_functions.fn_archive_log_file = fn_archive_log_file

    agent_mgr = agent_mgt(app_info, export_functions)

    return agent_mgr