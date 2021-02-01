import os
import shutil
from datetime import datetime

from ws.RLUtils.common.DotDict import DotDict
from ws.RLUtils.common.attr_mgt import attr_mgt
from ws.RLUtils.common.folder_paths import fn_get_rel_dot_folder_path

from ws.RLUtils.common.module_loader import load_function
from ws.RLUtils.monitoring.tracing.call_trace_mgt import call_trace_mgt
from ws.RLUtils.monitoring.tracing.log_mgt import log_mgt
from ws.RLUtils.platform_libs.pytorch.device_selection import get_device
from ws.RLUtils.setup.archive_mgt import archive_mgt
from ws.RLUtils.setup.interrupt_mgt import interrupt_mgt

def startup_mgt(demo_filepath, agent_filepath):
    ARGS_PY = 'ARGS.py'
    def _fn_init_arg_with_default_val(app_info, name, val):
        if app_info is None:
            app_info = {}
        else:
            app_info = DotDict(app_info.copy())
        app_info[name] = val
        return app_info

    def fn_bootstrap():
        demo_folder_path= os.path.dirname(demo_filepath)
        base_dot_path, demo_dot_path = fn_get_rel_dot_folder_path(current_path= demo_folder_path, base_path=agent_filepath)
        fn_get_args = load_function(function_name="fn_get_args", module_name="ARGS", module_dot_path=demo_dot_path)
        app_info = fn_get_args()
        app_info = _fn_init_arg_with_default_val(app_info, 'BASE_DOT_PATH_', base_dot_path)
        app_info = _fn_init_arg_with_default_val(app_info, 'DEMO_FOLDER_PATH_', demo_folder_path)
        app_info = _fn_init_arg_with_default_val(app_info, 'DEMO_DOT_PATH_', demo_dot_path)
        app_info = _fn_init_arg_with_default_val(app_info, 'RESULTS_REL_PATH', 'Results/')
        results_folder_path = os.path.join(app_info.DEMO_FOLDER_PATH_, app_info.RESULTS_REL_PATH)
        app_info = _fn_init_arg_with_default_val(app_info, 'RESULTS_PATH_', results_folder_path)
        return app_info

    def _fn_setup_logging():
        debug_mode = fn_get_key_as_bool('DEBUG_MODE')
        app_info.fn_log, app_info.fn_log_reset = log_mgt(log_dir=app_info.RESULTS_PATH_, show_debug=debug_mode)
        app_info.trace_mgr = call_trace_mgt(app_info.fn_log)
        pass

    def _fn_setup_env():
        subpackage_name = None
        if 'ENV_NAME' not in app_info.keys():
                print("ENV_NAME is missing")
        else:
            repo_name_parts = app_info.ENV_NAME.lower().rsplit('-', 1)
            app_info.ENV_REPO = repo_name_parts[0]
            subpackage_name = 'ws.RLEnvironments.' + app_info.ENV_REPO

        if subpackage_name is None:
            return app_info, None

        env_mgt = load_function(function_name="env_mgt", module_name="env_mgt", module_dot_path=subpackage_name)

        env = None
        if env_mgt is not None:
            env = env_mgt(app_info.ENV_NAME, app_info.STRATEGY)
            app_info.ACTION_DIMENSIONS = env.fn_get_action_size()
            app_info.STATE_DIMENSIONS = env.fn_get_state_size()
            app_info.ENV = env
        return env

    def _fn_setup_paths_in_app_info():
        app_info.AGENTS_DOT_PATH_ = app_info.BASE_DOT_PATH_ + '.RLAgents'
        app_info.AGENTS_CONFIG_DOT_PATH = app_info.BASE_DOT_PATH_ + '.RLAgents' + '._CommonAgentConfigurations'

        app_info.AGENT_FOLDER_PATH = app_info.AGENTS_DOT_PATH_ + '.{}'.format(app_info.STRATEGY)

        archive_container_path = app_info.DEMO_FOLDER_PATH_.replace('Demos', 'ARCHIVES')
        current_time_id = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        app_info.FULL_ARCHIVE_PATH_ = os.path.join(archive_container_path, current_time_id)

        results_folder = os.path.join(app_info.DEMO_FOLDER_PATH_, "Results")
        if os.path.exists(results_folder) is False:
            os.makedirs(results_folder)
        app_info.RESULTS_PATH_ = results_folder

        results_args_py_path = os.path.join(app_info.RESULTS_PATH_, ARGS_PY)
        if os.path.exists(results_args_py_path):
            os.remove(results_args_py_path)
        args_py_path = os.path.join(app_info.DEMO_FOLDER_PATH_, ARGS_PY)
        shutil.copy(args_py_path, app_info.RESULTS_PATH_)

    def _fn_setup_gpu():
        app_info.GPU_DEVICE = get_device(app_info)

        print('DEVICE: {}'.format(app_info.GPU_DEVICE))
        pass

    def _fn_load_agent_config():
        if 'AGENT_CONFIG' not in app_info.keys():
            return

        agent_config_path = app_info.AGENTS_CONFIG_DOT_PATH + '.' + app_info.AGENT_CONFIG
        fn_add_configs = load_function(function_name="fn_add_configs", module_name="AGENT_CONFIG",
                                       module_dot_path=agent_config_path)

        fn_add_configs(app_info)
        pass

    app_info = fn_bootstrap()
    fn_get_key_as_bool, fn_get_key_as_int, fn_get_key_as_str = attr_mgt(app_info)

    _fn_setup_paths_in_app_info()
    _fn_setup_logging()
    _fn_setup_gpu()
    _fn_setup_env()
    _fn_load_agent_config()

    if fn_get_key_as_bool('AUTO_ARCHIVE'):
        app_info.fn_archive = archive_mgt(
            results_path= app_info.RESULTS_PATH_,
            archive_path=app_info.FULL_ARCHIVE_PATH_,
            fn_log=app_info.fn_log,
            fn_log_reset = app_info.fn_log_reset,
        )

    if fn_get_key_as_bool('AUTO_INTERRUPT_HANDLING'):
        interrupt_mgt(app_info)

    return app_info




