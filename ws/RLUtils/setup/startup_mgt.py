import logging
import os
import shutil
from datetime import datetime

from ws.RLUtils.common.DotDict import DotDict
from ws.RLUtils.common.config_mgt import config_mgt
from ws.RLUtils.common.folder_paths import fn_separate_folderpath_and_filename, fn_get_rel_dot_folder_path
from ws.RLUtils.common.module_loader import load_function
from ws.RLUtils.monitoring.tracing.call_trace_mgt import call_trace_mgt
from ws.RLUtils.monitoring.tracing.log_mgt import log_mgt

from ws.RLUtils.platform_libs.pytorch.device_selection import get_device

def startup_mgt(caller_filepath):
    ROOT_DOT_PATH = 'ws'
    ARGS_PY = 'ARGS.py'

    def _fn_init_arg_with_default_val(app_info, name, val):
        if app_info is None:
            app_info = {}
        else:
            app_info = DotDict(app_info.copy())
        app_info[name] = val
        return app_info

    def fn_bootstrap(file_path):
        demo_folder_path, _ = fn_separate_folderpath_and_filename(file_path)
        demo_dot_path = fn_get_rel_dot_folder_path(demo_folder_path, '/ws/')
        fn_get_args = load_function(function_name="fn_get_args", module_tag="ARGS", subpackage_tag=demo_dot_path)
        app_info = fn_get_args()
        app_info = _fn_init_arg_with_default_val(app_info, 'DEMO_FOLDER_PATH_', demo_folder_path)
        app_info = _fn_init_arg_with_default_val(app_info, 'DEMO_DOT_PATH_', demo_dot_path)
        app_info = _fn_init_arg_with_default_val(app_info, 'RESULTS_REL_PATH', 'Results/')
        results_folder_path = os.path.join(app_info.DEMO_FOLDER_PATH_, app_info.RESULTS_REL_PATH)
        app_info = _fn_init_arg_with_default_val(app_info, 'RESULTS_PATH_', results_folder_path)
        # archive_dir = demo_folder_path.replace('/Demos/', '/Archives/')
        # app_info = _fn_init_arg_with_default_val(app_info, 'ARCHIVE_DIR_', archive_dir)

        # app_info = _fn_init_arg_with_default_val(app_info, 'LOGGER_', logging.getLogger(__name__))
        # app_info = _fn_init_arg_with_default_val(app_info, 'fn_log',
        #                                      log_mgt(log_dir=app_info.ARCHIVE_DIR_, fixed_log_file=True))
        # app_info = _fn_init_arg_with_default_val(app_info, 'trace_mgr', call_trace_mgt(app_info.fn_log))
        return app_info

    def _fn_setup_for_results(_app_info):
        pass

    def _fn_setup_logging(app_info):
        fn_get_key_as_bool, _, _ = config_mgt(app_info)
        debug_mode = fn_get_key_as_bool('DEBUG_MODE')
        session_repo = app_info.RESULTS_REL_PATH
        fn_log = log_mgt(log_dir=session_repo, show_debug=debug_mode)
        app_info.fn_log = fn_log
        pass

    def _fn_setup_env(app_info, verbose=False):
        subpackage_name = None
        if 'ENV_NAME' not in app_info.keys():
            if verbose:
                print("ENV_NAME is missing")
            pass
        else:
            repo_name_parts = app_info.ENV_NAME.lower().rsplit('-', 1)
            app_info['ENV_REPO'] = repo_name_parts[0]
            subpackage_name = 'ws.RLEnvironments.' + app_info['ENV_REPO']

        if subpackage_name is None:
            return app_info, None

        env_mgt = load_function(function_name="env_mgt", module_tag="env_mgt", subpackage_tag=subpackage_name)

        env = None
        if env_mgt is not None:
            env = env_mgt(app_info)
            app_info.ACTION_DIMENSIONS = env.fn_get_action_size()
            app_info.STATE_DIMENSIONS = env.fn_get_state_size()
        app_info.ENV = env
        return env

    def _fn_setup_paths_in_app_info(app_info):
        app_info.ROOT_DOT_PATH = ROOT_DOT_PATH
        app_info.AGENTS_DOT_PATH = ROOT_DOT_PATH + '.RLAgents'
        app_info.AGENTS_CONFIG_DOT_PATH = ROOT_DOT_PATH + '.RLAgents' + '.agent_configs'

        app_info.AGENT_FOLDER_PATH = app_info.AGENTS_DOT_PATH + '.{}'.format(app_info.STRATEGY)

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


        # app_info.ARCHIVE_PATH_BEFORE_ = os.path.join(app_info.FULL_ARCHIVE_PATH_, 'BEFORE')
        # app_info.ARCHIVE_PATH_AFTER_ = os.path.join(app_info.FULL_ARCHIVE_PATH_, 'AFTER')

        app_info.LOGGER_ =  logging.getLogger(__name__)
        app_info.fn_log = log_mgt(log_dir=app_info.RESULTS_PATH_, fixed_log_file=True)
        app_info.trace_mgr = call_trace_mgt(app_info.fn_log)

        pass

    def _fn_setup_gpu(_app_info, verbose=False):
        _app_info.GPU_DEVICE = get_device(_app_info)
        if verbose:
            print('DEVICE: {}'.format(_app_info.GPU_DEVICE))
        pass

    app_info = fn_bootstrap(caller_filepath)

    _fn_setup_paths_in_app_info(app_info)
    # _fn_setup_for_results(app_info)
    _fn_setup_logging(app_info)
    _fn_setup_gpu(app_info)
    _fn_setup_env(app_info)

    return app_info




