import logging
import os
import shutil
from datetime import date, datetime

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

    def _fn_get_another_archive_path():
        current_time_id = date.now().strftime("%Y_%m_%d_%H_%M_%S")
        new_archive_folder_path = os.path.join(app_info.RESULTS_ARCHIVE_PATH, current_time_id)
        return new_archive_folder_path

    def _fn_init_arg_with_default_val(args, name, val):
        if args is None:
            args = {}
        else:
            args = DotDict(args.copy())
        args[name] = val
        return args

    def fn_bootstrap(file_path):
        demo_folder_path, demo_file_name = fn_separate_folderpath_and_filename(file_path)
        demo_dot_path = fn_get_rel_dot_folder_path(demo_folder_path, '/ws/')
        fn_get_args = load_function(function_name="fn_get_args", module_tag="ARGS", subpackage_tag=demo_dot_path)
        args = fn_get_args()
        args = _fn_init_arg_with_default_val(args, 'DEMO_FOLDER_PATH_', demo_folder_path)
        args = _fn_init_arg_with_default_val(args, 'DEMO_FILE_NAME_', demo_file_name)
        args = _fn_init_arg_with_default_val(args, 'DEMO_DOT_PATH_', demo_dot_path)
        args = _fn_init_arg_with_default_val(args, 'RESULTS_REL_PATH', 'Results/')
        results_folder_path = os.path.join(args.DEMO_FOLDER_PATH_, args.RESULTS_REL_PATH)
        args = _fn_init_arg_with_default_val(args, 'RESULTS_PATH_', results_folder_path)
        if 'MODEL_NAME' in args:
            args = _fn_init_arg_with_default_val(args, 'MODEL_FILEPATH_',
                                                 os.path.join(results_folder_path, args.MODEL_NAME))
            args = _fn_init_arg_with_default_val(args, 'OLD_MODEL_FILEPATH_',
                                                 os.path.join(results_folder_path, 'old_' + args.MODEL_NAME))
        # current_dir = file_path.rsplit('/', 1)[0]
        archive_dir = demo_folder_path.replace('/Demos/', '/Archives/')
        args = _fn_init_arg_with_default_val(args, 'ARCHIVE_DIR_', archive_dir)
        args = _fn_init_arg_with_default_val(args, 'LOGGER_', logging.getLogger(__name__))
        args = _fn_init_arg_with_default_val(args, 'fn_record',
                                             log_mgt(log_dir=args.ARCHIVE_DIR_, fixed_log_file=True))
        args = _fn_init_arg_with_default_val(args, 'CALL_TRACER_', call_trace_mgt(args.fn_record))
        # args_copy = _fn_arg_defaults(args)
        return args

    def _fn_setup_for_results(_app_info):
        results_folder = os.path.join(_app_info.DEMO_FOLDER_PATH_, "Results")
        if os.path.exists(results_folder) is False:
            os.makedirs(results_folder)
        # _app_info.RESULTS_PATH_ = results_folder
        args_module_path = os.path.join(_app_info.DEMO_FOLDER_PATH_, ARGS_PY)
        if os.path.exists(args_module_path):
            shutil.copy(args_module_path, _app_info.RESULTS_REL_PATH)

    def _fn_setup_logging(app_info):
        fn_get_key_as_bool, _, _ = config_mgt(app_info)
        debug_mode = fn_get_key_as_bool('DEBUG_MODE')
        session_repo = app_info.RESULTS_REL_PATH
        fn_log = log_mgt(log_dir=session_repo, show_debug=debug_mode)
        app_info.FN_RECORD = fn_log
        # app_info.FN_RECORD('hi')
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

        archive_container_path = os.path.join(app_info.DEMO_FOLDER_PATH_.replace('Demos', 'ARCHIVES'),
                                                     app_info.DEMO_FILE_NAME_)
        current_time_id = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        app_info.RESULTS_ARCHIVE_PATH = os.path.join(archive_container_path, current_time_id)
        pass

    def _fn_setup_gpu(_app_info, verbose=False):
        _app_info.GPU_DEVICE = get_device(_app_info)
        if verbose:
            print('DEVICE: {}'.format(_app_info.GPU_DEVICE))
        pass

    app_info = fn_bootstrap(caller_filepath)

    _fn_setup_paths_in_app_info(app_info)
    _fn_setup_for_results(app_info)
    _fn_setup_logging(app_info)
    _fn_setup_gpu(app_info)
    _fn_setup_env(app_info)

    return app_info




