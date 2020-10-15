import os
from sys import getrecursionlimit

from ws.RLInterfaces.PARAM_KEY_NAMES import AGENT_FOLDER_PATH, ENV_NAME, ENV_REPO, DEMO_PATH, STATE_DIMENSIONS, \
    ACTION_DIMENSIONS, \
    RESULTS_BASE_PATH, \
    STRATEGY, RESULTS_CURRENT_PATH, RESULTS_ARCHIVE_PATH, GPU_DEVICE, ARCHIVE_SUB_FOLDER, FN_RECORD, \
    DEBUG_MODE, APP_INFO_SOURCE, ARCHIVES
from ws.RLUtils.common.config_mgt import config_mgt
from ws.RLUtils.common.fileops import get_json_data
from ws.RLUtils.common.loader_mgt import loader_mgt
from ws.RLUtils.common.module_loader import load_function
from ws.RLUtils.monitoring.tracing.log_mgt import log_mgt

from ws.RLUtils.platform_libs.pytorch.device_selection import get_device


def preparation_mgt(calling_filepath, verbose=False):
    filepathname_parts = calling_filepath.rsplit('/', 1)
    cwd = filepathname_parts[0]
    filename = filepathname_parts[1]
    filename_parts = filename.rsplit('_', 1)
    demo_name = filename_parts[0]
    app_info_file = demo_name + '_APP_INFO.JSON'

    # cwd, app_info_file = APP_INFO_FILE
    _app_info_path = os.path.join(cwd, app_info_file)
    _app_info = get_json_data(_app_info_path)

    def _fn_setup_for_results():
        _app_info[DEMO_PATH] = cwd
        _app_info[GPU_DEVICE] = get_device(_app_info)
        results_folder = os.path.join(_app_info[DEMO_PATH], "Results")
        if os.path.exists(results_folder) is False:
            os.makedirs(results_folder)

        env_folder = os.path.join(results_folder, _app_info[ENV_NAME])
        if os.path.exists(env_folder) is False:
            os.makedirs(env_folder)
        _app_info[RESULTS_BASE_PATH] = env_folder
        if verbose:
            print(env_folder)

    def _fn_setup_paths_in_app_info():
        try:
            # establish archiving paths
            _app_info[AGENT_FOLDER_PATH] = 'ws.RLAgents.{}'.format(_app_info[STRATEGY])
            fn_load_top_object = loader_mgt()
            # plugin_mgt = fn_load_top_object(_app_info[AGENT_FOLDER_PATH], 'plugin_mgt', 'plugin_mgt')
            #
            # fn_get_archive_name, fn_get_subfolder_name = plugin_mgt(_app_info)
            _app_info[ARCHIVE_SUB_FOLDER] = demo_name # fn_get_archive_name()

            base_path = _app_info[RESULTS_BASE_PATH]
            # x=  _app_info[DEMO_PATH].replace('Demos', '_Logs')
            # sub_folder_name = demo_name # fn_get_subfolder_name ()
            if _app_info[ARCHIVE_SUB_FOLDER] is not None:
                base_path = os.path.join(base_path, _app_info[ARCHIVE_SUB_FOLDER])

            _app_info[RESULTS_CURRENT_PATH] = os.path.join(base_path, 'Current')
            _app_info[RESULTS_ARCHIVE_PATH] = os.path.join(_app_info[DEMO_PATH].replace('Demos', ARCHIVES), demo_name)
            if verbose:
                print(_app_info[ARCHIVE_SUB_FOLDER])
            pass
        except Exception as x:
            if verbose:
                print(x)
            exit()

    def _fn_setup_logging():
        fn_get_key_as_bool, fn_get_key_as_int, fn_get_key_as_str = config_mgt(_app_info)
        debug_mode = fn_get_key_as_bool(DEBUG_MODE)
        session_repo = _app_info[RESULTS_CURRENT_PATH]
        fn_record = log_mgt(log_dir= session_repo, show_debug=debug_mode)
        _app_info[FN_RECORD] = fn_record
        pass

    def _fn_get_env():
        subpackage_name = None
        if ENV_NAME not in _app_info.keys():
            if verbose:
                print("ENV_NAME is missing")
            pass
        else:
            repo_name_parts = _app_info[ENV_NAME].lower().rsplit('-', 1)
            _app_info[ENV_REPO] = repo_name_parts[0]
            subpackage_name = 'ws.RLEnvironments.' + _app_info[ENV_REPO]

        if subpackage_name is None:
            return _app_info, None

        Env = load_function(function_name="Env", module_tag="Env", subpackage_tag=subpackage_name)

        env = None
        if Env is not None:
            env = Env(_app_info)
            _app_info[ACTION_DIMENSIONS] = env.fnGetActionDimensions()
            _app_info[STATE_DIMENSIONS] = env.fnGetStateDimensions()

        return env

    def _fn_misc_setup():
        # APP_INFO source
        _app_info[APP_INFO_SOURCE] = _app_info_path

        # GPU
        _app_info[GPU_DEVICE] = get_device(_app_info)
        if verbose:
            print('DEVICE: {}'.format(_app_info[GPU_DEVICE]))

    # setrecursionlimit(80) #!!!
    rl = getrecursionlimit()

    _fn_setup_for_results()
    _fn_setup_paths_in_app_info()
    _fn_setup_logging()
    _fn_misc_setup()

    env = _fn_get_env()
    # store_model_dot_path(_app_info, calling_filepath)

    return _app_info, env

