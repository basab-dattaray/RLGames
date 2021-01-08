import os
from pathlib import Path

from ws.RLUtils.common.config_mgt import config_mgt
from ws.RLUtils.common.module_loader import load_function
from ws.RLUtils.monitoring.tracing.log_mgt import log_mgt

from ws.RLUtils.platform_libs.pytorch.device_selection import get_device
from ws.RLUtils.setup.args_mgt import args_mgt


def fn_set_agent_path_in_app_info(_app_info, calling_root_path):
    package_root_path = str(Path(calling_root_path).parent.parent.parent)
    base_container_path = str(Path(package_root_path).parent)
    relative_package_path = package_root_path.replace(base_container_path, '')
    root_dot_path = relative_package_path.replace('/', '.')[1:]
    _app_info['ROOT_DOT_PATH'] = root_dot_path
    _app_info['AGENTS_DOT_PATH'] = root_dot_path + '.RLAgents'
    _app_info['AGENTS_CONFIG_DOT_PATH'] = root_dot_path + '.RLAgents' + '.agent_configs'
    pass

def fn_gpu_setup(_app_info, verbose= False):
    # GPU
    _app_info['GPU_DEVICE'] = get_device(_app_info)
    if verbose:
        print('DEVICE: {}'.format(_app_info['GPU_DEVICE']))
    pass

def fn_setup_for_results(_app_info, verbose= False):

    # app_info['GPU_DEVICE'] = get_device(app_info)
    results_folder = os.path.join(_app_info['DEMO_PATH'], "Results")
    if os.path.exists(results_folder) is False:
        os.makedirs(results_folder)

    env_folder = os.path.join(results_folder, _app_info['ENV_NAME'])
    if os.path.exists(env_folder) is False:
        os.makedirs(env_folder)
    _app_info['RESULTS_BASE_PATH'] = env_folder
    if verbose:
        print(env_folder)

def fn_setup_logging(app_info):
    fn_get_key_as_bool, fn_get_key_as_int, fn_get_key_as_str = config_mgt(app_info)
    debug_mode = fn_get_key_as_bool('DEBUG_MODE')
    session_repo = app_info['RESULTS_CURRENT_PATH']
    fn_log = log_mgt(log_dir= session_repo, show_debug=debug_mode)
    app_info['FN_RECORD'] = fn_log
    pass


def fn_get_env(app_info, verbose= False):
    subpackage_name = None
    if 'ENV_NAME' not in app_info.keys():
        if verbose:
            print("ENV_NAME is missing")
        pass
    else:
        repo_name_parts = app_info['ENV_NAME'].lower().rsplit('-', 1)
        app_info['ENV_REPO'] = repo_name_parts[0]
        subpackage_name = 'ws.RLEnvironments.' + app_info['ENV_REPO']

    if subpackage_name is None:
        return app_info, None

    env_mgt = load_function(function_name="env_mgt", module_tag="env_mgt", subpackage_tag=subpackage_name)

    env = None
    if env_mgt is not None:
        env = env_mgt(app_info)
        app_info['ACTION_DIMENSIONS'] = env.fn_get_action_size()
        app_info['STATE_DIMENSIONS'] = env.fn_get_state_size()

    return env

def fn_setup_paths_in_app_info(app_info, cwd,  verbose= False):
    fn_set_agent_path_in_app_info(app_info, cwd)
    app_info.AGENT_FOLDER_PATH =app_info.AGENTS_DOT_PATH  + '.{}'.format(app_info['STRATEGY'])

    base_path = app_info.RESULTS_BASE_PATH
    if app_info['ARCHIVE_SUB_FOLDER'] is not None:
        base_path = os.path.join(base_path, app_info['ARCHIVE_SUB_FOLDER'])

    app_info['RESULTS_CURRENT_PATH'] = os.path.join(base_path, 'Current')
    app_info['RESULTS_ARCHIVE_PATH'] = os.path.join(app_info['DEMO_PATH'].replace('Demos', 'ARCHIVES'), app_info['ARCHIVE_SUB_FOLDER'])
    if verbose:
        print(app_info['ARCHIVE_SUB_FOLDER'])

def startup_mgt(calling_filepath):
    _app_info = args_mgt(calling_filepath)
    filepathname_parts = calling_filepath.rsplit('/', 1)
    cwd = filepathname_parts[0]
    #
    filename = filepathname_parts[1]
    filename_parts = filename.rsplit('_', 1)

    _app_info['DEMO_PATH'] = cwd
    _app_info['ARCHIVE_SUB_FOLDER'] = filename_parts[0]

    fn_setup_for_results(_app_info)
    fn_setup_paths_in_app_info(_app_info, cwd)
    fn_setup_logging(_app_info)
    fn_gpu_setup(_app_info)
    env = fn_get_env(_app_info)
    return _app_info, env

