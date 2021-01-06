import os
from pathlib import Path

from ws.RLAgents.algo_lib.logic.support.agent_config_mgt import agent_config_mgt
from ws.RLUtils.common.config_mgt import config_mgt
from ws.RLUtils.common.fileops import get_json_data
from ws.RLUtils.common.module_loader import load_function
from ws.RLUtils.monitoring.tracing.log_mgt import log_mgt

from ws.RLUtils.platform_libs.pytorch.device_selection import get_device

ROOT_DOT_PATH = 'ws.RLAgents'
def fn_load_app(file_path):
    app_info, env = preparation_mgt(file_path)
    agent_config_mgt(app_info)
    subpackage_name = 'ws.RLAgents.{}'.format(app_info['STRATEGY'])
    agent_mgt = load_function(function_name="agent_mgt", module_tag="agent_mgt", subpackage_tag=subpackage_name)
    fn_init = agent_mgt(app_info, env)
    fn_init()

def fn_set_app_info_paths(_app_info, calling_root_path):
    package_root_path = str(Path(calling_root_path).parent.parent)
    base_container_path = str(Path(package_root_path).parent.parent)
    relative_package_path = package_root_path.replace(base_container_path, '')
    root_dot_path = relative_package_path.replace('/', '.')
    _app_info['ROOT_DOT_PATH'] = root_dot_path

def fn_gpu_setup(_app_info, verbose= False):
    # GPU
    _app_info['GPU_DEVICE'] = get_device(_app_info)
    if verbose:
        print('DEVICE: {}'.format(_app_info['GPU_DEVICE']))
    pass

def fn_setup_for_results(_app_info, verbose= False):

    # _app_info['GPU_DEVICE'] = get_device(_app_info)
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


def preparation_mgt(calling_filepath, verbose=False):
    filepathname_parts = calling_filepath.rsplit('/', 1)
    cwd = filepathname_parts[0]

    filename = filepathname_parts[1]
    filename_parts = filename.rsplit('_', 1)
    demo_name = filename_parts[0]
    app_info_file = demo_name + '_APP_INFO.JSON'

    _app_info_path = os.path.join(cwd, app_info_file)
    _app_info = get_json_data(_app_info_path)
    _app_info['APP_INFO_SOURCE'] = _app_info_path
    _app_info['DEMO_PATH'] = cwd





    def _fn_setup_paths_in_app_info():
        _app_info['AGENT_FOLDER_PATH'] = ROOT_DOT_PATH + '.{}'.format(_app_info['STRATEGY'])
        _app_info['ARCHIVE_SUB_FOLDER'] = demo_name # fn_get_archive_name()

        base_path = _app_info['RESULTS_BASE_PATH']
        if _app_info['ARCHIVE_SUB_FOLDER'] is not None:
            base_path = os.path.join(base_path, _app_info['ARCHIVE_SUB_FOLDER'])

        _app_info['RESULTS_CURRENT_PATH'] = os.path.join(base_path, 'Current')
        _app_info['RESULTS_ARCHIVE_PATH'] = os.path.join(_app_info['DEMO_PATH'].replace('Demos', 'ARCHIVES'), demo_name)
        if verbose:
            print(_app_info['ARCHIVE_SUB_FOLDER'])




    def _fn_get_env():
        subpackage_name = None
        if 'ENV_NAME' not in _app_info.keys():
            if verbose:
                print("ENV_NAME is missing")
            pass
        else:
            repo_name_parts = _app_info['ENV_NAME'].lower().rsplit('-', 1)
            _app_info['ENV_REPO'] = repo_name_parts[0]
            subpackage_name = 'ws.RLEnvironments.' + _app_info['ENV_REPO']

        if subpackage_name is None:
            return _app_info, None

        env_mgt = load_function(function_name="env_mgt", module_tag="env_mgt", subpackage_tag=subpackage_name)

        env = None
        if env_mgt is not None:
            env = env_mgt(_app_info)
            _app_info['ACTION_DIMENSIONS'] = env.fn_get_action_size()
            _app_info['STATE_DIMENSIONS'] = env.fn_get_state_size()

        return env

    fn_setup_for_results(_app_info)
    _fn_setup_paths_in_app_info()
    fn_setup_logging(_app_info)
    fn_gpu_setup(_app_info)

    env = _fn_get_env()

    return _app_info, env

