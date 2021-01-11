import os
from pathlib import Path

from ws.RLUtils.common.config_mgt import config_mgt
from ws.RLUtils.common.folder_paths import fn_separate_folderpath_and_filename
from ws.RLUtils.common.module_loader import load_function
from ws.RLUtils.monitoring.tracing.log_mgt import log_mgt

from ws.RLUtils.platform_libs.pytorch.device_selection import get_device
from ws.RLUtils.setup.args_mgt import args_mgt

ROOT_DOT_PATH = 'ws'
def _fn_setup_gpu(_app_info, verbose= False):
    _app_info['GPU_DEVICE'] = get_device(_app_info)
    if verbose:
        print('DEVICE: {}'.format(_app_info['GPU_DEVICE']))
    pass

def _fn_setup_for_results(_app_info):
    results_folder = os.path.join(_app_info['DEMO_FOLDER_PATH'], "Results")
    if os.path.exists(results_folder) is False:
        os.makedirs(results_folder)
    _app_info['RESULTS_BASE_PATH'] = results_folder

def _fn_setup_logging(app_info):
    fn_get_key_as_bool, fn_get_key_as_int, fn_get_key_as_str = config_mgt(app_info)
    debug_mode = fn_get_key_as_bool('DEBUG_MODE')
    session_repo = app_info['RESULTS_BASE_PATH']
    fn_log = log_mgt(log_dir= session_repo, show_debug=debug_mode)
    app_info['FN_RECORD'] = fn_log
    pass


def _fn_setup_env(app_info, verbose= False):
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
    app_info.ENV = env
    return env

def _fn_setup_paths_in_app_info(app_info):
    app_info['ROOT_DOT_PATH'] = ROOT_DOT_PATH
    app_info['AGENTS_DOT_PATH'] = ROOT_DOT_PATH + '.RLAgents'
    app_info['AGENTS_CONFIG_DOT_PATH'] = ROOT_DOT_PATH + '.RLAgents' + '.agent_configs'

    app_info.AGENT_FOLDER_PATH =app_info.AGENTS_DOT_PATH  + '.{}'.format(app_info['STRATEGY'])

    app_info['RESULTS_ARCHIVE_PATH'] = os.path.join(app_info['DEMO_FOLDER_PATH'].replace('Demos', 'ARCHIVES'), app_info['FULL_DEMO_PATHNAME'])


def startup_mgt(callar_filepath):
    _app_info = args_mgt(callar_filepath)

    demo_folder, demo_file_name = fn_separate_folderpath_and_filename(callar_filepath)

    _app_info['DEMO_FOLDER_PATH'] = demo_folder
    _app_info['FULL_DEMO_PATHNAME'] = demo_file_name

    _fn_setup_for_results(_app_info)
    _fn_setup_paths_in_app_info(_app_info)
    _fn_setup_logging(_app_info)
    _fn_setup_gpu(_app_info)
    env = _fn_setup_env(_app_info)
    # _app_info.ENV = env
    return _app_info




