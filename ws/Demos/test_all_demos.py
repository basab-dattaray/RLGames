import os

from ws.RLUtils.common.folder_paths import fn_get_rel_dot_folder_path
from ws.RLUtils.setup.preparation_mgt import fn_load_app



def _fn_run_file(rel_dir):
    fn_get_rel_dot_folder_path('__file__', '/ws/')
    parent_dir = os.getcwd()
    child_dir = os.path.join(parent_dir, rel_dir + '/gridworld')
    file = os.path.join(child_dir, 'demo_run.py')
    # fn_load_app(file)
    agent = agent_mgt(__file__). \
        fn_init()

def test_fn_get_state():
    _fn_run_file('Demo001_policy_iterator')
    _fn_run_file('Demo002_value_iterator')
    _fn_run_file('Demo003_monte_carlo')
    _fn_run_file('Demo004_sarsa')
    _fn_run_file('Demo005_qlearn')


