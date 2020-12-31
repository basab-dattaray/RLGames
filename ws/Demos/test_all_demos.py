import os

from ws.RLUtils.setup.preparation_mgt import fn_load_app
def _fn_run_file(rel_dir):
    parent_dir = os.getcwd()
    child_dir = os.path.join(parent_dir, rel_dir + '/gridworld')
    file = os.path.join(child_dir, 'demo_run.py')
    fn_load_app(file)

def test_fn_get_state():
    _fn_run_file('Demo001_policy_iterator')
    _fn_run_file('Demo002_value_iterator')


