import os
import shutil

import pytest

from ws.RLUtils.persist.pickle_mgt import pickle_mgr



@pytest.fixture(scope='module')
def fixture_ws_mgt():
    obj = {'d1': 3, 'd2': [1, 2], 'd3': {'d31': 'D31', 'd32': 'D32'}}
    dump_folder_path = os.path.join(os.getcwd(), 'tmp_dump_dir')
    dump_file_name = 'tmp.1.obj'
    yield dump_folder_path, dump_file_name, obj
    # dump_folder_path = os.path.join(os.getcwd(), '__tmp_dump_dir')
    if os.path.exists(dump_folder_path):
        shutil.rmtree(dump_folder_path)

def test_pickler_save_and_load(fixture_ws_mgt):
    dump_folder_path, dump_file_name, obj = fixture_ws_mgt

    fn_save, fn_load = pickle_mgr(dump_folder_path, dump_file_name)

    ret_val = fn_save(obj)
    assert ret_val == True

    ret_obj = fn_load()
    assert ret_obj['d1'] == 3

    d2 = ret_obj['d2']
    assert d2[1] == 2

    d3 = ret_obj['d3']
    assert d3['d32'] == 'D32'

    pass



