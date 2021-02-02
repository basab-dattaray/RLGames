


# def fn_traverse_dir(dir_path):
#     file_list = [directory for directory in os.listdir(dir_path) if os.dir_path.isfile(directory)]
#     test_file_list = [testfile for testfile in file_list if testfile.endswith('.py') and testfile.startswith('test')]
#     dir_list = [directory for directory in os.listdir(dir_path) if os.dir_path.isdir(directory)]
#     for rel_dir_path in dir_list:
#         sub_dir = os.dir_path.join(dir_path, rel_dir_path)
#         fn_traverse_dir(sub_dir)
import os

from ws.RLUtils.common.folder_paths import fn_get_rel_dot_folder_path
from ws.RLUtils.setup.exec_mgt import exec_mgt

if __name__ == "__main__":
    cwd = os.path.dirname(__file__)
    fn_traverse_dir, fn_stats = exec_mgt(__file__)
    fn_traverse_dir(cwd)
    total_count, failures = fn_stats()
    print(f'total_count: {total_count}, failures: {failures}')





