import importlib
import os


# def fn_traverse_dir(dir_path):
#     file_list = [directory for directory in os.listdir(dir_path) if os.dir_path.isfile(directory)]
#     test_file_list = [testfile for testfile in file_list if testfile.endswith('.py') and testfile.startswith('test')]
#     dir_list = [directory for directory in os.listdir(dir_path) if os.dir_path.isdir(directory)]
#     for rel_dir_path in dir_list:
#         sub_dir = os.dir_path.join(dir_path, rel_dir_path)
#         fn_traverse_dir(sub_dir)
from ws.RLUtils.common.folder_paths import fn_get_rel_dot_folder_path


def fn_traverse_dir(dir_path, base_path):
    for item in os.listdir(dir_path):
        abspath_item = os.path.join(dir_path, item)
        if os.path.isfile(abspath_item):
            if item.startswith('test') and item.endswith('.py'):
                prefix, dot_path = fn_get_rel_dot_folder_path(current_path=abspath_item,
                                                                          base_path=base_path)
                module_name = item.rsplit('.', 2)[0]
                module_dot_path = f'ws.{dot_path}.{module_name}'
                module_obj = importlib.import_module(module_dot_path, '')
                # function_does_exists = hasattr(module_obj, 'fn_exec_test1')
                if 'fn_exec_test' in module_obj.__dict__.keys():
                    fn_obj = getattr(module_obj, 'fn_exec_test')
                    fn_obj()
                    print(f'EXECUTED: {module_dot_path}')
                # with open(abspath_item) as source_file:
                #     exec(source_file.read())
        if os.path.isdir(abspath_item):
            fn_traverse_dir(abspath_item, base_path)


if __name__ == "__main__":
    cwd = os.path.dirname(__file__)
    fn_traverse_dir(cwd, __file__)





