import importlib
import os

from ws.RLUtils.common.folder_paths import fn_get_rel_dot_folder_path


def exec_mgt(base_path):
    _count = 0
    def fn_traverse_dir(dir_path):
        nonlocal _count
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

                        _count = _count + 1
                        print(f'{_count}: *@*@*@ *@*@*@ *@*@*@ *@*@*@ *@*@*@ *@*@*@ *@*@*@ *@*@*@ *@*@*@ EXECUTED: {module_dot_path}')
                        print('')

                    # with open(abspath_item) as source_file:
                    #     exec(source_file.read())
            if os.path.isdir(abspath_item):
                fn_traverse_dir(abspath_item)
    return fn_traverse_dir