import importlib
import os

from ws.RLUtils.common.folder_paths import fn_get_rel_dot_folder_path

def exec_mgt(base_path, file_prefix='test', file_postfix='.py'):
    _total_count = 0
    _failures = 0
    def fn_traverse_dir(dir_path):
        nonlocal _total_count, _failures
        sorted_list = sorted(os.listdir(dir_path))
        for item in sorted_list:
            abspath_item = os.path.join(dir_path, item)
            if os.path.isfile(abspath_item):
                if item.startswith(file_prefix) and item.endswith(file_postfix):
                    prefix, dot_path = fn_get_rel_dot_folder_path(current_path=abspath_item,
                                                                              base_path=base_path)
                    module_name = item.rsplit('.', 2)[0]
                    module_dot_path = f'ws.{dot_path}.{module_name}'
                    module_obj = importlib.import_module(module_dot_path, '')
                    # function_does_exists = hasattr(module_obj, 'fn_execute1')
                    if 'fn_execute' in module_obj.__dict__.keys():
                        fn_obj = getattr(module_obj, 'fn_execute')
                        error_msg = fn_obj()
                        if error_msg is not None:
                            _failures += 1
                            print(f'ERROR: {error_msg}')

                        _total_count = _total_count + 1

                        print('')
                        print(f'\033[1m{_total_count}: *** EXECUTED: {module_dot_path}\033[0m ')
                        print('')

                    # with open(abspath_item) as source_file:
                    #     exec(source_file.read())
            if os.path.isdir(abspath_item):
                fn_traverse_dir(abspath_item)

    def fn_stats():
        return _total_count, _failures

    return fn_traverse_dir, fn_stats