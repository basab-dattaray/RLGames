from ws.RLUtils.common.module_loader import get_module


def loader_mgr():
    def fn_load_top_object(init_search_dir, module_name, function_name, iterations=10):
        search_dir = init_search_dir
        module_path = '{}.{}'.format(search_dir, module_name)
        module_obj = get_module(module_path, None)
        while module_obj is None and iterations > 0:
            search_dir = search_dir.rsplit('.', 1)[0]
            module_path = '{}.{}'.format(search_dir, module_name)
            module_obj = get_module(module_path, None)

        obj = getattr(module_obj, function_name)
        return obj

    return fn_load_top_object
