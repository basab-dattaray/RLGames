import importlib


def get_module(tag, package_tag):
    try:
        obj = importlib.import_module(tag, package_tag)
        return obj
    except:
        return None


def load_function(function_name, module_name, module_dot_path, package_tag=None):
    MAX_ATTEMPT_COUNT = 7
    tag = module_dot_path + '.' + module_name
    obj = get_module(tag, package_tag)

    attempt_count = 0
    inner_subpackage_tag = module_dot_path
    while obj is None:
        inner_subpackage_tag = inner_subpackage_tag.rsplit('.', 1)[0]
        obj = get_module(inner_subpackage_tag + '.' + module_name, package_tag)

        attempt_count += 1
        if attempt_count > MAX_ATTEMPT_COUNT:
            return None

    fn_obj = getattr(obj, function_name)
    return fn_obj

def load_mgt_function(loc_dotpath, module_name):
    return load_function(module_name, module_name, loc_dotpath)
