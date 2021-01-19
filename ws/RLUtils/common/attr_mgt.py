def attr_mgt(app_info):
    def fn_get_key_as_bool(config_key):
        if config_key not in app_info.keys():
            return False
        if app_info[config_key] == 0:
            return False
        return True

    def fn_get_key_as_int(config_key, default=0):
        if config_key not in app_info.keys():
            return default

        result = int(app_info[config_key])
        return result

    def fn_get_key_as_str(config_key, default=""):
        if config_key not in app_info.keys():
            return default

        result = app_info[config_key]
        return result

    return fn_get_key_as_bool, fn_get_key_as_int, fn_get_key_as_str
