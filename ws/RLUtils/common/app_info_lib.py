class DotDict(dict):
    def __getattr__(self, name):
        return self[name]
    def __setattr__(self, name, value):
        self[name] = value

def fn_init_arg_with_default_val(app_info, name, val):
    if app_info is None:
        app_info = {}
    else:
        app_info = DotDict(app_info.copy())
    app_info[name] = val
    return app_info
