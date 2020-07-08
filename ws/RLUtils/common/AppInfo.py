from pathlib import Path

class AppInfo():

    @classmethod
    def fn_get_path_and_app_name(cls, file_info):
        app_name = Path(file_info).stem
        dir_path = file_info.rsplit('/', maxsplit=1)[0]
        return dir_path, app_name

    @classmethod
    def fn_arg_as_bool(cls, args, arg_key):
        if arg_key not in args.keys():
            return False
        if not bool(args[arg_key]):
            return False
        return True