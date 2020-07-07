from pathlib import Path

class AppInfo():

    @classmethod
    def fn_get_path_and_app_name(self, file_info):
        app_name = Path(file_info).stem
        dir_path = file_info.rsplit('/', maxsplit=1)[0]
        return dir_path, app_name