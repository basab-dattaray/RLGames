import math
import os
import shutil
from datetime import datetime
from time import time

def archive_mgt(app_info, fn_log= None,  fn_log_reset= None):
    result_folder_paths = app_info.RESULTS_PATH_
    archive_folder_path = app_info.FULL_ARCHIVE_PATH_
    _before_instance = True
    _start_time = time()

    def fn_get_elapsed_time(start_time):
        end_time = time()
        time_diff = int(end_time - start_time)
        mins = math.floor(time_diff / 60)
        secs = time_diff % 60
        start_time_human = datetime.fromtimestamp(start_time).strftime("%H:%M:%S")
        end_time_human = datetime.fromtimestamp(end_time).strftime("%H:%M:%S")
        time_stats = f'Time elapsed:    minutes: {mins}    seconds: {secs}  -----  (start_time:{start_time_human}, end_time:{end_time_human})'
        return time_stats

    def fn_archive():
        nonlocal  _before_instance
        try:

            if _before_instance:
                real_archive_path = os.path.join( archive_folder_path , 'BEFORE')
            else:
                real_archive_path = os.path.join( archive_folder_path , 'AFTER')
            if 'fn_save_model' in  app_info:
                if app_info.fn_save_model is not None:
                    app_info.fn_save_model()

            if os.path.exists(real_archive_path):
                shutil.rmtree(real_archive_path)

            if not _before_instance:
                time_stats = fn_get_elapsed_time(start_time= app_info.START_TIME_)
                app_info.fn_log(time_stats)


            shutil.copytree(result_folder_paths, real_archive_path, symlinks=False, ignore=None)

            if _before_instance:
                _before_instance = False
                if fn_log_reset is not None:
                    fn_log_reset()

            return "INFO:: Sucessfully Archived at {}".format(archive_folder_path)

        except Exception as x:
            print(x)
            raise Exception('Exception: fn_archive')

    fn_archive()

    return fn_archive
