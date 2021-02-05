import os
import shutil
from time import time

from ws.RLUtils.common.misc_functions import fn_get_elapsed_time


def archive_mgt(app_info, fn_log= None,  fn_log_reset= None):
    result_folder_paths = app_info.RESULTS_PATH_
    archive_folder_path = app_info.FULL_ARCHIVE_PATH_
    _before_instance = True
    _start_time = time()

    def fn_archive():
        nonlocal  _before_instance
        try:

            if _before_instance:
                real_archive_path = os.path.join( archive_folder_path , 'BEFORE')
            else:
                real_archive_path = os.path.join( archive_folder_path , 'AFTER')

            # if fn_save_model is not None:
            #     fn_save_model()

            if os.path.exists(real_archive_path):
                shutil.rmtree(real_archive_path)

            if not _before_instance:
                end_time = fn_get_elapsed_time(_start_time, fn_log)

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
