# import os
import shutil
from collections import namedtuple


def archive_mgt(fn_save_to_neural_net, fn_load_from_neural_net, model_folder_path):
    obj_archive_mgt = namedtuple('_', 'fn_save_archive_model, fn_load_archive_model, fn_archive_all')

    def fn_load_archive_model():
        if model_folder_path is None:
            return None

        ret = fn_load_from_neural_net(model_folder_path)
        return ret

    def fn_save_archive_model():
        fn_save_to_neural_net(model_folder_path)
        return model_folder_path

    def fn_archive_all(app_info, fn_save_archive_model=None):
        try:
            if fn_save_archive_model is not None:
                save_path = fn_save_archive_model()
                if save_path is None:
                    print("INFO:: unable to Save Model at {}".format(save_path))

            # shutil.copy(app_info.APP_INFO_SOURCE, app_info.RESULTS_PATH_)

            # Copy to Archive
            # number_of_archive_folders_to_remove = _fn_prune_archive_per_depth()
            # archive_folder = _fn_get_another_archive_path()
            if app_info.RESULTS_ARCHIVE_PATH is not None:
                shutil.copytree(app_info.RESULTS_PATH_, app_info.RESULTS_ARCHIVE_PATH, symlinks=False, ignore=None)

            return "INFO:: Sucessfully Archived at {}".format(app_info.RESULTS_ARCHIVE_PATH)

        except Exception as x:
            print(x)
            raise Exception('Exception: fn_archive_all')


    obj_archive_mgt.fn_save_archive_model = fn_save_archive_model
    obj_archive_mgt.fn_load_archive_model = fn_load_archive_model
    obj_archive_mgt.fn_archive_all = fn_archive_all


    return obj_archive_mgt
