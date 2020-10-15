import os
import shutil
from collections import namedtuple
from datetime import datetime as dt

from ws.RLInterfaces.PARAM_KEY_NAMES import APP_INFO_SOURCE, RESULTS_CURRENT_PATH

def archive_mgt(fn_save_to_neural_net, fn_load_from_neural_net, archive_folder_path, current_folder_path,
                max_result_count):
    obj_archive_mgt = namedtuple('_', 'fn_save_model, fn_load_model, fn_archive_all')
    def fn_sort_names_in_folder_by_newest(folder_path, name_exclusions=None):
        if name_exclusions is None:
            name_exclusions = []
        if not os.path.exists(folder_path):
            return None

        items = os.listdir(folder_path)
        items.sort(reverse=True)

        paths = [os.path.join(folder_path, basename) for basename in items if basename not in name_exclusions]
        if len(paths) == 0:
            return None
        return paths

    def _fn_prune_archive_per_depth():
        sorted_model_folders = fn_sort_names_in_folder_by_newest(archive_folder_path, [])
        num_subfolders_to_be_removed = 0
        if sorted_model_folders is not None:
            num_subfolders_to_be_removed = sorted_model_folders[max_result_count - 1:]
            for subfolder in num_subfolders_to_be_removed:
                if os.path.isdir(subfolder):
                    shutil.rmtree(subfolder)
        return num_subfolders_to_be_removed

    def fn_load_model():
        if current_folder_path is False:
            return None
        # _fn_prune_archive_per_depth()
        return fn_load_from_neural_net(current_folder_path)


    def _fn_get_another_archive_path():

        current_time_id = dt.now().strftime("%Y_%m_%d_%H_%M_%S")
        new_archive_folder_path = os.path.join(archive_folder_path, current_time_id)
        return new_archive_folder_path

    def fn_save_model():

        try:
            fn_save_to_neural_net(current_folder_path)

            return current_folder_path
        except Exception as x:
            return None



    def fn_archive_all(app_info, fn_save_model=None):
        try:
            if fn_save_model is not None:
                save_path = fn_save_model()
                if save_path is None:
                    print("INFO:: unable to Save Model at {}".format(save_path))


            # remove JSON files from results_current_path
            if os.path.exists(app_info[RESULTS_CURRENT_PATH]):
                files_in_results_path = os.listdir(app_info[RESULTS_CURRENT_PATH])
                # json_files = []
                for file_name in files_in_results_path:

                    if file_name.rsplit('.', 1)[1] == 'JSON':
                        file_path = os.path.join(app_info[RESULTS_CURRENT_PATH], file_name)
                        os.remove(file_path)

            shutil.copy(app_info[APP_INFO_SOURCE], app_info[RESULTS_CURRENT_PATH])

            # Copy to Archive
            number_of_archive_folders_to_remove = _fn_prune_archive_per_depth()
            archive_folder = _fn_get_another_archive_path()
            if archive_folder is not None:
                shutil.copytree(app_info[RESULTS_CURRENT_PATH], archive_folder, symlinks=False, ignore=None)

            return "INFO:: Sucessfully Archived at {}".format(archive_folder)

        except Exception as x:
            print(x)
            raise Exception('Exception: fn_archive_all')


    obj_archive_mgt.fn_save_model = fn_save_model
    obj_archive_mgt.fn_load_model = fn_load_model
    obj_archive_mgt.fn_archive_all = fn_archive_all


    return obj_archive_mgt
