import shutil

def archive_mgt(fn_save_to_neural_net, results_path, archive_path):


    def fn_archive(result_folder_path= None, archive_folder_path= None):
        try:
            if archive_folder_path is None:
                archive_folder_path = archive_path

            if archive_folder_path is None:
                return "FAILED: no archive path"

            if result_folder_path is None:
                result_folder_path = results_path

            if result_folder_path is None:
                return "FAILED: no result path"

            if fn_save_to_neural_net is not None:
                fn_save_to_neural_net(result_folder_path)

            shutil.copytree(result_folder_path, archive_folder_path, symlinks=False, ignore=None)

            return "INFO:: Sucessfully Archived at {}".format(archive_path)

        except Exception as x:
            print(x)
            raise Exception('Exception: fn_archive')

    return fn_archive
