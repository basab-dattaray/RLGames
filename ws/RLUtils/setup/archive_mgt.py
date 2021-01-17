import os
import shutil

def archive_mgt(fn_save_to_neural_net, results_path, archive_path, fn_log_reset= None):
    _before_instance = True

    def fn_archive(result_folder_path= None, archive_folder_path= None):
        nonlocal  _before_instance
        try:
            if archive_folder_path is None:
                archive_folder_path = archive_path

            if archive_folder_path is None:
                return "FAILED: no archive path"

            if result_folder_path is None:
                result_folder_path = results_path

            if result_folder_path is None:
                return "FAILED: no result path"

            if _before_instance:
                real_archive_path = os.path.join( archive_folder_path , 'BEFORE')
            else:
                real_archive_path = os.path.join( archive_folder_path , 'AFTER')

            if fn_save_to_neural_net is not None:
                fn_save_to_neural_net(result_folder_path)

            if os.path.exists(real_archive_path):
                shutil.rmtree(real_archive_path)
            shutil.copytree(result_folder_path, real_archive_path, symlinks=False, ignore=None)

            if _before_instance:
                _before_instance = False
                if fn_log_reset is not None:
                    fn_log_reset()


            return "INFO:: Sucessfully Archived at {}".format(archive_path)

        except Exception as x:
            print(x)
            raise Exception('Exception: fn_archive')

    fn_archive()

    return fn_archive
