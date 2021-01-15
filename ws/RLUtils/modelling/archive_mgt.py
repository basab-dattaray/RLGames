import shutil

def archive_mgt(fn_save_to_neural_net,  model_folder_path):

    def fn_archive(archive_path= None, result_path= None):
        try:
            if archive_path is None:
                return "FAILED: no archive path found"

            if result_path is None:
                result_path = model_folder_path

            if fn_save_to_neural_net is not None:
                fn_save_to_neural_net(result_path)

            shutil.copytree(model_folder_path, archive_path, symlinks=False, ignore=None)

            return "INFO:: Sucessfully Archived at {}".format(archive_path)

        except Exception as x:
            print(x)
            raise Exception('Exception: fn_archive')

    return fn_archive
