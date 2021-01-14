import shutil

def archive_mgt(fn_save_to_neural_net,  model_folder_path):

    def fn_archive(archive_path= None):
        try:
            if fn_save_to_neural_net is not None:
                fn_save_to_neural_net(model_folder_path)

            if archive_path is  None:
                return "FAILED: no archive path found"

            shutil.copytree(model_folder_path, archive_path, symlinks=False, ignore=None)

            return "INFO:: Sucessfully Archived at {}".format(archive_path)

        except Exception as x:
            print(x)
            raise Exception('Exception: fn_archive')

    return fn_archive
