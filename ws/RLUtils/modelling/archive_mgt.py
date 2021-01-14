# import os
import shutil
from collections import namedtuple


def archive_mgt(fn_save_to_neural_net,  model_folder_path):

    # def fn_save_archive_model():
    #     fn_save_to_neural_net(model_folder_path)
    #     return model_folder_path

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
            raise Exception('Exception: fn_archive_all')

    # obj_archive_mgt.fn_save_archive_model = fn_save_archive_model
    # obj_archive_mgt.fn_archive_all = fn_archive_all


    return fn_archive
