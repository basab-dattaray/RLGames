import signal


def interrupt_mgt(app_info):
    def exit_gracefully(signum, frame):
        app_info.fn_log('!!! TERMINATING EARLY!!!')
        archive_msg = app_info.fn_archive(archive_folder_path= app_info.FULL_ARCHIVE_PATH_,  fn_save_to_neural_net= None)
        app_info.fn_log(archive_msg)

        # app_info.ENV.fn_close()
        exit()

    signal.signal(signal.SIGINT, exit_gracefully)