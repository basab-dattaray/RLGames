import signal


def interrupt_mgt(app_info):
    def exit_gracefully(signum, frame):
        app_info.fn_log('!!! TERMINATING EARLY!!!')
        archive_msg = app_info.fn_archive()
        app_info.fn_log(archive_msg)

        # app_info.ENV.fn_close()
        exit()

    signal.signal(signal.SIGINT, exit_gracefully)