import os
import sys
from pickle import Pickler, Unpickler


def fn_getCheckpointFile(iteration):
    return '_iter_' + str(iteration) + '.tar'


def fn_save_train_examples(app_info, iteration, training_samples_buffer):
    folder = app_info.RESULTS_REL_PATH
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = os.path.join(folder, fn_getCheckpointFile(iteration) + ".examples")
    with open(filename, "wb+") as f:
        Pickler(f).dump(training_samples_buffer)


def fn_load_train_examples(app_info):
    modelFile = os.path.join(app_info.LOAD_FOLDER_FILE[0], app_info.LOAD_FOLDER_FILE[1])
    examplesFile = modelFile + ".examples"
    if not os.path.isfile(examplesFile):
        # app_info.LOGGER_.warning(f'File "{examplesFile}" with training_samples not found!')
        r = input("Continue? [y|size]")
        if r != "y":
            sys.exit()
    else:
        app_info.fn_log("File with training_samples found. Loading it...")
        with open(examplesFile, "rb") as f:
            training_samples_buffer = Unpickler(f).load()
        app_info.fn_log('Loading done!')



def fn_log_iteration_results(app_info, draws, iteration, nwins, pwins):
    update_threshold = 'update threshold: {}'.format(app_info.PASSING_SCORE)
    score = f'nwins:{nwins} pwins:{pwins} draws:{draws} {update_threshold}'
    app_info.fn_log(score)