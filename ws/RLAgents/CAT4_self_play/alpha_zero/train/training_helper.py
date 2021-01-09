import os
import sys
from pickle import Pickler, Unpickler


def fn_getCheckpointFile(iteration):
    return '_iter_' + str(iteration) + '.tar'


def fn_save_train_examples(args, iteration, training_samples_buffer):
    folder = args.REL_MODEL_PATH
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = os.path.join(folder, fn_getCheckpointFile(iteration) + ".examples")
    with open(filename, "wb+") as f:
        Pickler(f).dump(training_samples_buffer)


def fn_load_train_examples(args):
    modelFile = os.path.join(args.LOAD_FOLDER_FILE[0], args.LOAD_FOLDER_FILE[1])
    examplesFile = modelFile + ".examples"
    if not os.path.isfile(examplesFile):
        args.LOGGER_.warning(f'File "{examplesFile}" with training_samples not found!')
        r = input("Continue? [y|size]")
        if r != "y":
            sys.exit()
    else:
        args.fn_record("File with training_samples found. Loading it...")
        with open(examplesFile, "rb") as f:
            training_samples_buffer = Unpickler(f).load()
        args.fn_record('Loading done!')



def fn_log_iteration_results(args, draws, iteration, nwins, pwins):
    update_threshold = 'update threshold: {}'.format(args.SCORE_BASED_MODEL_UPDATE_THRESHOLD)
    score = f'nwins:{nwins} pwins:{pwins} draws:{draws} {update_threshold}'
    args.calltracer.fn_write(score)