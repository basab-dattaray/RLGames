import os
import pickle


def pickle_mgt(folder, name):
    _full_filepath = os.path.join(folder, name)

    def fn_save(obj):
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(_full_filepath, 'wb') as f:
            pickle.dump(obj, f)
        return True


    def fn_load():
        with open(_full_filepath, 'rb') as f:
            obj = pickle.load(f)
        return obj

    return fn_save, fn_load


