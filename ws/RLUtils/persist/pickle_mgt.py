import os
import pickle


def pickle_mgr(folder, name):
    _full_filepath = os.path.join(folder, name)

    def fn_save(obj):
        try:
            if not os.path.exists(folder):
                os.makedirs(folder)
            with open(_full_filepath, 'wb') as f:
                pickle.dump(obj, f)
            return True

        except Exception as x:
            return False

    def fn_load():
        try:
            with open(_full_filepath, 'rb') as f:
                obj = pickle.load(f)
            return obj

        except Exception as x:
            return None

    return fn_save, fn_load


