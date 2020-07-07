import inspect
import os
import shutil
from pickle import Pickler, Unpickler


class Persister():
    LOG_FOLDER = 'logs/'
    MODEL_FOLDER= 'workspace/'
    TEMP_FOLDER= 'temp'
    MODEL_FILE_NAME= 'model.tar'
    TEMP_FILE_NAME='temp.pth.tar'
    PARAMS_FILE_NAME='params.tar'

    SAMPLE_BATCH_FILE_NAME = 'example.tar'

    def __init__(self, cwd):
        self.cwd = cwd
        self.log_root = self.session_folder =os.path.join(cwd, Persister.LOG_FOLDER)
        self.model_folder =os.path.join(cwd, Persister.MODEL_FOLDER)
        self.temp_folder = os.path.join(cwd, Persister.TEMP_FOLDER)
        self.example_file_path = os.path.join(self.model_folder, Persister.SAMPLE_BATCH_FILE_NAME)
        self.__session_num = 0
        self.model_file_path = os.path.join(self.model_folder , Persister.MODEL_FILE_NAME)
        self.samples_file_path = os.path.join(os.path.join(self.cwd, Persister.MODEL_FOLDER), Persister.SAMPLE_BATCH_FILE_NAME)
        self.src_config_file_path = os.path.join(self.cwd, 'ConfigParams.py')
        self.params_file_path = os.path.join(self.model_folder , Persister.PARAMS_FILE_NAME)

    def fn_xfer_model_using_file(self, from_net, to_net):
        try:
            from_net.save_the_model(folder=self.temp_folder, filename=Persister.TEMP_FILE_NAME)
            to_net.load_checkpoint(folder=self.temp_folder, filename=Persister.TEMP_FILE_NAME)
            return None
        except Exception as x:
            return('fn_xfer_model_using_file: ' + x)

    def fn_save_model(self, neural_net):
        try:
            neural_net.save_the_model(folder=self.model_folder, filename=Persister.MODEL_FILE_NAME)
            return None
        except Exception as x:
            fn_name = inspect.getframeinfo(inspect.currentframe()).function
            return fn_name + "; " + str(x)

    def fn_model_exists(self):
        model_file_path = os.path.join(self.model_folder, Persister.MODEL_FILE_NAME)
        if os.path.exists(model_file_path):
            return True
        else:
            return  False

    def fn_load_model(self, neural_net):
        fn_name = inspect.getframeinfo(inspect.currentframe()).function
        try:
            file_path = os.path.join(self.model_folder, Persister.MODEL_FILE_NAME)
            if os.path.exists(file_path):
                neural_net.load_checkpoint(folder=self.model_folder, filename=Persister.MODEL_FILE_NAME)
                return None
            else:
                return fn_name + "; " + f"No File {file_path} available for Loading"
        except Exception as x:
            return fn_name + "; " + str(x)


    def fn_save_samples(self, samples_buffer):
        flag = "wb"
        if os.path.exists(self.example_file_path):
            flag = "wb+"
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)
        with open(self.example_file_path, flag) as f:
            Pickler(f).dump(samples_buffer)

    def fn_load_samples(self):
        samples_buffer = None
        fn_name = inspect.getframeinfo(inspect.currentframe()).function
        if not os.path.exists(self.example_file_path):
            return fn_name + "; " + f"No File {self.example_file_path} available for Loading", samples_buffer
        try:
            with open(self.example_file_path, "rb") as f:
                samples_buffer = Unpickler(f).load()

            return None, samples_buffer
        except Exception as x:
            return fn_name + "; " + f"Error loading {self.example_file_path}", samples_buffer


    def fn_save_params(self, params):
        flag = "wb"
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)
        with open(self.params_file_path, flag) as f:
            Pickler(f).dump(params)

    def fn_load_params(self):
        params = None
        fn_name = inspect.getframeinfo(inspect.currentframe()).function
        if not os.path.exists(self.params_file_path):
            return fn_name + "; " + f"No File {self.params_file_path} available for Loading", params
        try:
            with open(self.params_file_path, "rb") as f:
                params = Unpickler(f).load()

            return None, params
        except Exception as x:
            return fn_name + "; " + f"Error loading {self.params_file_path}", params

    def fn_cleanup(self):
        self.__fn_reset_model(retain_model = False, retain_examples = False)


        if os.path.exists(self.log_root):
            dirList = os.listdir(self.log_root)   # remove all session sub directories
            for dir in dirList:
                if os.path.isdir(dir):
                    shutil.rmtree(dir)
        else:
            os.mkdir(self.log_root)

        shutil.copy(self.src_config_file_path, self.log_root)


    def __fn_reset_model(self, retain_model= True, retain_examples=True):
        if not retain_model:
            if os.path.exists(self.model_folder):
                shutil.rmtree(self.model_folder)
            os.mkdir(self.model_folder)

        if not retain_examples:
            if os.path.exists(self.temp_folder):
                shutil.rmtree(self.temp_folder)
            os.mkdir(self.temp_folder)

    def __fn_increment_session(self):
        self.__session_num +=1

    def fn_create_new_session(self):
        self.__fn_reset_model()
        if not os.path.exists(self.log_root):
            os.mkdir(self.log_root)

        self.__fn_increment_session()

        session_id = self.fn_get_new_session_id()
        self.session_folder = os.path.join(self.log_root, session_id)

        if os.path.exists(self.session_folder):
            shutil.rmtree(self.session_folder)
        os.mkdir(self.session_folder)

        return session_id

    def fn_get_new_session_id(self):

        return f'session_{self.__session_num}'