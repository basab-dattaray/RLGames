class Recorder():
    def __init__(self, fn_record):
        self.fn_record = fn_record

    def fn_record_func_title(self, fn_name):
        self.fn_record()
        self.fn_record(f'<<<<<< {fn_name} >>>>>>')