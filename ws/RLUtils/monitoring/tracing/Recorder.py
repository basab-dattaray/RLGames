class Recorder():

    indent_size = 2
    def __init__(self, fn_record):
        self.fn_record = fn_record
        self.indent_count = 0

    def fn_record_func_title_begin(self, fn_name):
        self.fn_record()
        prefix = self.indent_count * ' '
        self.fn_record(f'{prefix}<<<<<< {fn_name} >>>>>>')
        self.indent_count += Recorder.indent_size


    def fn_record_func_title_end(self):
        self.indent_count -= Recorder.indent_size

    def fn_record_message(self, message='', indent=1):
        prefix = (self.indent_count + indent * self.indent_size) * ' '
        self.fn_record(f'{prefix}{message}')
