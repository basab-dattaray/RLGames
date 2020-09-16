class Recorder():
    def __init__(self, fn_record):
        self.fn_record = fn_record
        self.indent = 0

    def fn_record_func_title_begin(self, fn_name):
        self.fn_record()
        prefix = self.indent * ' '
        self.fn_record(f'{prefix}<<<<<< {fn_name} >>>>>>')
        self.indent += 2


    def fn_record_func_title_end(self):
        self.fn_record()
        self.indent -= 2

    def fn_record_message(self, message):
        self.fn_record()
        prefix = (self.indent + 2) * ' '
        self.fn_record(f'{prefix}{message}')
