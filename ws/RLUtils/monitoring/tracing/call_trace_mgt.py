from collections import namedtuple


def call_trace_mgt(fn_log):

    DEFAULT_INDENT = 2

    indent_count = 0

    def fn_log_func_title_begin(fn_name):
        nonlocal indent_count
        fn_log()
        prefix = indent_count * ' '
        fn_log(f'{prefix}<<<<<< {fn_name} >>>>>>')
        indent_count += DEFAULT_INDENT


    def fn_log_func_title_end():
        nonlocal indent_count
        indent_count -= DEFAULT_INDENT

    def fn_log_message(message='', indent=1):
        prefix = (indent_count + indent * DEFAULT_INDENT) * ' '
        fn_log(f'{prefix}{message}')

    call_trace_mgr = namedtuple('_', [
        'fn_log_func_title_begin',
        'fn_log_func_title_end',
        'fn_log_message',

    ])

    call_trace_mgr.fn_log_func_title_begin = fn_log_func_title_begin
    call_trace_mgr.fn_log_func_title_end = fn_log_func_title_end
    call_trace_mgr.fn_log_message = fn_log_message

    return call_trace_mgr