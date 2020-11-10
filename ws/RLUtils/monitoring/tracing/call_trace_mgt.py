from collections import namedtuple


def call_trace_mgt(fn_log):

    DEFAULT_INDENT = 2

    indent_count = 0

    def fn_enter_function(fn_name):
        nonlocal indent_count
        fn_log()
        prefix = indent_count * ' '
        fn_log(f'{prefix}<<<<<< {fn_name} >>>>>>')
        indent_count += DEFAULT_INDENT


    def fn_leave_function():
        nonlocal indent_count
        indent_count -= DEFAULT_INDENT

    def fn_write(message='', indent=1):
        prefix = (indent_count + indent * DEFAULT_INDENT) * ' '
        fn_log(f'{prefix}{message}')

    call_trace_mgr = namedtuple('_', [
        'fn_enter_function',
        'fn_leave_function',
        'fn_write',

    ])

    call_trace_mgr.fn_enter_function = fn_enter_function
    call_trace_mgr.fn_leave_function = fn_leave_function
    call_trace_mgr.fn_write = fn_write

    return call_trace_mgr