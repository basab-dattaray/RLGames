import logging
import os
from pathlib import Path

from ws.RLUtils.common.AppInfo import AppInfo
from ws.RLUtils.common.DotDict import DotDict
from ws.RLUtils.common.module_loader import load_function
from ws.RLUtils.monitoring.tracing.call_trace_mgt import call_trace_mgt
from ws.RLUtils.monitoring.tracing.log_mgt import log_mgt

def args_mgt(file_path= None):
    def _fn_init_arg_with_default_val(args, name, val):
        args = DotDict(args.copy())
        if name not in args:
            args[name] = val
        return args

    def _fn_arg_defaults(args, demo_folder, demo_name, file_path):
        args = _fn_init_arg_with_default_val(args, 'logger', logging.getLogger(__name__))
        args = _fn_init_arg_with_default_val(args, 'demo_folder', demo_folder)
        args = _fn_init_arg_with_default_val(args, 'demo_name', demo_name)
        args = _fn_init_arg_with_default_val(args, 'MODEL_NAME', 'model.tar')
        args = _fn_init_arg_with_default_val(args, 'REL_MODEL_PATH', 'model/')
        args = _fn_init_arg_with_default_val(args, 'temp_model_exchange_filename', '_tmp.tar')
        current_dir = file_path.rsplit('/', 1)[0]
        archive_dir = current_dir.replace('/Demos/', '/Archives/')
        args = _fn_init_arg_with_default_val(args, 'archive_dir', archive_dir)
        args = _fn_init_arg_with_default_val(args, 'fn_record',
                                             log_mgt(log_dir=archive_dir, fixed_log_file=True))
        args = _fn_init_arg_with_default_val(args, 'calltracer', call_trace_mgt(args.fn_record))
        src_model_folder = os.path.join(args.demo_folder, args.REL_MODEL_PATH)
        args = _fn_init_arg_with_default_val(args, 'src_model_file_path',
                                             os.path.join(src_model_folder, args.MODEL_NAME))
        args = _fn_init_arg_with_default_val(args, 'old_model_file_path',
                                             os.path.join(src_model_folder, 'old_' + args.MODEL_NAME))
        args = _fn_init_arg_with_default_val(args, 'REL_MODEL_PATH', 'model/')

        args = _fn_init_arg_with_default_val(args, 'DO_LOAD_MODEL', True)
        args = _fn_init_arg_with_default_val(args, 'DO_LOAD_SAMPLES', False)

        args = _fn_init_arg_with_default_val(args, 'num_of_successes_for_model_upgrade', 1)
        args = _fn_init_arg_with_default_val(args, 'run_recursive_search', True)

        args = _fn_init_arg_with_default_val(args, 'UCB_USE_LOG_IN_NUMERATOR', True)
        args = _fn_init_arg_with_default_val(args, 'UCB_USE_POLICY_FOR_EXPLORATION', True)
        return args

    demo_folder, demo_name = AppInfo.fn_get_path_and_app_name(file_path)

    FLAG = False
    if FLAG:
        base_folder = str(Path(demo_folder).parent.parent.parent.parent.parent.parent)

        relative_demo_path = demo_folder.replace(base_folder, '')
        demo_dot_path = relative_demo_path.replace('/', '.')[1:]
    else:
        index = demo_folder.find('/ws/')
        relative_demo_path = demo_folder[index:]
        demo_dot_path = relative_demo_path.replace('/', '.')[1:]

    fn_get_args = load_function(function_name="fn_get_args", module_tag="ARGS", subpackage_tag=demo_dot_path)
    args = fn_get_args()

    args_copy = _fn_arg_defaults(args, demo_folder, demo_name, file_path)

    return args_copy
