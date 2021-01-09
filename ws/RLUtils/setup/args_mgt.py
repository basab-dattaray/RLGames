import logging
import os
from pathlib import Path

from ws.RLUtils.common.AppInfo import AppInfo
from ws.RLUtils.common.DotDict import DotDict
from ws.RLUtils.common.folder_paths import fn_get_rel_dot_folder_path
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
        args = _fn_init_arg_with_default_val(args, 'LOGGER_', logging.getLogger(__name__))
        args = _fn_init_arg_with_default_val(args, 'DEMO_FOLDER_', demo_folder)
        args = _fn_init_arg_with_default_val(args, 'DEMO_NAME_', demo_name)
        args = _fn_init_arg_with_default_val(args, 'MODEL_NAME_', 'model.tar')
        args = _fn_init_arg_with_default_val(args, 'REL_MODEL_PATH', 'model/')
        args = _fn_init_arg_with_default_val(args, 'temp_model_exchange_filename', '_tmp.tar')
        current_dir = file_path.rsplit('/', 1)[0]
        archive_dir = current_dir.replace('/Demos/', '/Archives/')
        args = _fn_init_arg_with_default_val(args, 'archive_dir', archive_dir)
        args = _fn_init_arg_with_default_val(args, 'fn_record',
                                             log_mgt(log_dir=archive_dir, fixed_log_file=True))
        args = _fn_init_arg_with_default_val(args, 'calltracer', call_trace_mgt(args.fn_record))
        src_model_folder = os.path.join(args.DEMO_FOLDER_, args.REL_MODEL_PATH)
        args = _fn_init_arg_with_default_val(args, 'src_model_file_path',
                                             os.path.join(src_model_folder, args.MODEL_NAME_))
        args = _fn_init_arg_with_default_val(args, 'old_model_file_path',
                                             os.path.join(src_model_folder, 'old_' + args.MODEL_NAME_))
        args = _fn_init_arg_with_default_val(args, 'REL_MODEL_PATH', 'model/')

        args = _fn_init_arg_with_default_val(args, 'DO_LOAD_MODEL', True)
        args = _fn_init_arg_with_default_val(args, 'DO_LOAD_SAMPLES', False)

        args = _fn_init_arg_with_default_val(args, 'NUM_SUCCESSES_FOR_MODEL_UPGRADE_', 1)
        # args = _fn_init_arg_with_default_val(args, 'run_recursive_search', True)

        args = _fn_init_arg_with_default_val(args, 'UCB_USE_LOG_IN_NUMERATOR', True)
        args = _fn_init_arg_with_default_val(args, 'UCB_USE_POLICY_FOR_EXPLORATION', True)
        return args

    demo_folder, demo_name = AppInfo.fn_get_path_and_app_name(file_path)

    demo_dot_path = fn_get_rel_dot_folder_path(demo_folder, '/ws/')

    fn_get_args = load_function(function_name="fn_get_args", module_tag="ARGS", subpackage_tag=demo_dot_path)
    args = fn_get_args()

    args_copy = _fn_arg_defaults(args, demo_folder, demo_name, file_path)

    return args_copy


