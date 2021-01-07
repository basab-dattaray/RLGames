import logging
import os

from ws.RLUtils.common.AppInfo import AppInfo
from ws.RLUtils.common.DotDict import DotDict
from ws.RLUtils.monitoring.tracing.call_trace_mgt import call_trace_mgt
from ws.RLUtils.monitoring.tracing.log_mgt import log_mgt


def args_mgt(args, file_path):
    def _fn_init_arg_with_default_val(args, name, val):
        args = DotDict(args.copy())
        if name not in args:
            args[name] = val
        return args

    def _fn_arg_defaults(args, demo_folder, demo_name, file_path):
        args = _fn_init_arg_with_default_val(args, 'logger', logging.getLogger(__name__))
        args = _fn_init_arg_with_default_val(args, 'demo_folder', demo_folder)
        args = _fn_init_arg_with_default_val(args, 'demo_name', demo_name)
        args = _fn_init_arg_with_default_val(args, 'model_name', 'model.tar')
        args = _fn_init_arg_with_default_val(args, 'rel_model_path', 'model/')
        args = _fn_init_arg_with_default_val(args, 'temp_model_exchange_filename', '_tmp.tar')
        current_dir = file_path.rsplit('/', 1)[0]
        archive_dir = current_dir.replace('/Demos/', '/Archives/')
        args = _fn_init_arg_with_default_val(args, 'archive_dir', archive_dir)
        args = _fn_init_arg_with_default_val(args, 'fn_record',
                                             log_mgt(log_dir=archive_dir, fixed_log_file=True))
        args = _fn_init_arg_with_default_val(args, 'calltracer', call_trace_mgt(args.fn_record))
        src_model_folder = os.path.join(args.demo_folder, args.rel_model_path)
        args = _fn_init_arg_with_default_val(args, 'src_model_file_path',
                                             os.path.join(src_model_folder, args.model_name))
        args = _fn_init_arg_with_default_val(args, 'old_model_file_path',
                                             os.path.join(src_model_folder, 'old_' + args.model_name))
        args = _fn_init_arg_with_default_val(args, 'rel_model_path', 'model/')

        args = _fn_init_arg_with_default_val(args, 'do_load_model', True)
        args = _fn_init_arg_with_default_val(args, 'do_load_samples', False)

        args = _fn_init_arg_with_default_val(args, 'num_of_successes_for_model_upgrade', 1)
        args = _fn_init_arg_with_default_val(args, 'run_recursive_search', True)

        args = _fn_init_arg_with_default_val(args, 'mcts_ucb_use_log_in_numerator', True)
        args = _fn_init_arg_with_default_val(args, 'mcts_ucb_use_action_prob_for_exploration', True)
        return args

    demo_folder, demo_name = AppInfo.fn_get_path_and_app_name(file_path)
    args_copy = _fn_arg_defaults(args, demo_folder, demo_name, file_path)

    return args_copy
