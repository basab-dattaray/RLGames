from ws.RLUtils.monitoring.graphing.data_compaction.pipe_mgt import pipe_mgr
from ws.RLUtils.monitoring.graphing.data_compaction.plugin_for_averaging_mgt import plugin_for_averaging_mgr
from ws.RLUtils.monitoring.graphing.data_compaction.plugin_for_skipping_mgt import plugin_for_skipping_mgr


def datastream_mgr(fn_graph_event, average_interval=1, skip_interval=1):
    fn_compress_by_averaging = plugin_for_averaging_mgr()
    fn_compress_by_skip = plugin_for_skipping_mgr()
    _proxy_index = 0

    def fn_compress_datastream(x_index, y_vals):
        nonlocal _proxy_index

        if x_index is None:
            x_index = _proxy_index
            _proxy_index += 1

        fn_process_pipe1(x_index, y_vals)

    fn_process_pipe2 = pipe_mgr(average_interval, fn_compress_by_skip, fn_graph_event)
    fn_process_pipe1 = pipe_mgr(skip_interval, fn_compress_by_averaging, fn_process_pipe2)

    return fn_compress_datastream
