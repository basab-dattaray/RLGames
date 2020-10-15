from ws.RLUtils.monitoring.graphing.data_compaction.pipe_mgt import pipe_mgt
from ws.RLUtils.monitoring.graphing.data_compaction.plugin_for_averaging_mgt import plugin_for_averaging_mgt
from ws.RLUtils.monitoring.graphing.data_compaction.plugin_for_skipping_mgt import plugin_for_skipping_mgt


def datastream_mgt(fn_graph_event, average_interval=1, skip_interval=1):
    fn_compress_by_averaging = plugin_for_averaging_mgt()
    fn_compress_by_skip = plugin_for_skipping_mgt()
    _proxy_index = 0

    def fn_compress_datastream(x_index, y_vals):
        nonlocal _proxy_index

        if x_index is None:
            x_index = _proxy_index
            _proxy_index += 1

        fn_process_pipe1(x_index, y_vals)

    fn_process_pipe2 = pipe_mgt(average_interval, fn_compress_by_skip, fn_graph_event)
    fn_process_pipe1 = pipe_mgt(skip_interval, fn_compress_by_averaging, fn_process_pipe2)

    return fn_compress_datastream
