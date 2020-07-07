
from ws.RLUtils.monitoring.graphing.data_compaction.compaction_mgt import compaction_mgr


def plugin_for_skipping_mgr():
    _median_xindex = None
    def fn_compute_yval(number_of_entries, strand_num, yvals_for_strands):
        computed_yval_strand = yvals_for_strands[strand_num][_median_xindex]
        return computed_yval_strand

    def fn_compute_xindex(x_index_list_for_pipe):

        x_index = x_index_list_for_pipe[_median_xindex]
        return x_index

    fn_compress_stream_data = compaction_mgr(fn_compute_xindex, fn_compute_yval)

    def fn_compress(x_index_list_for_pipe, y_vals_list_for_pipe):
        nonlocal _median_xindex
        _median_xindex = int(len(x_index_list_for_pipe) / 2)
        return fn_compress_stream_data(x_index_list_for_pipe, y_vals_list_for_pipe)

    return fn_compress
