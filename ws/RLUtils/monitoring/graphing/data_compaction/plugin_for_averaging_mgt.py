from functools import reduce

from ws.RLUtils.monitoring.graphing.data_compaction.compaction_mgt import compaction_mgt


def plugin_for_averaging_mgt():
    _median_xindex = None
    def fn_compute_yval(number_of_entries, strand_num, yvals_for_strands):
        total_of_yval_strand = reduce((lambda x, y: x + y), yvals_for_strands[strand_num])
        average_of_yval_strand = total_of_yval_strand / number_of_entries
        return average_of_yval_strand

    def fn_compute_xindex(x_index_list_for_pipe):

        x_index = x_index_list_for_pipe[_median_xindex]
        return x_index

    fn_compress_stream_data = compaction_mgt(fn_compute_xindex, fn_compute_yval)

    def fn_compress(x_index_list_for_pipe, y_vals_list_for_pipe):
        nonlocal _median_xindex
        _median_xindex = int(len(x_index_list_for_pipe) / 2)
        return fn_compress_stream_data(x_index_list_for_pipe, y_vals_list_for_pipe)

    return fn_compress
