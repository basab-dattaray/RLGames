def compaction_mgt(fn_compute_xindex, fn_compute_yval):
    def fn_compress_stream_data(x_index_list_for_pipe, y_vals_list_for_pipe):
        number_of_entries = len(y_vals_list_for_pipe)
        number_of_strands = len(y_vals_list_for_pipe[0])

        yvals_for_strands = []
        for strand_num in range(number_of_strands):
            tmp_yvals = []
            yvals_for_strands.append(tmp_yvals)

        average_of_yvals_for_pipe = []
        for strand_num in range(number_of_strands):
            for entry_num in range(number_of_entries):
                yval_for_strand = y_vals_list_for_pipe[entry_num][strand_num]
                yvals_for_strands[strand_num].append(yval_for_strand)

            computed_yvals = fn_compute_yval(number_of_entries, strand_num, yvals_for_strands)
            average_of_yvals_for_pipe.append(computed_yvals)

        selected_xindex_for_pipe = fn_compute_xindex(x_index_list_for_pipe)
        # selected_xindex_for_pipe = x_index_list_for_pipe[x_index]

        return selected_xindex_for_pipe, average_of_yvals_for_pipe

    return fn_compress_stream_data
