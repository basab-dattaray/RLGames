def pipe_mgt(num_items, fn_pipe_compute, fn_next_in_chain):
    _y_vals_list = []
    _x_index_list = []
    _count = 0

    def fn_process_pipe(x_index, y_vals):
        nonlocal _count

        if _count <= num_items:
            _x_index_list.append(x_index)
            _y_vals_list.append(y_vals)
            _count += 1
        if _count == num_items:
            xindex_pipe, yvals_pipe = fn_pipe_compute(_x_index_list, _y_vals_list)

            # Reset pipe
            _count = 0
            _x_index_list.clear()
            _y_vals_list.clear()

            fn_next_in_chain(xindex_pipe, yvals_pipe)

    return fn_process_pipe
