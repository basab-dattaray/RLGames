
from collections import deque

def subplot_mgr(plt, axis,  graph_color):

    _y_vals = deque([])
    _x_indices = deque([])

    def fn_plot_line( x_index, y_val):

        _y_vals.append(y_val)
        _x_indices.append(x_index)

        line, = axis.plot(_x_indices, _y_vals, color=graph_color)

    def fn_plot_for_specific_axis(x_index, y_val):

        if x_index:
            fn_plot_line(x_index, y_val)

            plt.pause(0.000001)

    return fn_plot_for_specific_axis