import matplotlib.pyplot as plt
import datetime as dt

from ws.RLUtils.monitoring.graphing.subplot_mgt import subplot_mgt

def plot_mgt(title_prefix, x_axis_label, y_info_list,subplot_width = 6.4, subplot_height = 4.8):
    num_axes = len(y_info_list)

    plt.figure(figsize=(subplot_width, subplot_height * num_axes))
    plt.rcParams['axes.titlesize'] = 8
    _axes = [None] * num_axes

    subplot_base_num = 100 * num_axes
    fn_plot_for_specific_axis_list = []
    for i in range(num_axes):
        axes = None
        subplot_num = subplot_base_num + 11 + i
        if i == 0:
            axes = plt.subplot(subplot_num)
        else:
            axes = plt.subplot(subplot_num, sharex=_axes[0])

        graph_color = y_info_list[i]['color_black_background']
        axes.cla()
        axes.grid()
        axes.set_xlabel(x_axis_label)
        axes.set_ylabel(y_info_list[i]['axis_label'], color= graph_color)
        _axes[i] = axes

        fn_plot_for_specific_axis = subplot_mgt(plt, _axes[i], graph_color)
        fn_plot_for_specific_axis_list.append(fn_plot_for_specific_axis)

    _axes[0].set_title(title_prefix)

    def fn_set_the_title(title):
        _axes[0].set_title(title_prefix + title)

    def fn_plot(x_index, y_vals):

        for index in range(len(y_vals)):
            fn_plot_for_specific_axis_list[index](x_index, y_vals[index])

    def fn_save_as_pdf(save_path):
        plt.savefig(save_path)
        plt.close('all')
        pass


    return fn_plot, fn_save_as_pdf, fn_set_the_title