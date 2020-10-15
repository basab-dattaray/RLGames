from ws.RLUtils.monitoring.graphing.plot_mgt import plot_mgt


class Graph():
    
    def __init__(self, save_plot_at_file_path, title_prefix, fn_title_update_callback, x_config_item, y_config_list):
        self.no_graphing = False

        self.save_plot_at_file_path = save_plot_at_file_path
    
        self.artificial_index = -1
        self.fn_title_update_callback = fn_title_update_callback

        self.fn_plot, self.fn_save_as_pdf, self.fn_set_the_title = plot_mgt(
            title_prefix, x_config_item['axis_label'], y_config_list)

    def fn_supress_graphing(self):

        self.no_graphing = True

    def fn_graph_event(self, index, y_vals):

        if self.no_graphing: return

        self.artificial_index += 1

        x_index = index
        if x_index is None:
            x_index = self.artificial_index
        self.fn_plot(x_index, y_vals)

    def fn_graph_complete(self):
        if self.no_graphing: return

        self.fn_save_as_pdf(self.save_plot_at_file_path)


    def fn_update_graph_title(self, progress_info):
        title = self.fn_title_update_callback(progress_info)
        self.fn_set_the_title(title)

