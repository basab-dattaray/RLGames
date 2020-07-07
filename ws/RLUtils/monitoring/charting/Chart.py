from ws.RLUtils.monitoring.charting.Compactor import Compactor
from ws.RLUtils.monitoring.graphing.Graph import Graph
from ws.RLUtils.monitoring.graphing.data_compaction.datastream_mgt import datastream_mgr


class Chart():

    def __init__(self,
             save_plot_at_file_path, title_prefix, fn_title_update_callback, x_config_item, y_config_list,
             average_interval= 1, skip_interval= 1
        ):

        self.graph = Graph(
            save_plot_at_file_path, title_prefix, fn_title_update_callback,
            x_config_item, y_config_list
        )

        self.average_interval = average_interval
        self.skip_interval = skip_interval
        self.fn_compress_datastream = datastream_mgr(self.graph.fn_graph_event,
                                                     average_interval=self.average_interval,
                                                     skip_interval=self.skip_interval)

        self.compactor = Compactor(20, self.average_interval, self.skip_interval)
        # self.compactor = Compactor(20, self.log_interval, 1)

    def fn_record_event(self, index, array_of_y_vals):

        self.fn_compress_datastream(
            index,
            array_of_y_vals
        )

        # ref = self.compactor.fn_gen_skip_filter(index= index, y_vals= array_of_y_vals)
        # new_stuff = next(ref)

        pass



    def fn_close(self):
        self.graph.fn_graph_complete()

    def fn_update_title(self, progress_info):
        self.graph.fn_update_graph_title(progress_info)

