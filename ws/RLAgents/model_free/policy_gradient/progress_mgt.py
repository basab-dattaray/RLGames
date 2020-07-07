import os

from ws.RLInterfaces.PARAM_KEY_NAMES import NUM_EPISODES, LOG_MEAN_INTERVAL, LOG_SKIP_INTERVAL, RESULTS_CURRENT_PATH, \
    REWARD_GOAL, STRATEGY, ENV_NAME, CONSECUTIVE_GOAL_HITS, FN_RECORD
from ws.RLUtils.monitoring.charting.Chart import Chart
from ws.RLUtils.monitoring.graphing.Graph import Graph
from ws.RLUtils.monitoring.graphing.data_compaction.datastream_mgt import datastream_mgr
from ws.RLUtils.common.config_mgt import config_mgr



def progress_mgr(app_info):
    _, fn_get_key_as_int, _ = config_mgr(app_info)
    _consecutive_goal_hits_needed_for_success = fn_get_key_as_int(CONSECUTIVE_GOAL_HITS, default = 1)
    _consecutive_goal_hit_count = 0
    _plot_file_path = os.path.join(app_info[RESULTS_CURRENT_PATH], 'rewards_plot.pdf')
    _max_index = app_info[NUM_EPISODES]
    _log_interval = app_info[LOG_MEAN_INTERVAL]
    _plot_skip_interval = app_info[LOG_SKIP_INTERVAL]

    _x_config_item = {'axis_label': 'episodes'}
    _y_config_list = [{'axis_label': 'reward', 'color_black_background': 'green'}]
    _title_prefix = '{}:{}\n'.format(app_info[STRATEGY], app_info[ENV_NAME])

    _reward_goal = app_info[REWARD_GOAL]

    def fn_title_update_callback(progress_info):
        msg = 'Episode: {}/{}'.format(

            progress_info['episode_num'], progress_info['max_episode_num']
        )
        return msg

    _chart = Chart(
        _plot_file_path, _title_prefix,
        fn_title_update_callback,
        _x_config_item, _y_config_list,
        average_interval=_log_interval, skip_interval=_plot_skip_interval
    )

    fn_record = app_info[FN_RECORD]

    def print_it(episode_num, step_num, val):
        fn_record('SAMPLE GEN EPISODE {:8} \t Steps: {:6} \t Value: {:10.5f}  Goal: {:10.5f}'.
              format(episode_num, step_num, val, app_info[REWARD_GOAL]))

    def fn_show_training_progress(episode_num, val, step_num):

        print_it(episode_num, step_num, val)
        if episode_num % app_info[LOG_MEAN_INTERVAL] == 0:
            progress_info = {'max_episode_num': app_info[NUM_EPISODES], 'episode_num': episode_num}

            _chart.fn_record_event(
                episode_num,
                [val]
            )
            _chart.fn_update_title(progress_info)


    def fn_has_reached_goal(value, consecutive_goal_hits_needed_for_success):
        nonlocal _consecutive_goal_hit_count

        if consecutive_goal_hits_needed_for_success is None:
            consecutive_goal_hits_needed_for_success = _consecutive_goal_hits_needed_for_success

        hit_success = value >= _reward_goal

        if hit_success:
            _consecutive_goal_hit_count += 1
        else:
            _consecutive_goal_hit_count = 0

        done = _consecutive_goal_hit_count == consecutive_goal_hits_needed_for_success
        return done

    return _chart, fn_show_training_progress, fn_has_reached_goal
