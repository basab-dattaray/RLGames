
from collections import namedtuple
from time import sleep

from ws.RLAgents.Category3_PolicyGradientBased.progress_mgt import progress_mgt
from ws.RLUtils.common.attr_mgt import attr_mgt

from ws.RLUtils.common.module_loader import load_function
from ws.RLUtils.monitoring.tracing.tracer import tracer
from ws.RLUtils.setup.startup_mgt import startup_mgt


def agent_mgt(demo_path):
    app_info = startup_mgt(demo_path, __file__)
    fn_should_update_network= None
    if app_info.ENV is not None:
        fn_get_key_as_bool, fn_get_key_as_int, _ = attr_mgt(app_info)
        is_single_episode_result = fn_get_key_as_bool('REWARD_CALCULATED_FROM_SINGLE_EPISODES')

        impl_mgt = load_function(function_name= 'impl_mgt', module_name='impl_mgt', module_dot_path= app_info.AGENT_FOLDER_PATH)

        fn_act, fn_add_transition, fn_save_to_neural_net, fn_load_from_neural_net, fn_should_update_network = impl_mgt(app_info)

        chart, fn_show_training_progress, fn_has_reached_goal = progress_mgt(app_info)

        _episode_num = 0

        fn_log = app_info.fn_log

        _test_mode = False

    @tracer(app_info, verboscity= 4)
    def fn_set_test_mode():
        nonlocal  _test_mode
        _test_mode = True

        # env.display_mgr.fn_set_test_mode()
    def fn_run(fn_show_training_progress,
               supress_graph=False,
               fn_should_update_network=fn_should_update_network,
               consecutive_goal_hits_needed_for_success=None
               ):
        nonlocal _episode_num

        _episode_num = 1
        done = False
        while _episode_num <= app_info.NUM_EPISODES and not done:
            running_reward, num_steps = fn_run_episode(fn_should_update_network=fn_should_update_network)
            fn_show_training_progress(_episode_num, running_reward, num_steps)

            done = fn_has_reached_goal(running_reward, consecutive_goal_hits_needed_for_success)
            _episode_num += 1
            if not _test_mode:
                break
        chart.fn_close()

    def fn_run_episode(fn_should_update_network=None, do_render=False):

        state = app_info.ENV.fn_reset_env()
        running_reward = 0
        reward = 0
        step = 0
        done = False
        while step < app_info.MAX_STEPS_PER_EPISODE and not done:
            step += 1

            if do_render:
                app_info.ENV.fn_render()
                sleep(.01)

            action = fn_act(state)
            state, reward, done, _ = app_info.ENV.fn_take_step(action)

            fn_add_transition(reward, done)

            if fn_should_update_network is not None:
                fn_should_update_network(done)

            running_reward += reward

        app_info.ENV.fn_close()

        val = reward if is_single_episode_result else running_reward
        return val, step

    def fn_run_train():
        if app_info.ENV is None:
            return agent_mgr
        if fn_load_from_neural_net is not None:
            if fn_load_from_neural_net(app_info.RESULTS_PATH_):
                fn_log('SUCCESS in loading model')
            else:
                fn_log('FAILED in loading model')

        fn_run(fn_show_training_progress, fn_should_update_network=fn_should_update_network)
        archive_msg = app_info.fn_archive(fn_save_to_neural_net= fn_save_to_neural_net,)
        fn_log(archive_msg)
        return agent_mgr

    def fn_run_test():
        if app_info.ENV is None:
            return agent_mgr

        if fn_load_from_neural_net is not None:
            if fn_load_from_neural_net(app_info.RESULTS_PATH_):
                # fn_log('SUCCESS in loading model')
                pass
            else:
                fn_log('FAILED in loading model')
                return agent_mgr

        fn_run(fn_show_training_progress, supress_graph=True, consecutive_goal_hits_needed_for_success=1)
        fn_run_episode(do_render=True)
        return agent_mgr

    agent_mgr = namedtuple('_',
                                [
                                    'fn_run_train',
                                    'fn_run_test'
                                    'fn_set_test_mode',
                                    'APP_INFO',
                                ]
                           )
    agent_mgr.fn_run_train = fn_run_train
    agent_mgr.fn_run_test = fn_run_test
    agent_mgr.fn_set_test_mode = fn_set_test_mode
    agent_mgr.APP_INFO = app_info
    return agent_mgr

