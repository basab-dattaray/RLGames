
from collections import namedtuple
from time import sleep

from ws.RLAgents.D_PolicyGradientBased.progress_mgt import progress_mgt
from ws.RLUtils.common.attr_mgt import attr_mgt

from ws.RLUtils.common.module_loader import load_function, load_mgt_function


def agent_mgt(app_info, common_functions):
    # fn_run = impl_mgt(app_info)
    fn_should_update_network= None

    if app_info.ENV is not None:
        fn_get_key_as_bool, fn_get_key_as_int, _ = attr_mgt(app_info)
        is_single_episode_result = fn_get_key_as_bool('REWARD_CALCULATED_FROM_SINGLE_EPISODES')

        # impl_mgt = load_function(function_name= 'impl_mgt', module_name='impl_mgt', module_dot_path= app_info.AGENT_DOT_PATH)
        impl_mgt = load_mgt_function(loc_dotpath=app_info.AGENT_DOT_PATH, module_name='impl_mgt')

        fn_act, fn_add_transition, fn_save_model, fn_load_model, fn_should_update_network = impl_mgt(app_info)

        chart, fn_show_training_progress, fn_has_reached_goal = progress_mgt(app_info)

        _episode_num = 0

        fn_log = app_info.fn_log

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
            if 'TEST_MODE' in app_info:
                if app_info.TEST_MODE: # ONLY 1 episode needed
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
        if fn_load_model is not None:
            if fn_load_model():
                fn_log('SUCCESS in loading model')
            else:
                fn_log('FAILED in loading model')

        fn_run(fn_show_training_progress, fn_should_update_network=fn_should_update_network)
        archive_msg = app_info.fn_archive(fn_save_model= fn_save_model,)
        fn_log(archive_msg)
        return agent_mgr

    def fn_run_test():
        if app_info.ENV is None:
            return agent_mgr

        if fn_load_model is not None:
            if not fn_load_model():
                fn_log('FAILED in loading model')
                return agent_mgr

        fn_run(fn_show_training_progress, supress_graph=True, consecutive_goal_hits_needed_for_success=1)
        fn_run_episode(do_render=True)
        return agent_mgr

    agent_mgr = namedtuple('_',
                                [
                                    'fn_run_train',
                                    'fn_run_test'
                                    'fn_change_args',
                                    'APP_INFO',
                                ]
                           )
    agent_mgr.fn_run_train = fn_run_train
    agent_mgr.fn_run_test = fn_run_test
    agent_mgr.fn_change_args =  common_functions.fn_change_args
    agent_mgr.APP_INFO = app_info
    return agent_mgr

