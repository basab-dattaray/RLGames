import signal
from collections import namedtuple

from time import sleep

# from ws.RLInterfaces.PARAM_KEY_NAMES import *
from ws.RLAgents.CAT3_policy_gradient_based.progress_mgt import progress_mgt
from ws.RLUtils.common.config_mgt import config_mgt

from ws.RLUtils.common.module_loader import load_function
from ws.RLUtils.modelling.archive_mgt import archive_mgt
# from ws.RLUtils.setup.preparation_mgt import preparation_mgt
from ws.RLUtils.setup.startup_mgt import startup_mgt


def agent_mgt(caller_file):
    # app_info = preparation_mgt(caller_file)
    app_info = startup_mgt(caller_file)
    fn_get_key_as_bool, fn_get_key_as_int, _ = config_mgt(app_info)
    is_single_episode_result = fn_get_key_as_bool('REWARD_CALCULATED_FROM_SINGLE_EPISODES')
    env = app_info.ENV
    impl_mgt = load_function('impl_mgt', 'impl_mgt', app_info.AGENT_FOLDER_PATH)

    fn_act, fn_add_transition, fn_save_to_neural_net, fn_load_from_neural_net, fn_should_update_network = impl_mgt(app_info)

    refobj_archive_mgt = archive_mgt(
        fn_save_to_neural_net,
        fn_load_from_neural_net,
        app_info.RESULTS_ARCHIVE_PATH,
        app_info.RESULTS_BASE_PATH,
        app_info.MAX_RESULT_COUNT
    )

    chart, fn_show_training_progress, fn_has_reached_goal = progress_mgt(app_info)

    _episode_num = 0

    fn_log = app_info.FN_RECORD

    def exit_gracefully(signum, frame):
        fn_log('!!! TERMINATING EARLY!!!')
        msg = refobj_archive_mgt.fn_archive_all(app_info, fn_save_model=refobj_archive_mgt.fn_save_model)

        chart.fn_close()
        fn_log(msg)
        env.fn_close()
        exit()

    def fn_run(fn_show_training_progress,
               supress_graph=False,
               fn_should_update_network=fn_should_update_network,
               consecutive_goal_hits_needed_for_success=None
               ):
        nonlocal _episode_num

        # if supress_graph:
        #     fn_supress_graphing()

        signal.signal(signal.SIGINT, exit_gracefully)
        _episode_num = 1
        done = False
        while _episode_num <= app_info['NUM_EPISODES'] and not done:
            running_reward, num_steps = fn_run_episode(fn_should_update_network=fn_should_update_network)
            fn_show_training_progress(_episode_num, running_reward, num_steps)

            done = fn_has_reached_goal(running_reward, consecutive_goal_hits_needed_for_success)
            _episode_num += 1
        chart.fn_close()

    def fn_run_episode(fn_should_update_network=None, do_render=False):

        state = env.fn_reset_env()
        running_reward = 0
        reward = 0
        step = 0
        done = False
        while step < app_info['MAX_STEPS_PER_EPISODE'] and not done:
            step += 1

            if do_render:
                env.fn_render()
                sleep(.01)

            action = fn_act(state)
            state, reward, done, _ = env.fn_take_step(action)

            fn_add_transition(reward, done)

            if fn_should_update_network is not None:
                fn_should_update_network(done)

            running_reward += reward

        env.fn_close()

        val = reward if is_single_episode_result else running_reward
        return val, step

    def fn_run_train():
        if refobj_archive_mgt.fn_load_model is not None:
            if refobj_archive_mgt.fn_load_model():
                fn_log('SUCCESS in loading model')
            else:
                fn_log('FAILED in loading model')

        fn_run(fn_show_training_progress, fn_should_update_network=fn_should_update_network)

        fn_log(refobj_archive_mgt.fn_archive_all(app_info, refobj_archive_mgt.fn_save_model))
        return agent_mgr


    def fn_run_test():
        if refobj_archive_mgt.fn_load_model is None:
            fn_log("ERROR:: Loading Model: model function missing")
            return agent_mgr

        if not refobj_archive_mgt.fn_load_model():
            fn_log("ERROR:: Loading Model: model agent_configs missing")
            return agent_mgr

        fn_run(fn_show_training_progress, supress_graph=True, consecutive_goal_hits_needed_for_success=1)
        fn_run_episode(do_render=True)
        return agent_mgr

    agent_mgr = namedtuple('_',
                                [
                                    'fn_run_train',
                                    'fn_run_test'
                                ]
                           )
    agent_mgr.fn_run_train = fn_run_train
    agent_mgr.fn_run_test = fn_run_test

    return agent_mgr

