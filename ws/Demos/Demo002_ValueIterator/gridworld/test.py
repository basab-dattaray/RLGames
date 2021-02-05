from ws.RLUtils.setup.agent_dispatcher import agent_dispatcher

def fn_execute():
    agent_mgr = agent_dispatcher(__file__)
    agent_mgr. \
        fn_change_args(
            {
                'TEST_MODE': True,
            }
        ). \
        fn_setup_env(). \
        fn_run_env()
    return agent_mgr.APP_INFO.ERROR_MESSAGE_

if __name__ == "__main__":
    print(fn_execute())
