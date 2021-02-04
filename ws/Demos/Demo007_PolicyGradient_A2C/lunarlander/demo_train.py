from ws.RLUtils.setup.agent_dispatcher import agent_dispatcher

def fn_exec_test():
    agent_mgr = agent_dispatcher(__file__)
    agent_mgr. \
        fn_run_train()

if __name__ == "__main__":
    fn_exec_test()
