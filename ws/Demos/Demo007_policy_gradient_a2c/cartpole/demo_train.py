from ws.RLAgents.CAT3_policy_gradient_based.agent_mgt import agent_mgt

if __name__ == '__main__':
    fn_run_train, fn_run_test = agent_mgt(__file__)
    fn_run_train()


