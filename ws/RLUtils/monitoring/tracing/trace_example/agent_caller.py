from ws.RLUtils.monitoring.tracing.trace_example.agent_container import agent_container

def record_mgr():
    count = 0
    def fn_recorder():
        nonlocal count
        count += 1
        return count
    return fn_recorder

fn_recorder = record_mgr()

agent_container(fn_recorder).fn_test1().fn_test2()


