from ws.RLUtils.monitoring.tracing.ex2.agent_container import agent_container

def record_mgr():
    count = 0
    def fn_recorder():
        nonlocal count
        count += 1
        return count
    return fn_recorder

fn_recorder = record_mgr()

x = agent_container(fn_recorder)

print(x())
