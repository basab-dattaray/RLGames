from collections import namedtuple

# from ws.RLUtils.monitoring.tracing.tracer import tracer
# from ws.RLUtils.monitoring.tracing.trace_example_without_params.agent_caller import record_mgr
from ws.RLUtils.monitoring.tracing.tracer import tracer

def record_mgr():
    count = 0
    def fn_recorder():
        nonlocal count
        count += 1
        return count
    return fn_recorder

def agent_container():
    agent_container_ref = namedtuple('_', ['fn_test1','fn_test2'])

    fn_recorder = record_mgr()

    @tracer()
    def fn_test1():
        print('RUNNING fn_test1')
        return agent_container_ref

    @tracer()
    def fn_test2():
        print('RUNNING fn_test2')
        return agent_container_ref

    agent_container_ref.fn_test1 = fn_test1
    agent_container_ref.fn_test2 = fn_test2

    return agent_container_ref