from collections import namedtuple

from ws.RLUtils.monitoring.tracing.tracer import tracer


def agent_container(fn_recorder):
    agent_container_ref = namedtuple('_', ['fn_test1','fn_test2'])

    @tracer(fn_recorder)
    def fn_test1():
        print('RUNNING fn_test1')
        return agent_container_ref

    @tracer(fn_recorder)
    def fn_test2():
        print('RUNNING fn_test2')
        return agent_container_ref

    agent_container_ref.fn_test1 = fn_test1
    agent_container_ref.fn_test2 = fn_test2

    return agent_container_ref