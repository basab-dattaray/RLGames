from collections import namedtuple

from ws.RLUtils.monitoring.tracing.ex2.wrapper_maker import trace


def agent_container(recorder):
    agent_container_ref = namedtuple('_', ['fn_test1','fn_test2'])

    @trace(recorder)
    def fn_test1():
        print('RUNNING fn_test1')
        return agent_container_ref

    @trace(recorder)
    def fn_test2():
        print('RUNNING fn_test2')
        return agent_container_ref

    agent_container_ref.fn_test1 = fn_test1
    agent_container_ref.fn_test2 = fn_test2

    return agent_container_ref