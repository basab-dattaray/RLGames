from collections import namedtuple

from ws.RLUtils.monitoring.tracing.trace_example_with_self_contained_record_mgr.record_mgt import record_mgr
from ws.RLUtils.monitoring.tracing.trace_example_with_self_contained_record_mgr.tracer import tracer


def agent_container():
    args = {}
    args['fn_recorder'] = record_mgr()
    agent_container_ref = namedtuple('_', ['fn_test1','fn_test2'])

    @tracer(args)
    def fn_test1():
        print('RUNNING fn_test1')
        return agent_container_ref

    @tracer(args)
    def fn_test2():
        print('RUNNING fn_test2')
        return agent_container_ref

    agent_container_ref.fn_test1 = fn_test1
    agent_container_ref.fn_test2 = fn_test2

    return agent_container_ref