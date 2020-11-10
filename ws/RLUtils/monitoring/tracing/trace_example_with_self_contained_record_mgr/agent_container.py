from collections import namedtuple

from ws.RLUtils.monitoring.tracing.trace_example_with_self_contained_record_mgt.record_mgt import record_mgt
from ws.RLUtils.monitoring.tracing.trace_example_with_self_contained_record_mgt.tracer import tracer


def agent_container():
    args = {}
    args['fn_loger'] = record_mgt()


    # @tracer(args)
    def fn_test1():
        print('RUNNING fn_test1')
        return agent_container_ref

    fn_test1 = tracer(args)(fn_test1)

    # _fn_wrapper()

    @tracer(args)
    def fn_test2():
        print('RUNNING fn_test2')
        return agent_container_ref

    agent_container_ref = namedtuple('_', ['fn_test1','fn_test2'])
    agent_container_ref.fn_test1 = fn_test1
    agent_container_ref.fn_test2 = fn_test2

    return agent_container_ref