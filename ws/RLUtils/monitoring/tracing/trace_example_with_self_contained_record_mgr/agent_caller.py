# from ws.RLUtils.monitoring.tracing.trace_example.agent_container import agent_container
# from ws.RLUtils.monitoring.tracing.trace_example.record_mgt import record_mgt
from ws.RLUtils.monitoring.tracing.trace_example_with_self_contained_record_mgt.agent_container import agent_container
from ws.RLUtils.monitoring.tracing.trace_example_with_self_contained_record_mgt.record_mgt import record_mgt



agent_container().fn_test1().fn_test2().fn_test1()


