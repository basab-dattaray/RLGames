from ws.RLUtils.monitoring.tracing.trace_example.agent_container import agent_container
from ws.RLUtils.monitoring.tracing.trace_example.record_mgt import record_mgr

fn_recorder = record_mgr()

args = {}
args['rec_mgr'] = fn_recorder

agent_container(args).fn_test1().fn_test2()


