from ws.RLUtils.monitoring.tracing.trace_example.agent_container import agent_container
from ws.RLUtils.monitoring.tracing.trace_example.record_mgt import record_mgt

fn_loger = record_mgt()

app_info = {}
app_info.rec_mgt = fn_loger

agent_container(app_info).fn_test1().fn_test2()


