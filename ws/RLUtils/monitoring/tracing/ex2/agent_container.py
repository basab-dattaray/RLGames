from ws.RLUtils.monitoring.tracing.ex2.wrapper_maker import trace


def agent_container(recorder):
    @trace(recorder)
    def test2():
        return 'test1XML'
    return test2