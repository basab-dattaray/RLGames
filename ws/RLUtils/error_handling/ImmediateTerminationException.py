class ImmediateTerminationException(Exception):
    def __init__(self, *app_info, **kwargs):
        Exception.__init__(self, *app_info, **kwargs)
