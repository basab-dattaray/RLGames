# from ws.RLAgents.agent_configs.gridwell_1.AGENT_CONFIG import fn_add_configs
from ws.RLUtils.common.module_loader import load_function


def agent_config_mgt(app_info):
    agent_config_path = app_info['AGENTS_CONFIG_DOT_PATH'] + '.' + app_info['AGENT_CONFIG']
    fn_add_configs = load_function(function_name="fn_add_configs", module_tag="AGENT_CONFIG", subpackage_tag=agent_config_path)
    fn_add_configs(app_info)
    pass
