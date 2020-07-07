from ws.RLInterfaces.PARAM_KEY_NAMES import STRATEGY
from ws.RLAgents.model_free.function_approximation.dqn.agent_mgt import agent_mgr
from ws.RLUtils.setup.preparation_mgt import preparation_mgr
# from ws.RLUtils.common.module_loader import load_function

import os

if __name__ == "__main__":
    app_info, env = preparation_mgr(__file__)
    subpackage_name = 'ws.RLAgents.{}'.format(app_info[STRATEGY])

    # agent_mgr = load_function(function_name="agent_mgr", module_tag="agent_mgt", subpackage_tag=subpackage_name)

    fnTrain, fnSaveWeights, fnLoadWeights = agent_mgr(app_info, env)

    cwd = __file__.rsplit('/', 1)[0]
    model_dir = os.path.join(cwd, "models")
    fnLoadWeights(model_dir)
    fnTrain()

    fnSaveWeights(model_dir)
    env.fnClose()

