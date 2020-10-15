from ws.RLInterfaces.PARAM_KEY_NAMES import STRATEGY
from ws.RLAgents.model_free.function_approximation.dqn.agent_mgt import agent_mgt
from ws.RLUtils.setup.preparation_mgt import preparation_mgt
# from ws.RLUtils.common.module_loader import load_function

import os

if __name__ == "__main__":
    app_info, env = preparation_mgt(__file__)
    subpackage_name = 'ws.RLAgents.{}'.format(app_info[STRATEGY])

    # agent_mgt = load_function(function_name="agent_mgt", module_tag="agent_mgt", subpackage_tag=subpackage_name)

    fnTrain, fnSaveWeights, fnLoadWeights = agent_mgt(app_info, env)

    cwd = __file__.rsplit('/', 1)[0]
    model_dir = os.path.join(cwd, "models")
    fnLoadWeights(model_dir)
    fnTrain()

    fnSaveWeights(model_dir)
    env.fnClose()

