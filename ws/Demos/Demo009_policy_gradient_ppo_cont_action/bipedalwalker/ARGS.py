from ws.RLUtils.common.DotDict import DotDict

app_info = DotDict(
    {
        "STRATEGY": "CAT3_policy_gradient_based.fn_approx_yes__proximal_policy_approx.continuous_action",

        "NUM_EPOCHS": 80,
        "NUM_EPISODES": 10000,

        "MAX_STEPS_PER_EPISODE": 4500,

        "GAMMA": 0.99,
        "LEARNING_RATE": 0.0003,
        "EPSILON": 0.1,
        "CLIPPING_LOSS_RATIO": 0.2,
        "MAX_GRADIENT_NORM": 0.5,

        "UPDATE_STEP_INTERVAL": 10,
        "LOG_SKIP_INTERVAL": 1,

        "BUFFER_CAPACITY": 1000,

        "ENV_NAME": "BipedalWalker-v2",
        "ENV_DISPLAY_ON": 0,

        "ACTOR_HIDDEN_LAYER_NODES": [
            {
                "LAYER_TYPE": "LINEAR",
                "NUM_NODES": 64,
                "ACTIVATION_FN": "tanh"
            },
            {
                "LAYER_TYPE": "LINEAR",
                "NUM_NODES": 32,
                "ACTIVATION_FN": "tanh"
            }
        ],

        "CRITIC_HIDDEN_LAYER_NODES": [
            {
                "LAYER_TYPE": "LINEAR",
                "NUM_NODES": 64,
                "ACTIVATION_FN": "tanh"
            },
            {
                "LAYER_TYPE": "LINEAR",
                "NUM_NODES": 32,
                "ACTIVATION_FN": "tanh"
            }
        ],


        "REWARD_GOAL": 20,
        "LOG_MEAN_INTERVAL": 20,
        "MAX_RESULT_COUNT": 3,

        "SD_FOR_MULTIVARIATE_NORMAL_ACTION_DISTRIBUTION": 0.5,

        "FORCE_CPU_USE": -1,

        "AUTO_ARCHIVE": 1,
        "AUTO_INTERRUPT_HANDLING": 1,
    }
)

def fn_get_args():
    return app_info