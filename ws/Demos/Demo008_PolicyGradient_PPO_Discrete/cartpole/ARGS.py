from ws.RLUtils.common.DotDict import DotDict

app_info = DotDict(
    {
        "STRATEGY": "Category3_PolicyGradientBased.DeepLearning_ProximalPolicyApproximation.discrete_action",

        "NUM_EPOCHS": 4,
        "NUM_EPISODES": 1200,

        "MAX_STEPS_PER_EPISODE": 500,

        "GAMMA": 0.99,
        "LEARNING_RATE": 0.002,

        "EPSILON": 0.1,
        "CLIPPING_LOSS_RATIO": 0.2,
        "MAX_GRADIENT_NORM": 0.5,

        "UPDATE_STEP_INTERVAL" : 2000,

        "ENV_NAME": "CartPole-v1",
        "ENV_DISPLAY_ON": 0,

        "ACTOR_HIDDEN_LAYER_NODES": [
            {
                "LAYER_TYPE": "LINEAR",
                "NUM_NODES": 64,
                "ACTIVATION_FN": "relu"
            },
            {
                "LAYER_TYPE": "LINEAR",
                "NUM_NODES": 64,
                "ACTIVATION_FN": "relu"
            }
        ],

         "CRITIC_HIDDEN_LAYER_NODES": [
            {
                "LAYER_TYPE": "LINEAR",
                "NUM_NODES": 64,
                "ACTIVATION_FN": "relu"
            },
            {
                "LAYER_TYPE": "LINEAR",
                "NUM_NODES": 64,
                "ACTIVATION_FN": "relu"
            }
        ],

        "LOG_SKIP_INTERVAL": 1,
        "REWARD_GOAL": 499,
        "CONSECUTIVE_GOAL_HITS": 3,
        "LOG_MEAN_INTERVAL": 1,
        "MAX_RESULT_COUNT": 2,

        "AUTO_ARCHIVE": 1,
        "AUTO_INTERRUPT_HANDLING": 1,
    }
)

def fn_get_args():
    return app_info