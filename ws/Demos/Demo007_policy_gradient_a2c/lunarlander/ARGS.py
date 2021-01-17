from ws.RLUtils.common.DotDict import DotDict

app_info = DotDict(
    {
        "STRATEGY": "CAT3_policy_gradient_based.fn_approx_yes__actor_critic.a2c",

        "NUM_EPOCHS": 1,
        "NUM_EPISODES": 12000,

        "MAX_STEPS_PER_EPISODE": 500,

        "GAMMA": 0.99,
        "LEARNING_RATE": 0.002,

        "CLIPPING_LOSS_RATIO": 0.2,
        "MAX_GRADIENT_NORM": 0.5,

        "UPDATE_STEP_INTERVAL": 2000,

        "ENV_NAME": "LunarLander-v2",
        "ENV_DISPLAY_ON": 0,

        "LOG_SKIP_INTERVAL": 10,
        "REWARD_GOAL": 250,
        "CONSECUTIVE_GOAL_HITS": 2,
        "LOG_MEAN_INTERVAL": 5,
        "MAX_RESULT_COUNT": 100,

        "NUM_EPISODES_FOR_UPDATE": 1,
        "ENV_SEED": 523,

        "AUTO_ARCHIVE": 1,
    }
)

def fn_get_args():
    return app_info