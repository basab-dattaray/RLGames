from enum import Enum


class PolicyTypes(Enum):
    POLICY_TRAINING_EXECUTION = 1
    POLICY_TRAINING_EVAL = 2
    POLICY_TESTING_EVAL = 3