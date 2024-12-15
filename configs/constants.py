
"""Settings we used for the CoRL 2021 experiments."""

from typing import Dict
from ml_collections.config_dict import FrozenConfigDict

# The embodiments we used in the x-MAGICAL experiments.
EMBODIMENTS = frozenset([
    "shortstick",
    "mediumstick",
    "longstick",
    "gripper",
])

# All baseline pretraining strategies we ran for the CoRL experiments.
ALGORITHMS = frozenset([
    "xirl",
    "tcn",
    "lifs",
    "goal_classifier",
    "raw_imagenet",
])

# A mapping from x-MAGICAL embodiment to RL training iterations.
XMAGICALTrainingIterations = FrozenConfigDict({
    "longstick": 75_000,
    "mediumstick": 250_000,
    "shortstick": 500_000,
    "gripper": 500_000,
})

# A mapping from RLV environment to RL training iterations.
RLVTrainingIterations = FrozenConfigDict({
    "state_pusher": 500_000,
})

# A mapping from x-MAGICAL embodiment to Gym environment name.
XMAGICAL_EMBODIMENT_TO_ENV_NAME: Dict[str, str] = {
    k: f"SweepToTop-{k.capitalize()}-State-Allo-TestLayout-v0"
    for k in EMBODIMENTS
}
