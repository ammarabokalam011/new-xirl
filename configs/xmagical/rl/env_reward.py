
"""Env reward config."""

from base_configs.rl import get_config as _get_config
from configs.constants import XMAGICALTrainingIterations
from ml_collections import ConfigDict
from utils import copy_config_and_replace


def get_config(embodiment):
  """Parameterize base RL config based on provided embodiment.

  This simply modifies the number of training steps based on presets defined
  in `constants.py`.

  Args:
    embodiment (str): String denoting embodiment name.

  Returns:
    ConfigDict corresponding to given embodiment string.
  """
  config = _get_config()

  possible_configs = dict()
  for emb, iters in XMAGICALTrainingIterations.iteritems():
    possible_configs[emb] = copy_config_and_replace(
        config,
        {"num_train_steps": iters},
    )

  return possible_configs[embodiment]
