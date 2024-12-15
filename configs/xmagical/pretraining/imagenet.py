
"""Raw ImageNet config."""

from base_configs.pretrain import get_config as _get_config


def get_config():
  """Raw ImageNet config."""

  config = _get_config()

  config.model.model_type = "resnet18_features"

  return config
