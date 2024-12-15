
"""TCN config."""

from base_configs.pretrain import get_config as _get_config


def get_config():
  """TCN config."""

  config = _get_config()

  config.algorithm = "tcn"
  config.optim.train_max_iters = 4_000
  config.frame_sampler.strategy = "window"
  config.frame_sampler.num_frames_per_sequence = 40
  config.model.model_type = "resnet18_linear"
  config.model.normalize_embeddigs = False
  config.model.learnable_temp = False
  config.loss.tcn.pos_radius = 1
  config.loss.tcn.neg_radius = 4
  config.loss.tcn.num_pairs = 2
  config.loss.tcn.margin = 1.0
  config.loss.tcn.temperature = 0.1

  return config
