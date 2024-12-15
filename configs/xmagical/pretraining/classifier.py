
"""Goal classifier config."""

from base_configs.pretrain import get_config as _get_config


def get_config():
  """Goal classifier config."""

  config = _get_config()

  config.algorithm = "goal_classifier"
  config.optim.train_max_iters = 6_000
  config.frame_sampler.strategy = "last_and_randoms"
  config.frame_sampler.num_frames_per_sequence = 15
  config.model.model_type = "resnet18_classifier"
  config.model.normalize_embeddings = False
  config.model.learnable_temp = False

  return config
