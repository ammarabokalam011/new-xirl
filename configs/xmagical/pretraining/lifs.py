
"""LIFS config."""

from base_configs.pretrain import get_config as _get_config


def get_config():
  """LIFS config."""

  config = _get_config()

  config.algorithm = "lifs"
  config.optim.train_max_iters = 8_000
  config.frame_sampler.strategy = "variable_strided"
  config.model.model_type = "resnet18_linear_ae"
  config.model.embedding_size = 32
  config.model.normalize_embeddings = False
  config.model.learnable_temp = False
  config.loss.lifs.temperature = 0.1
  config.eval.downstream_task_evaluators = [
      "reward_visualizer",
      "kendalls_tau",
      "reconstruction_visualizer",
  ]

  return config
