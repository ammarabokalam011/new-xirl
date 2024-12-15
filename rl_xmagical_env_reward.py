
"""X-MAGICAL: Train a policy with the sparse environment reward."""

import subprocess

from absl import app
from absl import flags
from absl import logging
from configs.constants import EMBODIMENTS
from configs.constants import XMAGICAL_EMBODIMENT_TO_ENV_NAME
from torchkit.experiment import string_from_kwargs
from torchkit.experiment import unique_id

# pylint: disable=logging-fstring-interpolation

FLAGS = flags.FLAGS
CONFIG_PATH = "configs/xmagical/rl/env_reward.py"

flags.DEFINE_enum("embodiment", None, EMBODIMENTS, "Which embodiment to train.")
flags.DEFINE_list("seeds", [0, 5], "List specifying the range of seeds to run.")
flags.DEFINE_string("device", "cuda:0", "The compute device.")


def main(_):
  # Map the embodiment to the x-MAGICAL env name.
  env_name = XMAGICAL_EMBODIMENT_TO_ENV_NAME[FLAGS.embodiment]

  # Generate a unique experiment name.
  experiment_name = string_from_kwargs(
      env_name=env_name,
      reward="sparse_env",
      uid=unique_id(),
  )
  logging.info("Experiment name: %s", experiment_name)

  # Execute each seed in parallel.
  procs = []
  for seed in range(*list(map(int, FLAGS.seeds))):
    procs.append(
        subprocess.Popen([  # pylint: disable=consider-using-with
            "python",
            "train_policy.py",
            "--experiment_name",
            experiment_name,
            "--env_name",
            f"{env_name}",
            "--config",
            f"{CONFIG_PATH}:{FLAGS.embodiment}",
            "--seed",
            f"{seed}",
            "--device",
            f"{FLAGS.device}",
        ]))

  # Wait for each seed to terminate.
  for p in procs:
    p.wait()


if __name__ == "__main__":
  flags.mark_flag_as_required("embodiment")
  app.run(main)
