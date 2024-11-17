
import os.path as osp
import subprocess

from absl import app
from absl import logging
from configs.constants import ALGORITHMS
from configs.constants import EMBODIMENTS
from torchkit.experiment import string_from_kwargs
from torchkit.experiment import unique_id
import yaml
import random


import os
from configs.constants import XMAGICAL_EMBODIMENT_TO_ENV_NAME

# pylint: disable=logging-fstring-interpolation

# Mapping from pretraining algorithm to config file.
ALGO_TO_CONFIG = {
    "xirl": "configs/xmagical/pretraining/tcc.py",
    "lifs": "configs/xmagical/pretraining/lifs.py",
    "tcn": "configs/xmagical/pretraining/tcn.py",
    "goal_classifier": "configs/xmagical/pretraining/classifier.py",
    "raw_imagenet": "configs/xmagical/pretraining/imagenet.py",
}

# We want to pretrain on the entire 1k demonstrations.
MAX_DEMONSTRATIONS = -1
embodiment = "shortstick"

algo = "xirl"

unique_name = True,

random_number = random.randint(1, 1000)  # You can adjust the range as needed
experiment_name = f"/home/user/xirl/exp/exp{random_number}"
print("experiment name: ",experiment_name)
embodiments = EMBODIMENTS if embodiment is None else [embodiment]

for embodiment in embodiments:
    # Generate a unique experiment name.
    print("embodiment: ",embodiment)
    kwargs = {
        "dataset": "xmagical",
        "mode": "cross",
        "algo": algo,
        "embodiment": embodiment,
    }
    if unique_name:
      kwargs["uid"] = unique_id()
    logging.info("Experiment name: %s", experiment_name)
    
    # Train on all classes but the given embodiment.
    trainable_embs = tuple(EMBODIMENTS - set([embodiment]))
    try:
        print("Start of experment\n")
        subprocess.run(
            [
                "python",
                "pretrain.py",
                "--experiment_name",
                experiment_name,
                "--raw_imagenet" if algo == "raw_imagenet" else " ",
                "--config",
                f"{ALGO_TO_CONFIG[algo]}",
                "--config.data.pretrain_action_class",
                f"{repr(trainable_embs)}",
                "--config.data.downstream_action_class",
                f"{repr(trainable_embs)}",
                "--config.data.max_vids_per_class",
                f"{MAX_DEMONSTRATIONS}",
            ],
            check=True,
            stdout=subprocess.PIPE,  # Capture standard output
            stderr=subprocess.PIPE,   # Capture standard error
            text=True  
        )
        

        # Note: This assumes that the config.root_dir value has not been
        # changed to its default value of 'tmp/xirl/pretrain_runs/'.
        exp_path = osp.join("/tmp/xirl/pretrain_runs/", experiment_name)
        print("Output:", result.stdout)
        print("end of experment\n")
    except subprocess.CalledProcessError as e:
        print("Error:", e.stderr)
        print("Return Code:", e.returncode)
    # The 'goal_classifier' baseline does not need to compute a goal embedding.
    if algo != "goal_classifier":
      subprocess.run(
          [
              "python",
              "compute_goal_embedding.py",
              "--experiment_path",
              experiment_name,
          ],
          check=True,
      )
    
    # Dump experiment metadata as yaml file.
    with open(osp.join(experiment_name, "metadata.yaml"), "w") as fp:
      yaml.dump(kwargs, fp)

seeds = [0, 5]
device = "cuda:0"
experiment_name = '/home/user/xirl/exp/exp720'
with open(osp.join(experiment_name, "metadata.yaml"), "r") as fp:
    kwargs = yaml.load(fp, Loader=yaml.FullLoader)

if kwargs["algo"] == "goal_classifier":
    reward_type = "goal_classifier"
else:
    reward_type = "distance_to_goal"

# Map the embodiment to the x-MAGICAL env name.
env_name = XMAGICAL_EMBODIMENT_TO_ENV_NAME[kwargs["embodiment"]]

# Generate a unique experiment name.
# experiment_name = string_from_kwargs(
#   env_name=env_name,
#   reward="learned",
#   reward_type=reward_type,
#   mode=kwargs["mode"],
#   algo=kwargs["algo"],
#   uid=unique_id(),
# )
logging.info("Experiment name: %s", experiment_name)

# Execute each seed in parallel.
procs = []
for seed in range(*list(map(int, seeds))):
    procs.append(
        subprocess.Popen([  # pylint: disable=consider-using-with
            "python",
            "train_policy.py",
            "--experiment_name",
            experiment_name,
            "--env_name",
            f"{env_name}",
            "--config",
            f"configs/xmagical/rl/env_reward.py:{kwargs['embodiment']}",
            "--config.reward_wrapper.pretrained_path",
            f"{experiment_name}",
            "--config.reward_wrapper.type",
            f"{reward_type}",
            "--seed",
            f"{seed}",
            "--device",
            f"{device}",
            '--resume',
        ]))

# Wait for each seed to terminate.
for p in procs:
    p.wait()