"""Compute and store the mean goal embedding using a trained model."""

import os
import typing

from absl import app
from absl import flags
from absl import logging
import numpy as np
import torch
from torchkit import CheckpointManager
from tqdm.auto import tqdm
import utils
from xirl import common
from xirl.models import SelfSupervisedModel

# pylint: disable=logging-fstring-interpolation

FLAGS = flags.FLAGS

flags.DEFINE_string("experiment_path", None, "Path to model checkpoint.")
flags.DEFINE_boolean(
    "restore_checkpoint", True,
    "Restore model checkpoint. Disabling loading a checkpoint is useful if you "
    "want to measure performance at random initialization.")

# Function to embed using the self-supervised model
def embed_self_supervised(model, downstream_loader, device):
    goal_embs = []
    init_embs = []
    for class_name, class_loader in downstream_loader.items():
        logging.info("Embedding %s.", class_name)
        for batch in tqdm(iter(class_loader), leave=False):
            out = model.infer(batch["frames"].to(device))
            emb = out.numpy().embs
            init_embs.append(emb[0, :])
            goal_embs.append(emb[-1, :])
    goal_emb = np.mean(np.stack(goal_embs, axis=0), axis=0, keepdims=True)
    
    dist_to_goal = np.linalg.norm(np.stack(init_embs, axis=0) - goal_emb, axis=-1).mean()
    distance_scale = 1.0 / dist_to_goal
    return goal_emb, distance_scale

def setup():
    """Load the latest embedder checkpoint and dataloaders."""
    config = utils.load_config_from_dir(FLAGS.experiment_path)
    model = common.get_model(config)
    downstream_loaders = common.get_downstream_dataloaders(config, False)["train"]
    
    checkpoint_dir = os.path.join(FLAGS.experiment_path, "checkpoints")
    if FLAGS.restore_checkpoint:
        checkpoint_manager = CheckpointManager(checkpoint_dir, model=model)
        global_step = checkpoint_manager.restore_or_initialize()
        logging.info("Restored model from checkpoint %d.", global_step)
    else:
        logging.info("Skipping checkpoint restore.")
    
    return model, downstream_loaders

def main(_):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Use CUDA if available
    
    # Load self-supervised model and data loaders
    model, downstream_loader = setup()
    model.to(device).eval()
    
    goal_emb, distance_scale = embed_self_supervised(model, downstream_loader, device)
    
    # Save individual embeddings and distance scale to separate files
    utils.save_pickle(FLAGS.experiment_path, goal_emb, "self_supervised_emb.pkl")
    utils.save_pickle(FLAGS.experiment_path, distance_scale, "self_supervised_distance_scale.pkl")
    
    
	# Clear cache before embedding
    torch.cuda.empty_cache()

if __name__ == "__main__":
  flags.mark_flag_as_required("experiment_path")
  app.run(main)

ModelType = SelfSupervisedModel
DataLoaderType = typing.Dict[str, torch.utils.data.DataLoader]


def embed(
    model,
    downstream_loader,
    device,
):
  """Embed the stored trajectories and compute mean goal embedding."""
  goal_embs = []
  init_embs = []
  for class_name, class_loader in downstream_loader.items():
    logging.info("Embedding %s.", class_name)
    for batch in tqdm(iter(class_loader), leave=False):
      out = model.infer(batch["frames"].to(device))
      emb = out.numpy().embs
      init_embs.append(emb[0, :])
      goal_embs.append(emb[-1, :])
  goal_emb = np.mean(np.stack(goal_embs, axis=0), axis=0, keepdims=True)
  dist_to_goal = np.linalg.norm(
      np.stack(init_embs, axis=0) - goal_emb, axis=-1).mean()
  distance_scale = 1.0 / dist_to_goal
  return goal_emb, distance_scale


def setup():
  """Load the latest embedder checkpoint and dataloaders."""
  config = utils.load_config_from_dir(FLAGS.experiment_path)
  model = common.get_model(config)
  downstream_loaders = common.get_downstream_dataloaders(config, False)["train"]
  checkpoint_dir = os.path.join(FLAGS.experiment_path, "checkpoints")
  if FLAGS.restore_checkpoint:
    checkpoint_manager = CheckpointManager(checkpoint_dir, model=model)
    global_step = checkpoint_manager.restore_or_initialize()
    logging.info("Restored model from checkpoint %d.", global_step)
  else:
    logging.info("Skipping checkpoint restore.")
  return model, downstream_loaders


def main(_):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model, downstream_loader = setup()
  model.to(device).eval()
  goal_emb, distance_scale = embed(model, downstream_loader, device)
  utils.save_pickle(FLAGS.experiment_path, goal_emb, "goal_emb.pkl")
  utils.save_pickle(FLAGS.experiment_path, distance_scale, "distance_scale.pkl")


if __name__ == "__main__":
  flags.mark_flag_as_required("experiment_path")
  app.run(main)
