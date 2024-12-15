
"""LIFS trainer."""

from typing import Dict, List, Union

import torch
import torch.nn.functional as F
from xirl.trainers.base import Trainer

BatchType = Dict[str, Union[torch.Tensor, List[str]]]


class LIFSTrainer(Trainer):
  """A trainer that implements LIFS from [1].

  This should be used in conjunction with the VariableStridedSampler frame
  sampler, which assumes rough alignment between pairs of sequences and hence
  a time index can be used to correspond frames across sequences.

  Note that the authors of [1] do not implement a negative term in the
  contrastive loss. It is just a similarity (l2) loss with an autoencoding
  loss to prevent the embeddings from collapsing to trivial constants.

  References:
    [1]: https://arxiv.org/abs/1703.02949
  """

  def __init__(
      self,
      model,
      optimizer,
      device,
      config,
  ):
    super().__init__(model, optimizer, device, config)

    self.temperature = config.loss.lifs.temperature

  def compute_auxiliary_loss(
      self,
      out,
      batch,
  ):
    reconstruction = out.reconstruction
    frames = batch["frames"].to(self._device)
    b, t, _, _, _ = reconstruction.shape
    reconstruction = reconstruction.view((b * t, *reconstruction.shape[2:]))
    frames = frames.view((b * t, *frames.shape[2:]))
    _, _, sh, _ = reconstruction.shape
    _, _, h, _ = frames.shape
    scale_factor = sh / h
    frames_ds = F.interpolate(
        frames,
        mode="bilinear",
        scale_factor=scale_factor,
        recompute_scale_factor=False,
        align_corners=True,
    )
    return F.mse_loss(reconstruction, frames_ds)

  def compute_loss(
      self,
      embs,
      batch,
  ):
    del batch

    batch_size, num_cc_frames, num_dims = embs.shape

    # Compute pairwise squared L2 distances between embeddings.
    embs_flat = embs.view(-1, num_dims)
    distances = torch.cdist(embs_flat, embs_flat).pow(2)
    distances = distances / self.temperature

    # Each row in a batch corresponds to a frame sequence. Since this
    # baseline assumes rough alignment between sequences, we want columns,
    # i.e. frames in each row that belong to the same index to be close
    # together in embedding space.
    labels = torch.arange(num_cc_frames).unsqueeze(0).repeat(batch_size, 1)
    labels = labels.to(self._device)
    mask = labels.flatten()[:, None] == labels.flatten()[None, :]
    return (distances * mask.float()).sum(dim=-1).mean()
