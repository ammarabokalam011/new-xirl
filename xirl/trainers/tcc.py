
"""TCC trainer."""

from typing import Dict, List, Union

import torch
from xirl.losses import compute_tcc_loss
from xirl.trainers.base import Trainer

BatchType = Dict[str, Union[torch.Tensor, List[str]]]


class TCCTrainer(Trainer):
  """A trainer for Temporal Cycle Consistency Learning [1].

  References:
    [1]: arxiv.org/abs/1904.07846
  """

  def __init__(
      self,
      model,
      optimizer,
      device,
      config,
  ):
    super().__init__(model, optimizer, device, config)

    self.normalize_embeddings = config.model.normalize_embeddings
    self.stochastic_matching = config.loss.tcc.stochastic_matching
    self.loss_type = config.loss.tcc.loss_type
    self.similarity_type = config.loss.tcc.similarity_type
    self.cycle_length = config.loss.tcc.cycle_length
    self.temperature = config.loss.tcc.softmax_temperature
    self.label_smoothing = config.loss.tcc.label_smoothing
    self.variance_lambda = config.loss.tcc.variance_lambda
    self.huber_delta = config.loss.tcc.huber_delta
    self.normalize_indices = config.loss.tcc.normalize_indices

  def compute_loss(
      self,
      embs,
      batch,
  ):
    steps = batch["frame_idxs"].to(self._device)
    seq_lens = batch["video_len"].to(self._device)

    # Dynamically determine the number of cycles if using stochastic
    # matching.
    batch_size, num_cc_frames = embs.shape[:2]
    num_cycles = int(batch_size * num_cc_frames)

    return compute_tcc_loss(
        embs=embs,
        idxs=steps,
        seq_lens=seq_lens,
        stochastic_matching=self.stochastic_matching,
        normalize_embeddings=self.normalize_embeddings,
        loss_type=self.loss_type,
        similarity_type=self.similarity_type,
        num_cycles=num_cycles,
        cycle_length=self.cycle_length,
        temperature=self.temperature,
        label_smoothing=self.label_smoothing,
        variance_lambda=self.variance_lambda,
        huber_delta=self.huber_delta,
        normalize_indices=self.normalize_indices,
    )
