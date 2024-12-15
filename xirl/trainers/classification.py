
"""Goal classifier trainer."""

from typing import Dict, List, Union

import torch
import torch.nn.functional as F
from xirl.trainers.base import Trainer

BatchType = Dict[str, Union[torch.Tensor, List[str]]]


class GoalFrameClassifierTrainer(Trainer):
  """A trainer that learns to classifiy whether an image is a goal frame.

  This should be used in conjunction with the LastFrameAndRandomFrames frame
  sampler which ensures the batch of frame sequences consists of first
  one goal frame, then by N - 1 random other frames.
  """

  def compute_loss(
      self,
      embs,
      batch,
  ):
    del batch

    batch_size, num_cc_frames, _ = embs.shape

    # Create the labels tensor.
    row_tensor = torch.FloatTensor([1] + [0] * (num_cc_frames - 1))
    label_tensor = row_tensor.unsqueeze(0).repeat(batch_size, 1)
    label_tensor = label_tensor.to(self._device)

    return F.binary_cross_entropy_with_logits(
        embs.view(batch_size * num_cc_frames),
        label_tensor.view(batch_size * num_cc_frames),
    )
