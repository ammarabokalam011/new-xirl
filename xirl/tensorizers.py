
"""Tensorizers convert a packet of video data into a packet of video tensors."""

import abc
from typing import Any, Dict, Union

import numpy as np
import torch
import torchvision.transforms.functional as TF

from xirl.types import SequenceType

DataArrayPacket = Dict[SequenceType, Union[np.ndarray, str, int]]
DataTensorPacket = Dict[SequenceType, Union[torch.Tensor, str]]


class Tensorizer(abc.ABC):
  """Base tensorizer class.

  Custom tensorizers must subclass this class.
  """

  @abc.abstractmethod
  def __call__(self, x):
    pass


class IdentityTensorizer(Tensorizer):
  """Outputs the input as is."""

  def __call__(self, x):
    return x


class LongTensorizer(Tensorizer):
  """Converts the input to a LongTensor."""

  def __call__(self, x):
    return torch.from_numpy(np.asarray(x)).long()


class FramesTensorizer(Tensorizer):
  """Converts a sequence of video frames to a batched FloatTensor."""

  def __call__(self, x):
    assert x.ndim == 4, "Input must be a 4D sequence of frames."
    frames = []
    for frame in x:
      frames.append(TF.to_tensor(frame))
    return torch.stack(frames, dim=0)


class ToTensor:
  """Convert video data to video tensors."""

  MAP = {
      SequenceType.FRAMES: FramesTensorizer,
      SequenceType.FRAME_IDXS: LongTensorizer,
      SequenceType.VIDEO_NAME: IdentityTensorizer,
      SequenceType.VIDEO_LEN: LongTensorizer,
  }

  def __call__(self, data):
    """Iterate and transform the data values.

    Args:
      data: A dictionary containing key, value pairs where the key is an enum
        member of `SequenceType` and the value is either an int, a string or an
        ndarray respecting the key type.

    Raises:
      ValueError: If the input is not a dictionary or one of its keys is
        not a supported sequence type.

    Returns:
      The dictionary with the values tensorized.
    """
    tensors = {}
    for key, np_arr in data.items():
      tensors[key] = ToTensor.MAP[key]()(np_arr)
    return tensors
