
"""Types shared across modules."""

import enum


@enum.unique
class SequenceType(enum.Enum):
  """Sequence data types we know how to preprocess.

  If you need to preprocess additional video data, you must add it here.
  """

  FRAMES = "frames"
  FRAME_IDXS = "frame_idxs"
  VIDEO_NAME = "video_name"
  VIDEO_LEN = "video_len"

  def __str__(self):  # pylint: disable=invalid-str-returned
    return self.value
