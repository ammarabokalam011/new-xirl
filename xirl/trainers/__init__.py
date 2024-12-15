
"""Trainers."""

from .base import Trainer
from .classification import GoalFrameClassifierTrainer
from .lifs import LIFSTrainer
from .tcc import TCCTrainer
from .tcn import TCNCrossEntropyTrainer
from .tcn import TCNTrainer

__all__ = [
    "Trainer",
    "TCCTrainer",
    "TCNTrainer",
    "TCNCrossEntropyTrainer",
    "LIFSTrainer",
    "GoalFrameClassifierTrainer",
]
