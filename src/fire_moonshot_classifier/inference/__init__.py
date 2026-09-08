"""Deployment API: no SITL files, feature caches, labels or W&B required."""
from .predictor import TrajectoryPredictor, predict_trajectory

__all__ = ["TrajectoryPredictor", "predict_trajectory"]
