# Perception: frame source -> detection -> target choice -> position smoothing.
# The facade keeps module paths free to move.

from avis.perception.detector import Detector
from avis.perception.frame_source import FrameSource, WebcamSource
from avis.perception.target_selector import TargetSelector
from avis.perception.tracking_filter import KalmanFilter

__all__ = ["FrameSource", "WebcamSource", "Detector", "TargetSelector", "KalmanFilter"]
