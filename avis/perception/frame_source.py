# Frame source. One of the few interfaces in the project, and it has earned its
# place: WebcamSource, TelloSource and ReplaySource all implement it, and App
# reads frames without knowing which one it holds.

from abc import ABC, abstractmethod

import cv2


class FrameSource(ABC):
    """The frame source contract: where images come from."""

    @abstractmethod
    def read(self):
        """Return the next BGR frame, or None if there is no frame."""

    @abstractmethod
    def release(self):
        """Release the resource (camera or file)."""


class WebcamSource(FrameSource):
    """Webcam implementation of the source (OpenCV)."""

    def __init__(self, index=0):
        self._cap = cv2.VideoCapture(index)

    def read(self):
        ok, frame = self._cap.read()      # tuple unpacking: (success, frame)
        return frame if ok else None

    def release(self):
        self._cap.release()
