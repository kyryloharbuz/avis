# Visualisation: showing the frame with boxes and labels, and handling window input.
#
# WE KEEP AN INTERFACE HERE -- substitution is real: for screenless tests a
# HeadlessRenderer(Renderer) that draws nothing is handy (I have already used
# such a fake when checking App without a camera). A second implementation means
# the interface has paid for itself.

from abc import ABC, abstractmethod

import cv2


class Renderer(ABC):
    """The visualisation contract."""

    @abstractmethod
    def set_mouse_callback(self, callback):
        """Subscribe a function to the window's mouse events."""

    @abstractmethod
    def draw(self, frame, detections, target_id, fps, status, predicted_center=None):
        """Draw the frame with its markers and display it.
        status -- "VIS" (reliably visible) / "PRED" (steering by prediction) / "LOST".
        predicted_center -- (x, y) of the current Kalman hypothesis, if any."""

    @abstractmethod
    def is_open(self) -> bool:
        """Whether the window is still open."""


class HeadlessRenderer(Renderer):
    """Draws nothing. For REPLAY during tuning: the window and the drawing are
    the slowest part of a run, and while sweeping coefficients you look at the
    summary numbers, not the picture. Without a window a recording replays many
    times faster, and you can run many variants back to back from a script."""

    def set_mouse_callback(self, callback):
        pass

    def draw(self, frame, detections, target_id, fps, status, predicted_center=None):
        pass

    def is_open(self) -> bool:
        return True


class CvRenderer(Renderer):
    """The OpenCV implementation (an on-screen window)."""

    def __init__(self, window_name="avis"):
        self._window = window_name
        cv2.namedWindow(self._window)

    def set_mouse_callback(self, callback):
        cv2.setMouseCallback(self._window, callback)

    def draw(self, frame, detections, target_id, fps, status, predicted_center=None):
        # status may be compound: "VIS|AUTO" (tracking|flight state). For LOGIC
        # (colour, reticle) we take only the first part, but we display the whole
        # string -- so both the tracking and the flight mode are visible.
        base = status.split("|")[0]
        for det in detections:
            x1, y1, x2, y2 = map(int, det.xyxy)
            is_target = det.id == target_id
            color = (0, 0, 255) if is_target else (0, 255, 0)   # the target is red (BGR)
            thickness = 3 if is_target else 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            # Label: class name + track id, e.g. "chair #105".
            # If there is no name (e.g. in test stubs) we show just the id.
            label = f"{det.name} #{det.id}" if det.name else f"id={det.id}"
            # 0.4 is a small font; LINE_AA antialiases so small text stays legible.
            cv2.putText(frame, label, (x1, max(11, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        # THE KALMAN HYPOTHESIS. During an occlusion the screen used to show
        # NOTHING -- it looked as if tracking had died when in fact it was
        # working. We now draw where the system "believes" the target is: a
        # yellow reticle in PRED mode (steering blind).
        #
        # WORTH KNOWING WHEN THIS LOOKS WRONG: once the Kalman velocity has
        # decayed the reticle stops moving and can sit perfectly still for
        # seconds (measured PRED runs of 4-6 s). A motionless reticle plus a
        # motionless drone is the expected appearance of a long occlusion, not a
        # frozen program.
        if predicted_center is not None and base == "PRED":
            px, py = map(int, predicted_center)
            cv2.circle(frame, (px, py), 18, (0, 255, 255), 2)          # yellow circle
            cv2.line(frame, (px - 26, py), (px + 26, py), (0, 255, 255), 1)
            cv2.line(frame, (px, py - 26), (px, py + 26), (0, 255, 255), 1)
            cv2.putText(frame, "predicted", (px + 22, py - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.putText(frame, f"{fps:.1f} FPS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # The status line: the mode and which id we are tracking, at a glance.
        # It is also how you spot a bad re-acquisition (the id suddenly changes
        # to a stranger's).
        status_color = {"VIS": (0, 255, 0), "PRED": (0, 255, 255)}.get(base, (0, 0, 255))
        if target_id is None:
            status_label = status

        else:
            # Pull the target's name out of the current detections (if it is
            # visible right now).
            target_name = next((d.name for d in detections if d.id == target_id), "")
            status_label = f"{status}  {target_name} #{target_id}".rstrip()
        cv2.putText(frame, status_label, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        cv2.imshow(self._window, frame)

    def is_open(self) -> bool:
        return cv2.getWindowProperty(self._window, cv2.WND_PROP_VISIBLE) >= 1
