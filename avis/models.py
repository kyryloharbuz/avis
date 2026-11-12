# THE DATA LAYER -- the value objects every part of the system exchanges.
# Not behaviour but facts: "an object was found", "here is the error", "here is
# the command".
#
# This module imports nothing of ours -- not cv2, not YOLO, not an interface --
# so every other layer can depend on it without a cycle.

from dataclasses import dataclass


@dataclass
class Detection:
    """One detected object in a frame (box plus detector metadata)."""
    id: int                                  # TRACK id -- WHICH instance (the tracker's counter)
    xyxy: tuple[float, float, float, float]  # box coordinates: (x1, y1, x2, y2)
    conf: float                              # model confidence (0..1)
    cls: int                                 # CLASS id -- WHAT it is (0=person, 56=chair, ...)
    # Human-readable class name ("chair"), filled in by the detector from the
    # model. The "" default makes the field OPTIONAL -- older code and tests that
    # build a Detection without a name keep working.
    name: str = ""

    # The geometry formulas live on the object rather than being smeared across
    # the call sites.
    @property
    def center(self) -> tuple[float, float]:
        """Box centre (cx, cy)."""
        x1, y1, x2, y2 = self.xyxy
        return (x1 + x2) / 2, (y1 + y2) / 2

    @property
    def area(self) -> float:
        """Box area -- an indirect indicator of distance (larger = closer)."""
        x1, y1, x2, y2 = self.xyxy
        return (x2 - x1) * (y2 - y1)


@dataclass
class ControlError:
    """How far the target is from where it should be (the controller's input)."""
    error_x: float  # horizontal offset of the target centre from the frame centre (pixels)
    error_y: float  # vertical offset (pixels)
    area: float     # current box area (for distance control)
    # The same offsets, but NORMALISED: -1 = left frame edge, 0 = centre, +1 = right.
    # Why: in pixels "a lot" depends on resolution, in fractions it does not.
    # norm_x is what decides whether forward motion is allowed yet (see controller).
    norm_x: float = 0.0
    norm_y: float = 0.0
    # Distance IN METRES, computed from BOX HEIGHT (None -- there is no box).
    # Controlling in metres beats area or pixels: the error in metres is linear
    # in distance, whereas area ~ 1/distance^2 (from 2 m to 1 m it grows
    # FOURFOLD). That nonlinearity made the drone lurch up close and barely crawl
    # from far away.
    distance_m: float = None
    # The box touches a frame edge -> the box GEOMETRY is unreliable, because part
    # of the target is cut off and the box centre is displaced. Only the vertical
    # axis reads this. It does NOT mean "the distance cannot be measured" -- that
    # reading killed forward motion twice over, because side clipping happens
    # simply from drifting towards the edge.
    box_clipped: bool = False
    # The fraction of the frame the box occupies (0..1). A DIRECT proximity signal
    # that depends neither on calibration nor on pose -- unlike distance_m.
    # Used as a GUARD: a box filling half the screen = too close.
    frame_fill: float = 0.0


@dataclass
class Command:
    """A motion command for the drone (three independent axes, like RC sticks)."""
    yaw: float       # + = turn right,        - = left
    vertical: float  # + = up,                - = down
    forward: float   # + = forward (towards the target), - = backward
