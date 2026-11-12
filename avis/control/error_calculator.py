# Computing the control error: how far the target is displaced from the frame
# centre, and how close it is.
#
# A DistanceEstimator USED TO BE INJECTED HERE -- an object with calibration in a
# file and a choice of measure. It is gone: the distance is now computed by a
# single formula from box height (see avis/distance.py), so there is nothing
# left to inject. The class remains an object -- App receives it through the
# constructor like everything else.

from avis.distance import frame_fill, meters_from_height, touches_edge
from avis.models import ControlError


class ErrorCalculator:
    """Target centre plus its box -> control error on all three axes."""

    # We take cx, cy ready-made (the Kalman-smoothed centre) plus the area.
    # box -- the target's box (xyxy). Without it there is nothing to compute the
    # distance from.
    def compute(self, cx, cy, area, frame_width, frame_height, box=None) -> ControlError:
        ex = cx - frame_width / 2            # right of centre -> +, left -> -
        ey = cy - frame_height / 2           # below centre -> +, above -> -

        distance_m, clipped, fill = None, False, 0.0
        if box is not None:
            fill = frame_fill(box, frame_width, frame_height)
            clipped = touches_edge(box, frame_width, frame_height)
            # ONE measure -- box height. No choice and no switching: the
            # switching itself produced 0.18 m jumps, larger than the deadzone.
            distance_m = meters_from_height(box[3] - box[1])

        return ControlError(
            error_x=ex,
            error_y=ey,
            area=area,                       # the area is passed through as is
            # Normalised by the HALF-width: at the frame edge this is exactly +/-1.
            norm_x=ex / (frame_width / 2),
            norm_y=ey / (frame_height / 2),
            distance_m=distance_m,
            box_clipped=clipped,
            frame_fill=fill,
        )
