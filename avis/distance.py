# BOX SIZE -> DISTANCE. One measure of proximity: BOX HEIGHT.
#
# WHAT USED TO BE HERE AND WHY IT IS GONE.
# This file held a DistanceEstimator class: calibration in a file, a SECOND
# measure (shoulder width) and automatic switching between the two. Measurements
# on flight_20260926_224656 (2872 frames with a target in AUTO -- the frame count
# reproduces exactly) showed that this very design produced three separate
# operator complaints:
#
#   * 56% of measurements (1549 of 2741) went through WIDTH. And width depends
#     on POSE: when a person turns sideways the shoulders narrow, the system
#     reads it as "the target moved away" and flies at them. The measured
#     stature-to-width ratio was treated by the code as a fixed 2.26, while in
#     practice it varied widely: the band commonly quoted as 1.61 to 2.77 is
#     roughly p16-p95, and the full range over those frames is 0.67 to 4.64 with
#     a median of 2.15. The disagreement between the two measures on one and the
#     same box: -29% to +23%.
#
#   * 243 height<->width switches, median distance JUMP 0.18 m -- larger than
#     the whole deadzone (0.20 m). So a single change of measure flipped the
#     drone between "hold" and "chase" with no physical cause at all.
#
#   * calibration added a third variable where none was needed (see below).
#
# WHY HEIGHT, AND ONLY HEIGHT. A person's stature does NOT change when they turn
# sideways -- unlike shoulder width. It is the only available measure that is
# pose-invariant.
#
# WHY K IS NO LONGER CALIBRATED. Distance control is linear, and K enters it
# purely as a SCALE: the error (K/h - K/h_setpoint) is proportional to K. So
# changing K is identical to changing distance_kp. Calibration supplied no new
# information, it SILENTLY shifted the controller gain. K is now a fixed unit
# scale, needed only so that the setpoint and the on-screen readout are in
# metres rather than pixels.
#
# THE PHYSICAL LIMIT YOU MUST KNOW ABOUT. Height measures distance only while
# the person FITS in the frame. Closer in, the box runs into the frame edges,
# the height stops growing and the measurement saturates at
# nearest_measurable_m(H). This is exactly why a setpoint of 0.75 m was
# unreachable: the limit of 1.33 m lay ABOVE the setpoint, so the distance read
# as "too far" in essentially every frame and the drone pushed forward
# continuously.
#
# This is a property of the vertical FOV, and resolution does NOT change it: K
# and the frame height scale together (K/H = the fraction of the frame a person
# occupies at 1 m). So moving 480p -> 720p does not shift the limit -- verified.

# Unit scale (metres x pixels). Taken from a real Tello: a 1.7 m person gave a
# 318 px box at 2.00 m, i.e. 318 x 2.00 = 636, rounded to 635 below. NOT
# calibration -- just so that metres are metres.
DISTANCE_K = 635.0


def meters_from_height(box_height_px):
    """Distance in metres from box height. None -- the box is invalid.

    Pinhole camera model: height_in_pixels = f x real_height / distance,
    therefore distance = K / height_in_pixels."""
    if box_height_px is None or box_height_px <= 0:
        return None
    return DISTANCE_K / box_height_px


def height_for(distance_m):
    """The inverse problem: what box height to expect at a given distance.
    Needed for diagnostics -- so a setpoint in metres can be checked for
    feasibility before the flight."""
    if distance_m is None or distance_m <= 0:
        return None
    return DISTANCE_K / distance_m


def nearest_measurable_m(frame_h):
    """The NEAREST distance box height is still able to express.

    Closer than this the person does not fit in the frame, the box runs into its
    edges, and the measurement freezes at this value. The distance setpoint MUST
    be further out with margin: otherwise the system considers the target "far"
    even when it is standing right against the drone."""
    if not frame_h:
        return None
    return DISTANCE_K / float(frame_h)


def frame_fill(xyxy, frame_w, frame_h):
    """What FRACTION of the frame the box occupies (0..1) -- the PROXIMITY GUARD.

    Deliberately kept apart from the distance measurement and given a different
    role. The frame fraction depends on pose (arms spread raise it without the
    distance changing), so it must NOT be trusted with the "chase" decision. But
    it depends neither on calibration nor on box clipping -- so the "stop, we are
    too close" decision can be trusted to it, and that is what keeps it useful.

    It works even where height has already saturated: a box filling half the
    screen means "very close" in any pose."""
    x1, y1, x2, y2 = xyxy
    w = max(0.0, min(x2, frame_w) - max(x1, 0.0))
    h = max(0.0, min(y2, frame_h) - max(y1, 0.0))
    return (w * h) / float(frame_w * frame_h)


def touches_edge(xyxy, frame_w, frame_h, margin=2):
    """Whether the box touches a frame edge.

    It means exactly one thing: the box GEOMETRY is unreliable, because part of
    the target has been cut off. Only the vertical axis uses this -- the centre
    of a clipped box is displaced, and the target's "usual level" cannot be
    learned from it.

    IMPORTANT, WHAT THIS IS NO LONGER. The same flag used to mean "the distance
    cannot be measured" and killed forward motion. That broke the approach
    twice over, because side clipping happens simply when the target has drifted
    towards the frame edge -- which has nothing to do with proximity."""
    x1, y1, x2, y2 = xyxy
    return (x1 <= margin or y1 <= margin
            or x2 >= frame_w - margin or y2 >= frame_h - margin)
