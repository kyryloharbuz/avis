# Kalman filter (constant-velocity model) for smoothing and predicting the
# target's position. A plain class, no interface.
#
# WHY IT WAS ADDED: Wi-Fi latency. Video and commands share the link, so every
# decision is made on an image that is already old. Tracking the target's
# VELOCITY instead of its last known box is what makes the loop usable at that
# delay. The same estimate holds the target when it briefly disappears -- that
# is the PRED state -- and it smooths the detector's box jitter, which would
# otherwise twitch the control.
#
# WHAT IT DOES NOT DO, stated so the claim is not read as more than it is:
# predict() is called with the frame-to-frame dt, so the state is advanced to
# the CURRENT frame, not led forward by the link delay. Compensating the delay
# itself would mean predicting dt + lag ahead; that is not done here and has
# not been measured.
#
# State = [x, y, vx, vy] (position + velocity). We only measure (x, y).

import numpy as np


class KalmanFilter:
    # process_var is the "process noise" (how abruptly the target may change
    #   velocity); measurement_var is the "measurement noise" (how much to
    #   distrust the detector). These two are the main tuning knobs.
    # blind_decay_after -- how many seconds of flying "blind" (no measurements)
    #   we keep trusting the constant velocity; blind_decay_per_second -- the
    #   fraction the velocity decays to per second after that (0.05 = 5% left).
    # WHY THE DECAY: a constant-velocity model is only honest over a short
    #   horizon. Without decay the prediction extrapolates FOREVER and flies
    #   off the frame (x=3000), so the PID chases a phantom and re-acquisition
    #   looks for the target where it cannot possibly be. With decay the
    #   prediction coasts to a stop near where the target was last seen.
    #
    # KNOWN CONSEQUENCE, worth stating plainly because it looks like a bug.
    # Once the velocity has decayed the prediction FREEZES in place, and
    # nothing here limits how long that lasts. Measured by replaying
    # flight_20261002_185914: PRED runs of 87, 77, 63 and 56 frames, i.e. 4-6
    # seconds of a motionless phantom box on screen with error_x stuck at a
    # constant value (-222, +30, +270 px respectively). The drone correctly
    # does NOT act on it -- FrameResult.has_target is False in PRED -- but
    # the operator sees a confident-looking box and no motion. If you need the
    # loop to give up sooner, cap the PRED duration in the pipeline; do not
    # "fix" it here, because freezing is exactly what this decay is for.
    def __init__(self, process_var=50.0, measurement_var=4.0,
                 blind_decay_after=0.4, blind_decay_per_second=0.05):
        self._process_var = process_var
        self._blind_decay_after = blind_decay_after
        self._blind_decay_per_second = blind_decay_per_second
        self._time_since_correct = 0.0   # seconds since the last real measurement
        self._H = np.array([[1, 0, 0, 0],       # H: we observe x and y only
                            [0, 1, 0, 0]], dtype=float)
        self._R = np.eye(2) * measurement_var   # R: measurement noise
        self._x = None  # state 4x1 [[x],[y],[vx],[vy]] (None = not initialised)
        self._P = None  # covariance 4x4 (our uncertainty)

    @property
    def initialized(self) -> bool:
        return self._x is not None

    def reset(self):
        self._x = None
        self._P = None
        self._time_since_correct = 0.0

    def _transition(self, dt):                  # F: x += vx*dt, y += vy*dt
        return np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=float)

    def _process_noise(self, dt):               # Q: process noise (constant velocity)
        q = self._process_var
        dt2 = dt * dt
        dt3 = dt2 * dt
        return q * np.array([
            [dt3 / 3, 0,       dt2 / 2, 0],
            [0,       dt3 / 3, 0,       dt2 / 2],
            [dt2 / 2, 0,       dt,      0],
            [0,       dt2 / 2, 0,       dt],
        ], dtype=float)

    def _init_state(self, cx, cy):              # first measurement: position from the sensor
        self._x = np.array([[cx], [cy], [0.0], [0.0]], dtype=float)
        self._P = np.eye(4) * 1000.0            # large uncertainty to begin with

    # The classic Kalman cycle as two SEPARATE steps: PREDICT, then (if there is
    # a measurement) CORRECT. They used to be fused into a single update(); we
    # split them because App predicts ONCE per frame and reuses that prediction
    # for target re-acquisition as well -- predicting twice would be wrong.

    # PREDICT: advance the state by dt; return (x, y).
    # Calling this WITHOUT a measurement (target occluded or gone) lets the
    # state coast on its past velocity. That coasting is the memory of motion.
    def predict(self, dt):
        if self._x is None:
            return None
        F = self._transition(dt)
        self._x = F @ self._x
        self._P = F @ self._P @ F.T + self._process_noise(dt)

        # VELOCITY DECAY while blind: once we have had no measurement for longer
        # than blind_decay_after, damp vx and vy smoothly. This keeps the
        # prediction from flying off to infinity; it settles roughly where the
        # target was last seen.
        self._time_since_correct += dt
        if self._time_since_correct > self._blind_decay_after:
            factor = self._blind_decay_per_second ** dt   # e.g. 0.05^dt per frame
            self._x[2] *= factor                          # vx
            self._x[3] *= factor                          # vy
        return float(self._x[0]), float(self._x[1])

    # CORRECT: refine the state with a fresh measurement (cx, cy); return the
    # smoothed (x, y). Call AFTER predict(). The first measurement just
    # initialises the filter.
    def correct(self, cx, cy):
        self._time_since_correct = 0.0   # a real measurement resets the decay
        if self._x is None:
            self._init_state(cx, cy)
            return cx, cy
        z = np.array([[cx], [cy]], dtype=float)          # measurement (2x1)
        y = z - self._H @ self._x                        # innovation: fact - prediction
        S = self._H @ self._P @ self._H.T + self._R
        K = self._P @ self._H.T @ np.linalg.inv(S)       # Kalman gain: whom to trust
        self._x = self._x + K @ y
        self._P = (np.eye(4) - K @ self._H) @ self._P
        return float(self._x[0]), float(self._x[1])
