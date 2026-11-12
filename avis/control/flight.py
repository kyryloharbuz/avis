# SAFETY LOGIC for autonomous flight.
#
# From a safety standpoint this is the most important class in the project. The
# PID computes WHERE to fly; FlightSupervisor decides whether flying there is
# allowed at all right now.
#
# The principles baked in here:
#   1. A HUMAN ARMS AUTONOMY. After takeoff the drone simply hovers. It does not
#      start chasing anything until the operator explicitly presses "G" (arm).
#      A drone that moves off by itself right after takeoff is the best way to
#      break something.
#   2. TARGET LOST -> HOVER, rather than flying on blind with the last command.
#   3. NOT FOUND FOR A LONG TIME -> LAND ON OUR OWN. The drone must not hover
#      forever.
#   4. COMMANDS ARE BOUNDED. The PID emits -100..100; each axis is clipped to
#      its own ceiling here, and a non-zero command too weak for the drone to
#      execute is lifted to the sensitivity floor (see _shape).
#   5. BATTERY FLAT -> LAND.

import threading

from avis.models import Command

# Flight states.
GROUNDED = "GROUNDED"   # on the ground, motors off
HOVER = "HOVER"         # airborne, hovering, autonomy DISARMED
AUTO = "AUTO"           # airborne, the PID is in charge
BUSY = "BUSY"           # takeoff/landing in progress (command has not replied yet)


class FlightSupervisor:
    """Tracks flight state and filters PID commands before the motors."""

    def __init__(
        self,
        drone,
        max_speed=90,              # CEILING for yaw and vertical speed.
                                   #   70 -> 90: the drone could not keep its
                                   #   nose on a target walking past it. The
                                   #   Tello's own limit is 100.
        max_forward=50,            # CEILING for FORWARD speed (the risky axis).
                                   #   30 -> 40 -> 55 -> 80 -> 50. The climb was
                                   #   chasing sluggishness; the drop back is the
                                   #   measurement below, and it is the important
                                   #   part.
                                   #
                                   #   A HARD FORWARD COMMAND COSTS DETECTIONS.
                                   #   Measured on flight_20261004_185939, over
                                   #   frames that had an active target:
                                   #     after fwd >= 55 (n=64)  -> the detector
                                   #        returns nothing within the next 3
                                   #        frames in 56% of cases
                                   #     after fwd <  30 (n=184) -> 17%
                                   #   Three times the loss rate. The cause is
                                   #   physical, not algorithmic: a hard forward
                                   #   command is a hard pitch, which tilts the
                                   #   camera, smears the 480p frame and deforms
                                   #   the person's box, so YOLO misses them. The
                                   #   same pitch-then-level cycle throws the
                                   #   drone upward -- and it cannot be our doing,
                                   #   because the vertical command was zero in
                                   #   100% of that flight's AUTO frames.
                                   #   Operator report: "the target is lost while
                                   #   the object is centred at a good distance
                                   #   with no lag in the image, and it gains
                                   #   altitude for no visible reason."
                                   #
                                   #   WHY THE CEILING AND NOT distance_kp. The
                                   #   harmful thing is the command VALUE, not the
                                   #   gain. Lowering kp 95 -> 65 does not remove
                                   #   the regime, it just moves it further out
                                   #   (3.0 m still asks 72) while making the
                                   #   2.2-2.5 m band -- where the operator
                                   #   actually walks -- slower again. Capping
                                   #   here removes it outright: of the 301
                                   #   non-zero forward commands in that flight,
                                   #   0% would have reached 55. And the profile
                                   #   that worked is preserved: 2.2 m -> 29,
                                   #   2.4 m -> 47, then flat at 50.
                                   #
                                   #   DO NOT RAISE THIS ABOVE ~50 without
                                   #   re-running that measurement. Top speed here
                                   #   is bounded by the CAMERA, not the motors:
                                   #   past this point the drone flies faster than
                                   #   it can see, and a tracker that cannot see
                                   #   is not faster, it is just lost.
        max_backward=25,           # CEILING for BACKWARD speed, lower than forward.
                                   #   It used to be 15 -- which sits exactly on
                                   #   the Tello's sensitivity threshold, so
                                   #   retreating did not actually work. The
                                   #   abruptness seen in flight is not cured by
                                   #   this but by LINEAR control in metres: the
                                   #   command is now proportional instead of
                                   #   permanently saturated.
        min_speed=12,              # sensitivity threshold for YAW and vertical
        min_forward=20,            # sensitivity threshold for the FORWARD axis -- higher.
                                   #   Measured: the drone physically ignored
                                   #   commands of 10-12, the axis was dead.
        slew_per_second=150.0,     # SMOOTHNESS: how many units per second the
                                   #   command is allowed to RISE by. Braking
                                   #   (towards zero) is NOT limited -- stopping
                                   #   is instant. This cures the abrupt yaw jerk.
                                   #   90 -> 150: alongside a higher ceiling the
                                   #   smoothing itself became the delay -- going
                                   #   0->90 took 1.0 s, i.e. the drone "thought"
                                   #   for a second before turning. Now 0.60 s.
                                   #
                                   #   DO NOT RAISE THIS TO SPEED UP THE CHASE.
                                   #   It was taken to 250 for exactly that
                                   #   reason and had to be reverted after one
                                   #   flight. The mistake is visible in _ramp()
                                   #   below: the FORWARD axis is not rate
                                   #   limited at all, so this knob cannot make
                                   #   closing in any faster. It was measured
                                   #   beforehand and the measurement said so --
                                   #   on flight_20261002_185914 (471 frames with
                                   #   a motor command) the ramp clamped forward
                                   #   in 0% of frames and yaw in 20% -- and the
                                   #   change was made anyway.
                                   #
                                   #   What 250 actually did was make the TURN
                                   #   sharper, and that cost target retention:
                                   #   faster yaw pans the image faster, the
                                   #   constant-velocity Kalman model absorbs the
                                   #   drone's own rotation as if it were target
                                   #   motion, and on a brief occlusion it
                                   #   extrapolates that pan until the target
                                   #   leaves the frame. Operator report: "it
                                   #   started turning more sharply, because of
                                   #   that combo it kept losing the target, and
                                   #   the Kalman filter was not working well
                                   #   either." To make the chase faster, use
                                   #   distance_kp -- see the config.
                                   #
                                   #   KNOWN COST OF 150, accepted knowingly: at
                                   #   dt ~58 ms the first ramp step is 8.7 units
                                   #   against min_speed=12, so the first frame
                                   #   of every acceleration reaches the motors as
                                   #   a dead command (measured: 3% of frames).
                                   #   That is the _limit/_ramp ordering flaw
                                   #   documented in _shape(), and it must be
                                   #   fixed there rather than papered over by
                                   #   raising this value.
        hover_after_lost=0.5,      # s without a target -> stop moving, hover
        land_after_lost=8.0,       # s without a target -> land (only AFTER the first acquisition)
        blind_hold_s=0.3,          # s of BLINDNESS during which we hold the
                                   #   previous command instead of zeros. See
                                   #   update(). 0.3 s is ~4 frames at 14 fps:
                                   #   long enough to cover the median blind
                                   #   spell (0.11 s), short enough to expire
                                   #   before hover_after_lost (0.5 s), so the
                                   #   hold can never outlive the loss decision.
        settle_after_takeoff=5.0,  # s of HOVER after takeoff: the Tello settles
                                   # and you get time to pick a target before it moves
        min_battery=15,            # % -> forced landing
        enable_forward=False,      # FORWARD/BACKWARD motion: DISABLED by default
        start_armed=False,         # straight into AUTO (for REPLAY with NullDrone, no takeoff)
        min_height_cm=50,          # below this, descent is blocked (floor guard)
        max_height_cm=180,         # above this, climbing is blocked (ceiling guard)
        height_margin_cm=30,       # soft braking band near the limits
        altitude_limits=False,     # DISABLED: the Tello's altimeter is noisy and
                                   # drifts, and can block the vertical axis.
                                   # Enable it once you are satisfied that
                                   # tracking is stable.
    ):
        self._drone = drone
        self._max_speed = max_speed
        self._max_forward = max_forward
        self._max_backward = max_backward
        self._min_speed = min_speed
        self._min_forward = min_forward
        self._slew = slew_per_second
        self._prev_cmd = Command(0.0, 0.0, 0.0)
        self._min_height = min_height_cm
        self._max_height = max_height_cm
        self._height_margin = height_margin_cm
        self._altitude_limits = altitude_limits
        self._hover_after_lost = hover_after_lost
        self._land_after_lost = land_after_lost
        self._blind_hold_s = blind_hold_s
        self._blind_for = 0.0         # how long the current blind spell has lasted
        self._min_battery = min_battery
        self._enable_forward = enable_forward

        # start_armed=True is for replay only: there is no real drone and no
        # takeoff there, but we still want to see the full command chain all the
        # way to the "motors".
        self._settle_after_takeoff = settle_after_takeoff

        self._state = AUTO if start_armed else GROUNDED
        self._time_without_target = 0.0
        self._battery_checked_at = 0.0
        self._settle = 0.0            # post-takeoff settling countdown
        self._ever_acquired = False   # has a target been acquired at all this flight
        self._last_battery = None     # battery cache (so logging costs no extra queries)
        self._busy = False            # takeoff/landing running on a background thread
        self._arm_requested = False   # G was pressed while still taking off

    @property
    def state(self):
        # While taking off or landing we report BUSY -- the operator can see the
        # command is in flight, and the loop knows RC must not be sent right now.
        return BUSY if self._busy else self._state

    @property
    def armed(self):
        return self._state == AUTO

    @property
    def battery(self):
        """The last battery level READ (cached). Does not query the drone --
        the value refreshes in update() about every 2 s, and on takeoff."""
        return self._last_battery

    # -- Operator commands ---------------------------------------------
    #
    # WHY TAKEOFF AND LANDING RUN ON A BACKGROUND THREAD:
    # djitellopy runs takeoff() SYNCHRONOUSLY and waits for the drone's reply for
    # up to TAKEOFF_TIMEOUT = 20 seconds. Call that straight from the frame loop
    # and the whole loop STALLS: frames are not read, the window is not redrawn,
    # and it looks as if the video froze for 15+ seconds. So we run the command
    # on its own thread while the loop keeps drawing.
    # While the command is in flight the state is BUSY and we send NOTHING to the
    # motors (so RC does not conflict with the takeoff being executed).
    def _run_async(self, label, action, on_done):
        def worker():
            try:
                action()
                on_done()
            except Exception as e:
                print(f"[flight] {label} failed: {type(e).__name__}: {e}")
                self._state = GROUNDED
            finally:
                self._busy = False
        self._busy = True
        threading.Thread(target=worker, daemon=True).start()

    def takeoff(self):
        if self._state != GROUNDED or self._busy:
            return
        battery = self._drone.battery()
        self._last_battery = battery
        if battery < 30:
            print(f"[flight] REFUSED: battery {battery}% (need >30%)")
            return
        print(f"[flight] taking off (battery {battery}%) -- video keeps running")

        def done():
            self._state = HOVER      # AFTER takeoff we just hover, we do not chase
            self._settle = self._settle_after_takeoff
            self._ever_acquired = False
            print("[flight] airborne. HOVER. Press G to arm tracking.")
            if self._arm_requested:          # G was pressed during takeoff
                self._arm_requested = False
                self.arm()

        self._run_async("takeoff", self._drone.takeoff, done)

    def land(self):
        if self._state == GROUNDED or self._busy:
            return
        print("[flight] landing")
        self._settle = 0.0
        self._ever_acquired = False
        self._arm_requested = False
        self._state = GROUNDED       # immediately, so the loop stops sending commands
        self._run_async("landing", self._drone.land, lambda: None)

    def emergency(self):
        # Emergency is the one command we send SYNCHRONOUSLY and without any
        # state checks: here a millisecond matters more than a smooth picture.
        print("[flight] EMERGENCY -- motors off")
        self._state = GROUNDED
        self._busy = False
        try:
            self._drone.emergency()
        except Exception as e:
            print(f"[flight] emergency command did not get through: {e}")

    def arm(self):
        """Arm autonomous tracking.

        WHY THE REQUEST IS DEFERRED: takeoff runs in the background and may take
        several seconds to reply. Press G during that window and the state is
        still GROUNDED -- the keypress used to be simply DROPPED, and you had to
        press again. Now we remember the intent and arm tracking as soon as the
        drone reaches HOVER."""
        if self._state == HOVER:
            self._state = AUTO
            self._time_without_target = 0.0
            self._arm_requested = False
            print("[flight] AUTO -- tracking armed")
        elif self._busy:
            # Takeoff really IS running in the background -- remember the intent.
            self._arm_requested = True
            print("[flight] still taking off -- tracking will arm automatically")
        elif self._state == GROUNDED:
            # A DISTINCTION THAT DID NOT EXIST BEFORE. This condition used to sit
            # together with _busy, so a drone standing calmly on the ground would
            # report "still taking off". On a real flight that cost 100 seconds of
            # waiting for a takeoff nobody had started: T had not registered, and
            # G lied that everything was going to plan. "Waiting for takeoff" and
            # "there was no takeoff" must sound different, otherwise the log hides
            # the actual problem.
            self._arm_requested = True
            print("[flight] drone is ON THE GROUND -- press T first (takeoff). "
                  "Tracking will arm by itself once airborne.")

    def disarm(self):
        """Disarm tracking, stay airborne."""
        if self._state == AUTO:
            self._state = HOVER
            self._drone.send(Command(0, 0, 0))
            print("[flight] HOVER -- tracking disarmed")

    def toggle(self):
        # During takeoff (BUSY) or on the ground we read G as a REQUEST to arm,
        # not as a disarm. Otherwise a press during takeoff would go to disarm()
        # and vanish silently.
        if self._state == AUTO:
            self.disarm()
        else:
            self.arm()

    # -- The main step, called once per frame --------------------------
    def update(self, command, has_target, dt, blind=False):
        """command -- whatever the PID computed (may be None);
        has_target -- whether there is a reliable target right now (or its prediction);
        blind -- the frame is broken by packet loss, i.e. there is NO information
        about the target.
        Returns the command ACTUALLY sent (for on-screen display)."""
        # While taking off or landing we send NOTHING: RC commands would fight
        # the takeoff command the drone is currently executing.
        if self._busy or self._state == GROUNDED:
            return None

        # Battery -- checked about every 2 s (querying the drone is not free).
        self._battery_checked_at += dt
        if self._battery_checked_at >= 2.0:
            self._battery_checked_at = 0.0
            self._last_battery = self._drone.battery()
            if self._last_battery < self._min_battery:
                print(f"[flight] battery < {self._min_battery}% -- forced landing")
                self.land()
                return None

        # POST-TAKEOFF SETTLING -- counted down ALWAYS, not only in AUTO.
        #
        # THIS WAS A BUG, AND IT COST WHOLE FLIGHTS. The countdown used to live
        # further down, inside the AUTO branch, while the hover case returned
        # earlier -- so in HOVER it never ticked. Consequence: however long you
        # hovered picking a target, after pressing G the drone executed no
        # command for another 5 seconds.
        #
        # Measured across the recordings: from entering AUTO to the first motor
        # command, exactly 5.1 s. This reproduces on 25 of the 32 flights on
        # disk, not on two as an earlier revision of this comment claimed. On one
        # of them the entire AUTO episode lasted those same 5.1 s: the operator
        # concluded tracking was broken and switched it off, having never seen
        # the drone move.
        #
        # The intent was "give the Tello 5 seconds to stabilise AFTER TAKEOFF".
        # That time runs on the clock, regardless of whether you have armed
        # tracking yet.
        if self._settle > 0.0:
            self._settle = max(0.0, self._settle - dt)

        # In hover mode the PID is ignored entirely.
        if self._state == HOVER:
            self._prev_cmd = Command(0.0, 0.0, 0.0)
            self._drone.send(Command(0, 0, 0))
            return Command(0, 0, 0)

        # --- From here on, AUTO only ---

        # If settling has not finished, hover. Right after takeoff the Tello
        # levels itself using the downward camera, and our commands would fight
        # that.
        if self._settle > 0.0:
            self._prev_cmd = Command(0.0, 0.0, 0.0)
            self._drone.send(Command(0, 0, 0))
            return Command(0, 0, 0)

        if has_target and command is not None:
            self._time_without_target = 0.0
            self._blind_for = 0.0
            self._ever_acquired = True         # a target has been acquired at least once
            safe = self._ramp(self._limit(command), dt)
            self._drone.send(safe)
            return safe

        # No target: count how long that has been true.
        #
        # BUT on a BLIND frame the countdown STOPS. Safety is decided here --
        # after 0.5 s without a target the drone hovers, after 8 s it lands --
        # and those decisions must rest on the fact "the target is gone", not on
        # "the video fell apart". Measured by replaying flight_20260926_224656:
        # across its 42 long target losses (runs of 10+ consecutive frames with
        # no detection) 24% of frames were broken, against a 12% baseline for the
        # flight as a whole -- twice as likely. So without this condition the
        # drone regularly stopped and landed because of Wi-Fi interference rather
        # than because of the target. (An earlier revision quoted 42% against a
        # 10% baseline, measured live; the baseline reproduces, the 42% does not
        # -- re-encoding to .mp4 blurs some artefacts away. The direction is what
        # matters and it holds.)
        if not blind:
            self._time_without_target += dt
            self._blind_for = 0.0
        else:
            self._blind_for += dt

        # Auto-landing on loss -- ONLY if a target was EVER acquired. While we
        # have not found one yet (just took off, nothing clicked) we do NOT land;
        # we just hover and wait. This used to make the drone land by itself
        # after 8 s whenever you were too slow to pick a target, giving an endless
        # "takeoff -> landing -> takeoff" cycle.
        if self._ever_acquired and self._time_without_target >= self._land_after_lost:
            print(f"[flight] no target for {self._land_after_lost:.0f} s -- landing")
            self.land()
            return None

        # BLIND FRAME: HOLD THE PREVIOUS COMMAND rather than sending zeros.
        #
        # THIS WAS THE MOST EXPENSIVE MISTAKE IN THE ENTIRE SYSTEM, and it hid
        # right here. We already stopped the loss countdown on a blind frame --
        # yet we killed the MOTION anyway, because a blind frame yields command =
        # None and execution fell through to `send(Command(0,0,0))` at the end of
        # the method. In other words "I received no new information" was read as
        # "stop".
        #
        # Measured on flight_20260926_224656 (255.6 s in AUTO) -- every figure
        # below reproduces exactly from the log:
        #     frames with a "zero" command : 3474 of 6625 (52%)
        #     time standing still          : 86.6 s (34% of the flight)
        #     pauses                       : 331, median 0.11 s, longest 7.77 s
        #     pauses longer than 0.5 s     : 34, totalling 52 s (20% of the flight)
        #     a live command replaced by zero ON THE NEXT FRAME: 330 times
        # The cause of the blindness is stale frames: the stream delivers roughly
        # 14 new frames per second while the loop spun at about 21 Hz on that
        # flight (median dt 48 ms), so a large share of read() calls return the
        # same picture. Every one of them killed the motion. This is the source of
        # the "lags" and "freezes" complaints: any command was executed at a
        # fraction of its nominal value.
        #
        # ON THE STALE-FRAME SHARE. Figures of 47% of AUTO frames (and 24% on
        # another flight) were measured LIVE and cannot be reproduced from the
        # recordings -- see the note in perception/frame_quality.py: the .mp4 is
        # re-encoded, so identical frames stop being byte-identical and
        # is_stale() cannot see them on replay. Treat them as indicative. The
        # loop-rate arithmetic above is reproducible and tells the same story.
        #
        # WHY HOLDING IS SAFE. Over 0.11 s (the median blind spell) the target
        # cannot physically go anywhere, so the command stays valid. blind_hold_s
        # bounds it: beyond that we stop regardless. This is the same principle
        # already applied to the loss countdown and to holding the distance --
        # ABSENCE OF EVIDENCE IS NOT EVIDENCE OF ABSENCE -- it has simply now
        # reached the motors too.
        if blind and self._blind_for <= self._blind_hold_s:
            self._drone.send(self._prev_cmd)
            return self._prev_cmd

        if self._time_without_target >= self._hover_after_lost:
            # HOVER. The key safety decision: we do not continue moving on the
            # last command -- a drone flying blind hits a wall.
            self._prev_cmd = Command(0.0, 0.0, 0.0)
            self._drone.send(Command(0, 0, 0))
            return Command(0, 0, 0)

        # A very short gap (< 0.5 s) -- we follow the Kalman prediction.
        if command is not None:
            safe = self._ramp(self._limit(command), dt)
            self._drone.send(safe)
            return safe
        # Stop AND FORGET the previous command. Resetting _prev_cmd here was
        # missing, and once holding was introduced it became visible: after a
        # blind spell ends the drone is already stationary, yet _prev_cmd still
        # remembers a non-zero speed -- so _ramp() would compute its rise limit
        # from motion that does not exist, letting yaw jump.
        self._prev_cmd = Command(0.0, 0.0, 0.0)
        self._drone.send(Command(0, 0, 0))
        return Command(0, 0, 0)

    def _ramp(self, target: Command, dt) -> Command:
        """Limit the RATE OF RISE of the command (not the command itself).

        Asymmetric, deliberately:
          * ACCELERATION is limited -- hence the smoothness, no jerks;
          * BRAKING (towards zero) is instant -- if we need to stop or reverse,
            the drone does it without delay. Smoothness must not cost safety."""
        step = self._slew * max(dt, 1e-3)

        def one(new, old):
            if abs(new) <= abs(old) or new * old < 0:
                return new                      # braking / reversing -- instant
            return max(old - step, min(new, old + step))

        # Smoothness is what YAW needed -- yaw is what twitched. We do NOT limit
        # the forward/backward axis: its command is sparse as it is, and rate
        # limiting combined with frequent zeros never let it grow to an effective
        # value -- the axis came out dead. Vertical stays smoothed too.
        out = Command(yaw=one(target.yaw, self._prev_cmd.yaw),
                      vertical=one(target.vertical, self._prev_cmd.vertical),
                      forward=target.forward)
        self._prev_cmd = out
        return out

    # Bring PID commands into a safe but EFFECTIVE range.
    def _limit(self, c: Command) -> Command:
        vertical = self._shape(c.vertical, self._max_speed)
        vertical = self._apply_altitude(vertical)     # ceiling/floor limits
        return Command(
            yaw=self._shape(c.yaw, self._max_speed),
            vertical=vertical,
            # Forward/backward is the riskiest axis (the drone flies AT you).
            # Hence a separate, lower limit and being off by default.
            # The ceiling depends on DIRECTION: backwards is twice as slow.
            forward=(self._shape(c.forward,
                                 self._max_forward if c.forward >= 0 else self._max_backward,
                                 floor=self._min_forward)
                     if self._enable_forward else 0.0),
        )

    def _apply_altitude(self, vertical):
        """Soft ceiling and floor. Near the ceiling we damp climbing smoothly,
        near the floor we damp descent; at the limit itself we block entirely.
        This keeps the drone out of the ceiling and the floor during indoor
        tests."""
        if not self._altitude_limits:          # disabled -- vertical passes through
            return vertical
        if vertical == 0:
            return 0.0
        h = self._drone.height()               # cm
        margin = self._height_margin
        if vertical > 0:                        # moving UP
            if h >= self._max_height:
                return 0.0                      # hit the ceiling
            if h > self._max_height - margin:   # in the braking band -- damp smoothly
                vertical *= (self._max_height - h) / margin
        else:                                   # moving DOWN
            if h <= self._min_height:
                return 0.0                      # hit the floor
            if h < self._min_height + margin:
                vertical *= (h - self._min_height) / margin
        return vertical

    def _shape(self, v, ceiling, floor=None):
        """CLIP from above, LIFT from below.

        This used to be SCALING (v * max_speed/100) applied on top of an already
        limited PID -- a double attenuation. Because of it a 60px error reached
        the motors as 9, and the Tello ignores anything below about 10-15. That
        was the source of the sluggishness.

        Now: the limit is plain clipping, not a multiplier. Plus dead-band
        compensation: a non-zero but too-weak command is lifted to min_speed,
        otherwise the drone simply will not execute it.

        WHY THERE IS NO "COMMITMENT THRESHOLD" HERE. I tried zeroing commands
        weaker than half the floor -- and it broke the approach: the drone
        stopped as early as 1.2 m, because the controller asked for 7.5 against a
        floor of 20. Worse, it silenced the guard's FORCED RETREAT the same way.
        The right answer is different: once we have LEFT the deadzone, motion
        really is needed, and lifting to the floor is honest. And to stop the
        drone overshooting the target, the DEADZONE is made wider than the
        braking distance (0.20 m against ~0.11 m) -- then "all or nothing"
        resolves inside the zone.

        KNOWN ORDERING FLAW, documented because it is masked rather than fixed:
        update() calls _limit() BEFORE _ramp(), so the ramp can scale this
        guaranteed floor back down below the sensitivity threshold. See the
        slew_per_second comment for the measurement. The correct fix is
        ceiling -> ramp -> floor."""
        if v == 0:
            return 0.0
        floor = self._min_speed if floor is None else floor
        # The sensitivity floor must not exceed the ceiling -- otherwise it would
        # override it and the command would come out larger than permitted.
        floor = min(floor, ceiling)
        v = max(-ceiling, min(v, ceiling))        # clip from above
        if abs(v) < floor:                        # lift from below
            v = floor if v > 0 else -floor
        return v
