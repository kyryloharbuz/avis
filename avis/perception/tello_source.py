# Frame source for the DJI/Ryze Tello drone.
#
# THIS IS WHAT THE ABSTRACTION WAS BUILT FOR. The whole pipeline (YOLO,
# tracking, Kalman, PID, rendering) stays UNCHANGED -- main.py picks this source
# instead of the webcam and nothing downstream notices.
#
# Two Tello traps, both sealed inside this class:
#   1) COLOUR. The Tello hands back an RGB frame (djitellopy does
#      np.array(frame.to_image()), and PIL is RGB). OpenCV expects BGR
#      everywhere. Without the conversion the picture comes out blue-tinted and
#      YOLO works on distorted colour.
#   2) THE "EMPTY" FIRST FRAME. Until video arrives djitellopy holds a BLACK
#      PLACEHOLDER, np.zeros([300,400,3]) -- which is NOT None. So the naive
#      `if frame is None` check does not fire and black frames reach YOLO. Here
#      we wait for the first real frame inside the constructor.

import time

import cv2
from djitellopy import Tello

from avis.perception.frame_source import FrameSource


class TelloSource(FrameSource):
    """Tello video stream in BGR (compatible with the rest of the pipeline)."""

    # first_frame_timeout -- seconds to wait for the first real frame.
    # low_latency -- drop the resolution to 480p and lower the bitrate.
    #   WHY THIS IS FREE FOR US: the detector runs at imgsz=320, so it squeezes
    #   the frame down to 320px ANYWAY. Pushing 720p over Wi-Fi only to throw
    #   away two thirds of the pixels is pure latency. 480p gives the same
    #   detection with less data on the link: lower latency and fewer artefacts
    #   from packet loss. Turn it off (low_latency=False) when you want a
    #   good-looking picture for a demo video rather than minimum lag.
    # tello -- a READY-MADE Tello object from outside. Needed when the same
    #   drone must both serve video and accept commands: two independent
    #   Tello() instances would fight over one UDP port. If not supplied we
    #   create our own (view-only mode).
    def __init__(self, first_frame_timeout=10.0, low_latency=True, tello=None):
        t_start = time.time()
        self._tello = tello if tello is not None else Tello()

        # -- WHY STARTUP TOOK 10-15 s AND HOW IT WAS SPED UP -----------------
        # By default djitellopy waits RESPONSE_TIMEOUT=7 s for a reply and makes
        # RETRY_COUNT=3 attempts. So ONE unanswered command costs up to 21
        # seconds. And we have four sources of exactly that delay:
        #   * connect: the first UDP packet is often lost -> 7 s for nothing;
        #   * three video settings -- these are SDK 2.0 commands, and a plain
        #     Tello (not EDU) simply STAYS SILENT -> up to 21 s more each.
        # The fix: install short timeouts for the duration of init, then RESTORE
        # the normal ones before flight (so takeoff/land keep their full margin).
        #
        # ASSIGNING Tello.RESPONSE_TIMEOUT ALONE DOES NOTHING, and this comment
        # claimed otherwise for a long time. In djitellopy the signature is
        #     def send_control_command(self, command, timeout: int = RESPONSE_TIMEOUT)
        # so the default is bound ONCE, while the class body executes, to 7.
        # Rebinding the class attribute afterwards cannot reach an already-bound
        # default. Proof from a real session log: our code had just set 2.0 and
        # the drone still reported "Aborting command 'command'. Did not receive a
        # response after 7 seconds".
        # The timeout only takes effect when passed EXPLICITLY at the call site,
        # which is what the handshake below now does. We still set the class
        # attribute because RETRY_COUNT is read at runtime (self.retry_count) and
        # does work, and because any djitellopy call that does forward a timeout
        # will pick it up.
        saved_timeout, saved_retry = Tello.RESPONSE_TIMEOUT, Tello.RETRY_COUNT
        try:
            Tello.RESPONSE_TIMEOUT = 2.0
            Tello.RETRY_COUNT = 1
            # Handshake with an EXPLICIT timeout (int -- djitellopy enforces the
            # annotation, a float raises TypeError). This both honours the short
            # timeout and lets us turn an unreachable drone into a readable
            # message instead of a socket traceback.
            self._handshake(timeout=2)
            # Now that SDK mode is on, connect() is cheap: the command answers
            # immediately, so its frozen 7 s default never comes into play. We
            # still call it for the state-packet wait it performs.
            self._tello.connect()             # UDP handshake (192.168.10.1:8889)

            # Battery is a safety matter: below ~20% the Tello may refuse to fly.
            battery = self._tello.get_battery()
            print(f"[tello] battery: {battery}%")
            if battery < 20:
                print("[tello] WARNING: battery low, flight needs >20%")

            # Video settings go BEFORE streamon. They are OPTIONAL, so we give
            # them very little time and NO retries: no answer, we move on.
            if low_latency:
                Tello.RESPONSE_TIMEOUT = 0.8
                Tello.RETRY_COUNT = 1
                self._try_video_setting("480p resolution",
                                        self._tello.set_video_resolution, Tello.RESOLUTION_480P)
                self._try_video_setting("30 FPS",
                                        self._tello.set_video_fps, Tello.FPS_30)
                self._try_video_setting("3 Mbit/s bitrate",
                                        self._tello.set_video_bitrate, Tello.BITRATE_3MBPS)
        finally:
            # MANDATORY: restore the normal timeouts for flight commands.
            Tello.RESPONSE_TIMEOUT, Tello.RETRY_COUNT = saved_timeout, saved_retry

        self._tello.streamon()                        # turn the video stream on
        self._reader = self._tello.get_frame_read()   # background frame reader
        self._t_start = t_start

        # Wait for the first NON-black frame (see trap 2 above).
        deadline = time.time() + first_frame_timeout
        while time.time() < deadline:
            frame = self._reader.frame
            if frame is not None and frame.any():     # .any() = not all pixels zero
                h, w = frame.shape[:2]
                # Print the TOTAL startup time -- so progress is visible, and so
                # it is clear where the time goes if this ever gets slow again.
                print(f"[tello] video up: {w}x{h}  (startup {time.time() - self._t_start:.1f} s)")
                return
            time.sleep(0.05)

        # Gave up -- shut the stream down and say WHY (so you do not debug a
        # black screen).
        self._tello.streamoff()
        raise RuntimeError(
            f"No Tello video within {first_frame_timeout:.0f} s. "
            "Check: connected to Wi-Fi TELLO-XXXXXX, VPN off, "
            "UDP allowed by the firewall, drone firmware up to date."
        )

    def _handshake(self, timeout):
        """Enter SDK mode, turning "the drone is not there" into a readable message.

        WHY THIS EXISTS. Everything that can go wrong before the first frame goes
        wrong HERE, and djitellopy reports it as a raw traceback out of a socket
        call. Seen in practice:
          * OSError [Errno 64] Host is down -- macOS knows the route is dead.
            Means the drone is off, asleep (it sleeps after ~15 min idle), or the
            laptop is on a different Wi-Fi.
          * TelloException -- packets go out but nothing answers.
        Both say the same thing to the operator, and neither needs a stack trace
        to say it."""
        try:
            self._tello.send_control_command("command", timeout=timeout)
        except OSError as e:
            raise RuntimeError(
                f"Cannot reach the Tello at 192.168.10.1 ({e.strerror}). "
                "Check: the drone is powered on and its light is blinking, "
                "the laptop is connected to Wi-Fi TELLO-XXXXXX, VPN is off. "
                "The drone also goes to sleep after a few minutes idle -- "
                "power-cycle it."
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"The Tello is not answering at 192.168.10.1 ({type(e).__name__}). "
                "Packets are going out but there is no reply: usually the wrong "
                "Wi-Fi network, a firewall blocking UDP, or another program "
                "(the official Tello app) holding the connection."
            ) from e

    # Each setting is applied separately and tolerantly: if the firmware does
    # not know the command, the stream must still come up (just at the default
    # 720p).
    #
    # NOTE ON THE COST. These go through set_video_* helpers, which do not accept
    # a timeout, so they inherit the same frozen 7 s default described above --
    # the 0.8 s set by the caller does not reach them. On a drone that answers
    # these commands that costs nothing; on one that stays silent each one burns
    # 7 s per attempt. If startup ever feels slow again, this is where to look,
    # and the fix is to call send_control_command("setresolution 480p",
    # timeout=1) directly rather than through the helper.
    @staticmethod
    def _try_video_setting(label, setter, value):
        try:
            setter(value)
            print(f"[tello] {label}: ok")
        except Exception as e:
            print(f"[tello] {label}: unsupported ({type(e).__name__}) -- leaving as is")

    def read(self):
        frame = self._reader.frame
        if frame is None or not frame.any():
            return None                                # the stream died
        # RGB (from PIL) -> BGR (what OpenCV and our whole pipeline expect).
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def release(self):
        # Clean up after ourselves: without streamoff the drone keeps pushing
        # its UDP stream.
        try:
            self._tello.streamoff()
        finally:
            self._tello.end()
