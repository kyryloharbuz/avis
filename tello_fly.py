# Manual Tello control with video. This utility has no autonomous behaviour.
#
# Run: .venv/bin/python tello_fly.py
#
# Controls: T takeoff, L land, W/S forward/backward, A/D left/right,
# Q/E yaw, R/F up/down, X emergency motor stop, Esc land and exit.
#
# Use a clear indoor area, install propeller guards, keep battery above 50%,
# and maintain a direct landing path. Never fly over people or animals.

import select
import sys
import termios
import time
import tty

import cv2
from djitellopy import Tello


class TerminalKeys:
    """Read individual keys from the terminal without requiring Enter.

    Terminal input avoids macOS OpenCV-window focus issues. The terminal uses
    cbreak mode and non-blocking ``select`` reads.
    """

    def __init__(self):
        self._fd = None
        self._old = None
        # IDE run windows may not expose a terminal; OpenCV input still works.
        if sys.stdin.isatty():
            self._fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)

    @property
    def active(self):
        return self._fd is not None

    def get(self):
        """Return the pressed character, or None when no input is available."""
        if self._fd is None:
            return None
        # A zero timeout returns immediately.
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    def restore(self):
        """Restore the terminal mode."""
        if self._fd is not None and self._old is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

SPEED = 35           # horizontal speed (-100..100)
SPEED_VERTICAL = 40
SPEED_YAW = 60


def overlay(frame, tello, flying, last_key):
    """Draw flight telemetry over the video frame."""
    state = "AIRBORNE" if flying else "GROUNDED"
    color = (0, 165, 255) if flying else (0, 255, 0)
    cv2.putText(frame, state, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    try:
        info = f"bat {tello.get_battery()}%  h {tello.get_height()}cm"
    except Exception:
        info = "telemetry unavailable"
    cv2.putText(frame, info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, "T=takeoff  L=land  WASD/QE/RF  X=emergency", (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    if last_key:
        cv2.putText(frame, last_key, (frame.shape[1] - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def main():
    tello = Tello()
    tello.connect()

    battery = tello.get_battery()
    print(f"Battery: {battery}%")
    if battery < 30:
        print("STOP: flight requires at least 30% battery; 50% or more is recommended.")
        sys.exit(1)

    # 480p reduces latency, which matters more than image quality for manual control.
    for setter, value in ((tello.set_video_resolution, Tello.RESOLUTION_480P),
                          (tello.set_video_fps, Tello.FPS_30)):
        try:
            setter(value)
        except Exception:
            pass

    tello.streamon()
    reader = tello.get_frame_read()

    # Wait for a real frame; the initial placeholder is black rather than None.
    deadline = time.time() + 10
    while time.time() < deadline and not (reader.frame is not None and reader.frame.any()):
        time.sleep(0.05)

    flying = False
    last_key = ""
    keys = TerminalKeys()          # terminal input avoids macOS focus issues

    print("\n" + "=" * 58)
    if keys.active:
        print("Controls are active in this terminal.")
        print("Keep focus here and press the keys directly.")
    else:
        print("Terminal input is unavailable; click the video window to control it.")
    print("  T — takeoff     L — land     Esc or Ctrl+C — exit")
    print("  W/S forward/backward   A/D lateral   Q/E yaw   R/F up/down")
    print("=" * 58 + "\n")

    try:
        while True:
            frame = reader.frame
            if frame is None or not frame.any():
                continue
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            overlay(bgr, tello, flying, last_key)
            cv2.imshow("tello manual", bgr)

            # Accept input from either the OpenCV window or the terminal.
            key = cv2.waitKey(1) & 0xFF
            typed = keys.get()
            if typed:
                key = ord(typed.lower()) if typed not in ('\x1b', '\x03') else 27

            # Zero commands hold position when no movement key is pressed.
            lr = fb = ud = yaw = 0

            if key == ord('t') and not flying:
                print("Taking off ...")
                tello.takeoff()
                flying = True
                last_key = "TAKEOFF"
            elif key == ord('l') and flying:
                print("Landing ...")
                tello.send_rc_control(0, 0, 0, 0)   # stop before landing
                tello.land()
                flying = False
                last_key = "LAND"
            elif key == ord('x'):
                # Emergency stops motors immediately; use only when landing is impossible.
                print("EMERGENCY MOTOR STOP")
                tello.emergency()
                flying = False
                last_key = "EMERGENCY"
            elif key == 27:                          # Esc
                break
            elif flying:
                # Movement is available only while airborne.
                if key == ord('w'): fb, last_key = SPEED, "FORWARD"
                elif key == ord('s'): fb, last_key = -SPEED, "BACKWARD"
                elif key == ord('a'): lr, last_key = -SPEED, "LEFT"
                elif key == ord('d'): lr, last_key = SPEED, "RIGHT"
                elif key == ord('r'): ud, last_key = SPEED_VERTICAL, "UP"
                elif key == ord('f'): ud, last_key = -SPEED_VERTICAL, "DOWN"
                elif key == ord('q'): yaw, last_key = -SPEED_YAW, "YAW LEFT"
                elif key == ord('e'): yaw, last_key = SPEED_YAW, "YAW RIGHT"
                elif key == 255: last_key = ""       # no input

            # Send RC commands every frame; zero values command a hover.
            if flying:
                tello.send_rc_control(lr, fb, ud, yaw)

            if cv2.getWindowProperty("tello manual", cv2.WND_PROP_VISIBLE) < 1:
                break

    except KeyboardInterrupt:
        print("\nCtrl+C — landing the drone ...")
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
    finally:
        keys.restore()
        # Always attempt a safe landing on shutdown.
        try:
            if flying:
                print("Emergency shutdown — landing the drone ...")
                tello.send_rc_control(0, 0, 0, 0)
                tello.land()
        except Exception:
            pass
        try:
            tello.streamoff()
            tello.end()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
