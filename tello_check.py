# Pre-flight Tello diagnostics. This script checks connectivity and video only;
# it never commands takeoff.
#
# Usage:
#   python tello_check.py           connection, battery, and temperature
#   python tello_check.py --video   also open a video window
#   python tello_check.py --sdk     also query the SDK version (Tello EDU only)
#   python tello_check.py --hq      keep 720p instead of the low-latency 480p
#   python tello_check.py --force   open video even if battery or temperature
#                                   failed the safety check

import sys
import time


def check_connection(tello):
    """Connect to the drone and verify its basic telemetry."""
    print("Connecting to 192.168.10.1:8889 ...")
    tello.connect()

    battery = tello.get_battery()
    temperature = tello.get_temperature()
    print(f"  battery:     {battery}%")
    print(f"  temperature: {temperature}°C")
    print(f"  height:      {tello.get_height()} cm")
    print("Connection established.")

    # `sdk?` is supported only by SDK 2.0 (Tello EDU), so query it on demand.
    if "--sdk" in sys.argv:
        print(f"  SDK: {tello.query_sdk_version()}")

    # Safety checks before video testing.
    ok = True
    if battery < 20:
        print(f"\n  STOP: battery is {battery}% — Tello will reject takeoff below 20%.")
        print("        Charge above 50% before a video test and 70% before flight.")
        ok = False
    elif battery < 50:
        print(f"\n  Warning: {battery}% battery is limited (about 13 min at 100%).")

    # Tello cools itself with propeller airflow and can overheat while stationary.
    if temperature >= 85:
        print(f"\n  STOP: {temperature}°C — the drone is overheating and may shut down.")
        print("        Power it off and let it cool for about 10 minutes.")
        ok = False

    return ok


def check_video(tello):
    """Verify that the live video stream is available."""
    import cv2
    from djitellopy import Tello

    # Use a lower-latency 480p stream unless --hq explicitly keeps 720p.
    if "--hq" not in sys.argv:
        for label, setter, value in (
            ("480p", tello.set_video_resolution, Tello.RESOLUTION_480P),
            ("30fps", tello.set_video_fps, Tello.FPS_30),
            ("3Mbps", tello.set_video_bitrate, Tello.BITRATE_3MBPS),
        ):
            try:
                setter(value)
                print(f"  video: {label} ok")
            except Exception as e:
                print(f"  video: {label} unsupported ({type(e).__name__})")

    tello.streamon()
    reader = tello.get_frame_read()
    print("Waiting for the first frame ...")

    # djitellopy exposes a black placeholder rather than None before video arrives.
    deadline = time.time() + 10
    while time.time() < deadline and not (reader.frame is not None and reader.frame.any()):
        time.sleep(0.05)

    if not (reader.frame is not None and reader.frame.any()):
        print("Video did not start within 10 seconds. See the guidance below.")
        tello.streamoff()
        return

    h, w = reader.frame.shape[:2]
    print(f"Video started: {w}x{h}. Press q or Esc to exit.")

    frames = 0
    t0 = time.time()
    try:
        while True:
            frame = reader.frame
            if frame is None or not frame.any():
                continue
            # Tello provides RGB frames; OpenCV expects BGR.
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frames += 1
            fps = frames / max(time.time() - t0, 1e-6)
            cv2.putText(bgr, f"{fps:.1f} FPS  {w}x{h}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("tello check", bgr)
            key = cv2.waitKey(1)
            if key != -1 and (key & 0xFF) in (ord('q'), 27):
                break
    finally:
        tello.streamoff()
        cv2.destroyAllWindows()
        print(f"Average FPS: {frames / max(time.time() - t0, 1e-6):.1f}")


def main():
    from djitellopy import Tello

    tello = Tello()
    try:
        healthy = check_connection(tello)
        if "--video" in sys.argv:
            # Avoid video testing when battery or temperature is unsafe.
            if healthy or "--force" in sys.argv:
                check_video(tello)
            else:
                print("\nVideo test skipped because of the safety checks above.")
                print("Resolve the issue, then run: python tello_check.py --video")
                print("Or force it at your own risk: ... --video --force")
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}\n")
        print("Common causes:")
        print("  1. Connect to Wi-Fi 'TELLO-XXXXXX', not the home network.")
        print("  2. Disable any active VPN.")
        print("  3. Allow UDP through System Settings → Network → Firewall.")
        print("  4. Close any previous Python process using the drone.")
        print("  5. Update Tello firmware if streamon reports 'Unknown command'.")
        sys.exit(1)
    finally:
        try:
            tello.end()
        except Exception:
            pass


if __name__ == "__main__":
    main()
