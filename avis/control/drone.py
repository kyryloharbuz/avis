# The actuator: the thing that ACCEPTS control commands.
#
# Three implementations justify the interface: TelloDrone flies, NullDrone
# prints and keeps a simulated altitude, SimDrone starts that altitude wherever
# a test needs it. The stubs are what make the autonomous logic debuggable
# without risking the aircraft.

from abc import ABC, abstractmethod

from avis.models import Command


class Drone(ABC):
    """The drone contract: take off, land, accept a velocity command."""

    @abstractmethod
    def takeoff(self):
        """Take off (blocks until altitude is reached)."""

    @abstractmethod
    def land(self):
        """Soft landing."""

    @abstractmethod
    def emergency(self):
        """EMERGENCY: cut the motors instantly (the drone falls)."""

    @abstractmethod
    def send(self, command: Command):
        """Send motion velocities. Zeros = hover in place."""

    @abstractmethod
    def battery(self) -> int:
        """Charge in percent."""

    @abstractmethod
    def height(self) -> int:
        """Current altitude in CENTIMETRES (for the ceiling/floor limits)."""


class NullDrone(Drone):
    """A stub: nothing flies. Takeoff/landing print, and the vertical command is
    integrated into a simulated altitude so the ceiling and floor limits can be
    tested without an aircraft."""

    def __init__(self):
        self._height = 100.0     # cm -- simulated altitude (we start at 1 m)

    def takeoff(self):
        print("[drone] TAKEOFF (simulated)")

    def land(self):
        print("[drone] LAND (simulated)")

    def emergency(self):
        print("[drone] EMERGENCY (simulated)")

    def send(self, command: Command):
        # Loosely "integrate" altitude from the vertical velocity, so the
        # ceiling/floor limits can be covered by tests WITHOUT a real drone.
        self._height += command.vertical * 0.05

    def battery(self) -> int:
        return 100

    def height(self) -> int:
        return self._height


class SimDrone(NullDrone):
    """NullDrone at a given altitude -- handy for altitude-limit tests."""

    def __init__(self, height=100.0):
        super().__init__()
        self._height = height


class TelloDrone(Drone):
    """The real Tello.

    IMPORTANT: it takes a READY-MADE Tello object rather than creating its own.
    The reason is that video and commands both go through one UDP socket: two
    independent Tello objects would fight over the port. So in main.py the drone
    is created ONCE and passed both to TelloSource and here."""

    def __init__(self, tello):
        self._tello = tello

    def takeoff(self):
        self._tello.takeoff()

    def land(self):
        self._tello.send_rc_control(0, 0, 0, 0)   # stop the motion first
        self._tello.land()

    def emergency(self):
        self._tello.emergency()

    def send(self, command: Command):
        # Our Command -> four Tello channels. left_right is unused: we bring the
        # target to centre by yawing, the way a human operator does.
        # round(), not int(): int truncates towards zero (12.9 -> 12), i.e. it
        # systematically ate up to one unit of speed on every axis. round() gives
        # an honest value.
        #
        # THE MAPPING IS WORTH DOUBLE-CHECKING IF ALTITUDE MISBEHAVES, since a
        # swapped channel here would look exactly like a control bug. The
        # signature is send_rc_control(left_right, forward_backward, up_down,
        # yaw), and the assignment below has been verified against it.
        self._tello.send_rc_control(
            0,                          # left_right (lateral) -- unused
            round(command.forward),     # forward_backward -- distance
            round(command.vertical),    # up_down -- altitude
            round(command.yaw),         # yaw -- rotation
        )

    def battery(self) -> int:
        return self._tello.get_battery()

    def height(self) -> int:
        # get_height() reads CACHED state (the Tello pushes it continuously on a
        # separate thread), so this is not a UDP round trip and calling it every
        # frame is cheap. In cm.
        return self._tello.get_height()
