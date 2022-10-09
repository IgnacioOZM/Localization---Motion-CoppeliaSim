from abc import ABC, abstractmethod


class Robot(ABC):
    """Abstract base class to control mobile robots."""

    def __init__(self, client_id: int, track: float, wheel_radius: float):
        """Robot class initializer.

        Args:
            client_id: CoppeliaSim connection handle.
            track: Distance between the centerline of two wheels on the same axle [m].
            wheel_radius: Radius of the wheels [m].

        """
        self._client_id = client_id
        self._track = track
        self._wheel_radius = wheel_radius

    @abstractmethod
    def move(self, v: float, w: float):
        """Solve inverse kinematics and send commands to the motors.

        Args:
            v: Linear velocity of the robot center [m/s].
            w: Angular velocity of the robot center [rad/s].

        """
        pass

    @abstractmethod
    def sense(self):
        """Acquire sensor readings."""
        pass
