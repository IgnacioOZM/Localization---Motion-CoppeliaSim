import numpy as np
import sim

from robot import Robot
from typing import Any, Dict, List, Tuple


class RobotP3DX(Robot):
    """Class to control the Pioneer 3-DX robot."""

    # Constants
    SENSOR_RANGE = 1.0     # Ultrasonic sensor range [m]
    TRACK = 0.33           # Distance between same axle wheels [m]
    WHEEL_RADIUS = 0.0975  # Radius of the wheels [m]

    # Sensor location and orientation (x, y, theta) in
    # the robot coordinate frame
    SENSORS = [(0.1067, 0.1382, 1.5708),
               (0.1557, 0.1250, 0.8727),
               (0.1909, 0.0831, 0.5236),
               (0.2095, 0.0273, 0.1745),
               (0.2095, -0.0273, -0.1745),
               (0.1909, -0.0785, -0.5236),
               (0.1558, -0.1203, -0.8727),
               (0.1067, -0.1382, -1.5708),
               (-0.1100, -0.1382, -1.5708),
               (-0.1593, -0.1203, -2.2689),
               (-0.1943, -0.0785, -2.6180),
               (-0.2129, -0.0273, -2.9671),
               (-0.2129, 0.0273, 2.9671),
               (-0.1943, 0.0785, 2.6180),
               (-0.1593, 0.1203, 2.2689),
               (-0.1100, 0.1382, 1.5708)]

    def __init__(self, client_id: int, dt: float):
        """Pioneer 3-DX robot class initializer.

        Args:
            client_id: CoppeliaSim connection handle.
            dt: Sampling period [s].

        """
        Robot.__init__(self, client_id, track=self.TRACK, wheel_radius=self.WHEEL_RADIUS)
        self._dt = dt
        self._motors = self._init_motors()
        self._sensors = self._init_sensors()
        self._init_encoders()

    def move(self, v: float, w: float):
        """Solve inverse differential kinematics and send commands
            to the motors.

        Args:
            v: Linear velocity of the robot center [m/s].
            w: Angular velocity of the robot center [rad/s].

        """
        # Define motor speeds
        left_wheel = (v - (RobotP3DX.TRACK / 2) * w) / RobotP3DX.WHEEL_RADIUS
        right_wheel = (v + (RobotP3DX.TRACK / 2) * w) / RobotP3DX.WHEEL_RADIUS

        # Set motors speeds
        rc = sim.simxSetJointTargetVelocity(self._client_id, self._motors['left'], left_wheel, sim.simx_opmode_oneshot)
        rc = sim.simxSetJointTargetVelocity(
            self._client_id, self._motors['left'], left_wheel,
            sim.simx_opmode_oneshot)
        rc = sim.simxSetJointTargetVelocity(
            self._client_id, self._motors['right'], right_wheel,
            sim.simx_opmode_oneshot)

    def sense(self) -> Tuple[List[float], float, float]:
        """Read ultrasonic sensors and encoders.

        Returns:
            z_us: Distance from every ultrasonic sensor to the
                closest obstacle [m].
            z_v: Linear velocity of the robot center [m/s].
            z_w: Angular velocity of the robot center [rad/s].

        """
        # Read ultrasonic sensors
        z_us = [float('inf')] * len(self.SENSORS)

        for sensor in self._sensors:
            _, is_valid, detected_point, _, _ = sim.simxReadProximitySensor(
                self._client_id, sensor, sim.simx_opmode_buffer)

            if is_valid:
                distance = np.linalg.norm(detected_point)
                z_us[self._sensors.index(sensor)] = distance
            # Id the meassure is hogher than 1.0 m the meassure is not valid.
            # else:
                # Give the sensor a value of > than 1 because it measures
                # inf when not valid.
                # z_us[self._sensors.index(sensor)] = 10

        # Read encoders
        z_v, z_w = self._sense_encoders()

        return z_us, z_v, z_w

    def _init_encoders(self):
        """Initialize encoder streaming."""
        for i in range(3):
            sim.simxGetFloatSignal(self._client_id, 'leftEncoder', sim.simx_opmode_streaming)
            sim.simxGetFloatSignal(self._client_id, 'rightEncoder', sim.simx_opmode_streaming)

    def _init_motors(self) -> Dict[str, int]:
        """Acquire motor handles.

        Returns: {'left': handle, 'right': handle}

        """
        motors = {'left': None, 'right': None}
        motors_name = ['Pioneer_p3dx_leftMotor', 'Pioneer_p3dx_rightMotor']

        # Acquire handles (in _init_motors)
        rc, motors['left'] = sim.simxGetObjectHandle(self._client_id, motors_name[0], sim.simx_opmode_blocking)
        rc, motors['right'] = sim.simxGetObjectHandle(self._client_id, motors_name[1], sim.simx_opmode_blocking)

        return motors

    def _init_sensors(self) -> List[Any]:
        """Acquire ultrasonic sensor handles and initialize US
            and encoder streaming.

        Returns: List with ultrasonic sensor handles.

        """
        sensors = [None] * len(self.SENSORS)
        sensors_name = 'Pioneer_p3dx_ultrasonicSensor'

        # TODO: Complete with your code.
        for i in range(1, 17):
            handle_name = sensors_name + str(i)
            _, sensors[i - 1] = sim.simxGetObjectHandle(self._client_id, handle_name, sim.simx_opmode_blocking)
            sim.simxReadProximitySensor(self._client_id, sensors[i - 1], sim.simx_opmode_streaming)

        return sensors

    def _sense_encoders(self) -> Tuple[float, float]:
        """Solve forward differential kinematics from encoder readings.

        Returns:
            z_v: Linear velocity of the robot center [m/s].
            z_w: Angular velocity of the robot center [rad/s].

        """
        # TODO: Complete with your code.
        z_v = 0.0
        z_w = 0.0

        # Read encoder rotation
        _, left_value = sim.simxGetFloatSignal(self._client_id, "leftEncoder", sim.simx_opmode_buffer)
        _, right_value = sim.simxGetFloatSignal(self._client_id, "rightEncoder", sim.simx_opmode_buffer)

        z_v = (right_value + left_value) * RobotP3DX.WHEEL_RADIUS / 2
        z_w = (right_value - left_value) * \
            RobotP3DX.WHEEL_RADIUS / (2 * RobotP3DX.TRACK)

        return z_v, z_w
