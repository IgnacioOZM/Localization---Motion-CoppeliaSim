import math

from typing import List, Tuple
from enum import IntEnum
from random import choices

WALL_DIST_SETPOINT = 0.35
SENSOR_MAX_VALUE = 2.0
DIST_TOLERANCE = 0.003
SIDE_SAFE_DIST = 0.35
FRONT_STOPPING_DIST = 0.4
FRONT_SAFE_DIST = 0.3
SENSOR_RANGE = 1.0
LINEAR_VELOCITY_STEP = 0.05
MAX_LINEAR_VELOCITY = 1.2
ROTATING_LINEAR_VELOCITY = 0.0
ROTATING_ANGULAR_VELOCITY = 0.6


class ControlTypes(IntEnum):
    "Class to  define the different types of control."
    Stop = 0,
    FullForward = 1,
    FollowBoth = 2,
    FollowRight = 3,
    FollowLeft = 4,
    RotateRight = 5,
    RotateLeft = 6,
    Rotate180 = 7


class States(IntEnum):
    Forward = 0,
    Stop = 1,
    Explore = 2,
    Rotate = 3,
    RotateLeft = 4,
    RotateRight = 5,
    Rotate180 = 6


class My_PID():
    def __init__(self, kp, ki, kd, sample_time) -> None:

        self._Kp = kp  # Proportional gain;
        self._Kd = kd   # Derivative gain;
        self._Ki = ki   # Integral gain
        self._sample_time = sample_time

        self._e_prev = 0.0
        self._I = 0.0
        self._P = 0.0
        self._D = 0.0

    def calculate(self, error):
        if error == float('inf') or error == -1 * float('inf'):
            return 0

        self._P = self._Kp * error
        self._I += self._Ki * error * self._sample_time
        self._D = self._Kd * (error - self._e_prev) / self._sample_time

        self._e_prev = error

        return self._P + self._I + self._D

    def reset(self):
        self._e_prev = 0.0
        self._I = 0.0
        self._P = 0.0
        self._D = 0.0


class Navigation:
    """Class for short-term path planning."""

    def __init__(self, dt: float):
        """Navigation class initializer.

        Args:
            dt: Sampling period [s].

        """
        self._velocity = 0.001
        self._step_flag = True
        self._angular_velocity = 0.0

        self._action = None
        self._previous_action = None

        self._left_sensor = 0.0
        self._right_sensor = 0.0
        self._front_sensor = 0.0
        self._wall_difference = 0.0

        self._sample_time = dt
        self._state = States.Forward

        # PID control for wall following
        self._pid_pared = My_PID(
            kp=1, ki=0, kd=0.5, sample_time=self._sample_time)

        # PID control for following the left wall
        self._pid_left = My_PID(
            kp=1.5, ki=0, kd=1, sample_time=self._sample_time)

        # PID control for following the right wall
        self._pid_right = My_PID(
            kp=1.5, ki=0, kd=1, sample_time=self._sample_time)

    def explore(self, z_us: List[float], z_v: float, z_w: float) -> Tuple[float, float]:
        """Wall following exploration algorithm.

        Args:
            z_us: Distance from every ultrasonic sensor to
                  the closest obstacle [m].
            z_v: Linear velocity of the robot center [m/s].
            z_w: Angular velocity of the robot center [rad/s].

        Returns:
            v: Linear velocity [m/s].
            w: Angular velocity [rad/s].

        """

        """
        Path selection:
        - Initial startup:
              if both walls are detected -> advance between both walls
              if only one wall is detected -> follow that wall
              if no walls are detected -> forward till a wall is detected
        - Obstacle maneuver:
              if following one or both walls -> if there is no wall in front
                  turn in the direction of the wall else turn on the
                  opposite direction
              if both side walls and front wall are detected ->
                  look for empty wall or turn around.
        """

        # Get distances measurements
        self._top_left_sensor = z_us[0] if z_us[0] != float(
            'inf') else SENSOR_MAX_VALUE
        self._bottom_left_sensor = z_us[15] if z_us[15] != float(
            'inf') else SENSOR_MAX_VALUE
        self._left_sensor = (self._top_left_sensor +
                             self._bottom_left_sensor) / 2

        self._top_right_sensor = z_us[7] if z_us[7] != float(
            'inf') else SENSOR_MAX_VALUE
        self._bottom_right_sensor = z_us[8] if z_us[8] != float(
            'inf') else SENSOR_MAX_VALUE
        self._right_sensor = (self._top_right_sensor +
                              self._bottom_right_sensor) / 2

        self._front_left_sensor = z_us[3] if z_us[3] != float(
            'inf') else SENSOR_MAX_VALUE
        self._front_right_sensor = z_us[4] if z_us[4] != float(
            'inf') else SENSOR_MAX_VALUE
        self._front_sensor = (self._front_left_sensor +
                              self._front_right_sensor) / 2

        if self._state == States.Forward:
            print('Forward!!!             ', end='\n')
            if self._front_sensor > SENSOR_RANGE:
                if self._left_sensor < SENSOR_RANGE and self._right_sensor < SENSOR_RANGE:
                    self._follow_both()
                elif self._left_sensor < SENSOR_RANGE and self._right_sensor > SENSOR_RANGE:
                    self._follow_left()
                elif self._right_sensor < SENSOR_RANGE and self._left_sensor > SENSOR_RANGE:
                    self._follow_right()
                else:
                    self._forward()
            else:
                self._state = States.Stop

        elif self._state == States.Stop:
            print('Stopping                ', end='\n')
            if self._front_sensor > SENSOR_RANGE:
                self._state = States.Forward
            elif SENSOR_RANGE >= self._front_sensor > FRONT_STOPPING_DIST:
                self._stop()
            elif FRONT_STOPPING_DIST >= self._front_sensor > FRONT_SAFE_DIST:
                self._state = States.Explore
            else:
                print('Backwards!!!           ', end='\n')
                self._backwards()

        elif self._state == States.Explore:
            if self._front_sensor <= SENSOR_RANGE:
                # Path selection
                if self._left_sensor >= SENSOR_RANGE and self._right_sensor >= SENSOR_RANGE:
                    if self._previous_action == ControlTypes.FollowLeft:
                        self._state = States.RotateLeft
                    elif self._previous_action == ControlTypes.FollowRight:
                        self._state = States.RotateRight
                    else:
                        # When the robot is in an open space we choose randomly which side turn to
                        self._state = choices(
                            [States.RotateLeft, States.RotateRight], [0.5, 0.5])[0]
                elif self._left_sensor > SENSOR_RANGE:
                    self._state = States.RotateLeft
                elif self._right_sensor > SENSOR_RANGE:
                    self._state = States.RotateRight
                else:
                    self._state = States.Rotate180
            else:
                self._state = States.Forward

        elif self._state == States.RotateLeft:
            print('Rotate left...       ', end='\n')
            print(f'#### Front Sensor: {self._front_sensor}', end='\n')
            print(
                f'#### Angle: {math.degrees(self._right_wall_angle())}', end='\n')
            angle = abs(math.degrees(self._right_wall_angle()))
            if self._front_sensor >= SENSOR_MAX_VALUE and angle <= 5.0:
                self._state = States.Forward
            elif self._front_sensor <= FRONT_SAFE_DIST:
                self._state = States.Stop
            else:
                self._rotate_left()

        elif self._state == States.RotateRight:
            print('Rotate right...      ', end='\n')
            print(f'#### Front Sensor: {self._front_sensor}', end='\n')
            print(
                f'#### Angle: {math.degrees(self._left_wall_angle())}', end='\n')

            angle = abs(math.degrees(self._left_wall_angle()))
            if self._front_sensor >= SENSOR_MAX_VALUE and angle <= 5.0:
                self._state = States.Forward
            elif self._front_sensor >= FRONT_SAFE_DIST:
                self._rotate_right()
            else:
                self._state = States.Stop

        elif self._state == States.Rotate180:
            print('Rotate 180...        ', end='\n')
            if self._front_sensor >= SENSOR_MAX_VALUE:
                left_angle = abs(math.degrees(self._left_wall_angle()))
                right_angle = abs(math.degrees(self._right_wall_angle()))

                if left_angle <= 5.0 or right_angle <= 5:
                    self._state = States.Forward
                else:
                    self._rotate_180()

            elif self._left_sensor <= SENSOR_RANGE and self._right_sensor <= SENSOR_RANGE:
                self._rotate_180()

            elif self._left_sensor > SENSOR_RANGE:
                self._rotate_left()

            elif self._right_sensor > SENSOR_RANGE:
                self._rotate_right()

        self._previous_action = self._action

        return self._velocity, self._angular_velocity

    def _stop(self):
        self._velocity = 0.3
        self._angular_velocity = 0.0

    def _forward(self):
        self._velocity_step()
        self._angular_velocity = 0.0

    def _backwards(self):
        # Velocity step
        self._velocity = -1 * MAX_LINEAR_VELOCITY / 3.0
        self._angular_velocity = 0.0

    def _follow_both(self):
        print('Following both walls...', end='\n')
        # PID Controller
        wall_difference = self._right_sensor - self._left_sensor
        self._angular_velocity = self._pid_pared.calculate(
            -1 * wall_difference)

        self._velocity_step()

    def _right_wall_angle(self):
        angle = math.atan((self._bottom_right_sensor -
                          self._top_right_sensor) / 0.2167)
        return angle

    def _left_wall_angle(self):
        angle = math.atan((self._bottom_left_sensor -
                          self._top_left_sensor) / 0.2167)
        return angle

    def _follow_left(self):
        print('Following left wall... ', end='\n')
        # PID Controller
        angle = self._left_wall_angle()
        dist_to_wall = ((self._bottom_left_sensor +
                        self._top_left_sensor) / 2) * math.cos(angle)

        self._angular_velocity = -1 * self._pid_left.calculate(
            WALL_DIST_SETPOINT - dist_to_wall)

        self._velocity_step()

    def _follow_right(self):
        print('Following right wall...', end='\n')
        # PID Controller
        angle = self._right_wall_angle()
        dist_to_wall = ((self._bottom_right_sensor +
                        self._top_right_sensor) / 2) * math.cos(angle)

        self._angular_velocity = self._pid_right.calculate(
            WALL_DIST_SETPOINT - dist_to_wall)

        self._velocity_step()

    def _rotate_right(self):
        self._action = ControlTypes.RotateRight
        self._velocity = ROTATING_LINEAR_VELOCITY
        self._angular_velocity = -1 * ROTATING_ANGULAR_VELOCITY

    def _rotate_left(self):
        self._action = ControlTypes.RotateLeft
        self._velocity = ROTATING_LINEAR_VELOCITY
        self._angular_velocity = ROTATING_ANGULAR_VELOCITY

    def _rotate_180(self):
        self._action = ControlTypes.Rotate180
        self._velocity = 0.0
        self._angular_velocity = ROTATING_ANGULAR_VELOCITY

    def _velocity_step(self):
        if self._velocity < MAX_LINEAR_VELOCITY:
            if self._step_flag:
                self._velocity += LINEAR_VELOCITY_STEP / 4
                if self._velocity >= MAX_LINEAR_VELOCITY * 0.35:
                    self._step_flag = False
            else:
                self._velocity += LINEAR_VELOCITY_STEP
        else:
            self._velocity = MAX_LINEAR_VELOCITY
