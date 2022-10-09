from cmath import pi, sqrt
import math
from re import X
from threading import local
import numpy as np
import os
import random

from map import Map
from matplotlib import pyplot as plt
from typing import List, Tuple

from robot_p3dx import RobotP3DX
from sklearn.cluster import DBSCAN


class ParticleFilter:
    """Particle filter implementation."""

    def __init__(self, map_object: Map, sensors: List[Tuple[float, float, float]],
                 sensor_range: float, particle_count: int = 100, sense_noise: float = 0.5,
                 v_noise: float = 0.05, w_noise: float = 0.05, figure_size: Tuple[float, float] = (7, 7)):
        """Particle filter class initializer.

        Args:
            map_object: Map of the environment.
            sensors: Robot sensors location [m] and orientation [rad] in the robot coordinate frame (x, y, theta).
            sensor_range: Sensor measurement range [m]
            particle_count: Number of particles.
            sense_noise: Measurement standard deviation [m].
            v_noise: Linear velocity standard deviation [m/s].
            w_noise: Angular velocity standard deviation [rad/s].
            figure_size: Figure window dimensions.

        """
        self._map = map_object
        self._sensors = sensors
        self._sense_noise = sense_noise
        self._sensor_range = sensor_range
        self._v_noise = v_noise
        self._w_noise = w_noise
        self._iteration = 0

        # Systematic Resampling Method
        self._rng = np.random.default_rng()
        self._minimum_number_of_particles = 20

        self._particles = self._init_particles(particle_count)
        self._ds, self._phi = self._init_sensor_polar_coordinates(sensors)
        self._figure, self._axes = plt.subplots(1, 1, figsize=figure_size)

    def move(self, v: float, w: float, dt: float):
        """Performs a motion update on the particles.

        Args:
            v: Linear velocity [m/s].
            w: Angular velocity [rad/s].
            dt: Sampling time [s].

        """
        self._iteration += 1

        # x = x + v * cos(th)  * dt
        # y = y + v * sin(th) * dt
        # th = th + w * dt

        new_particles = np.zeros(self._particles.shape, dtype=object)
        v_noise = np.random.normal(
            loc=v, scale=self._v_noise, size=self._particles.shape[0])
        w_noise = np.random.normal(
            loc=w, scale=self._w_noise, size=self._particles.shape[0])

        x_array = self._particles[:, 0].astype(np.float32)
        y_array = self._particles[:, 1].astype(np.float32)
        th_array = self._particles[:, 2].astype(np.float32)

        x_array = x_array + v_noise * np.cos(th_array) * dt
        y_array = y_array + v_noise * np.sin(th_array) * dt
        th_array = th_array + w_noise * dt
        th_array = np.arctan2(np.sin(th_array), np.cos(th_array))

        new_particles[:, 0] = x_array
        new_particles[:, 1] = y_array
        new_particles[:, 2] = th_array

        for i in range(0, self._particles.shape[0]):
            intersection, _ = self._map.check_collision(
                segment=[(self._particles[i, 0], self._particles[i, 1]),
                         (new_particles[i, 0], new_particles[i, 1])],
                compute_distance=False)
            if intersection:
                new_particles[i, 0] = intersection[0]
                new_particles[i, 1] = intersection[1]

        self._particles = new_particles

    def resample(self, measurements: List[float]):
        """Samples a new set of particles using the resampling wheel method.

        Args:
            measurements: Sensor measurements [m].

        """

        particle_probability = np.zeros(len(self._particles), dtype=object)
        new_particles = np.zeros(self._particles.shape, dtype=object)

        for i in range(0, len(self._particles)):
            particle_probability[i] = self._measurement_probability(
                measurements=measurements, particle=self._particles[i])

        n = len(self._particles)
        index = random.randint(0, n - 1)
        beta = 0.0
        w_max = max(particle_probability)

        for i in range(n):
            beta += random.uniform(0.0, 2.0 * w_max)

            while beta > particle_probability[index]:
                beta -= particle_probability[index]
                index = (index + 1) % n

            x = self._particles[index][0]
            y = self._particles[index][1]
            theta = self._particles[index][2]
            new_particles[i] = [x, y, theta]

        self._particles = new_particles

    def resample_v2(self, measurements: List[float]):
        """Samples a new set of particles using the resampling wheel method.

        Args:
            measurements: Sensor measurements [m].

        """
        particle_probability = np.zeros(len(self._particles), dtype=object)
        new_particles = np.zeros(self._particles.shape, dtype=object)

        for i in range(0, len(self._particles)):
            particle_probability[i] = self._measurement_probability(
                measurements=measurements, particle=self._particles[i])
        particle_weights = particle_probability / sum(particle_probability)

        neff = self.neff(particle_weights)
        n = len(self._particles)

        if(neff < n):
            if n > self._minimum_number_of_particles:
                n = max(n - 50, self._minimum_number_of_particles)
            new_particles = np.zeros((n, 3), dtype=object)

            # Systematic Resampling Method
            positions = (self._rng.random() + np.arange(n)) / n
            cumulative_sum = np.cumsum(particle_weights)
            i, j = 0, 0
            while i < n:
                if positions[i] < cumulative_sum[j]:
                    new_particles[i] = self._particles[j]
                    i += 1
                else:
                    j += 1

            self._check, centroid = self.cluster_estimate()
            if self._check:
                self._avg_state = centroid

            self._particles = new_particles

    def neff(self, weights):
        return 1. / np.sum(np.square(weights))

    def cluster_estimate(self):
        self._db = DBSCAN(eps=1, min_samples=3).fit(self._particles)
        labels = self._db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        if n_clusters == 1:
            centroid = np.zeros(3)
            for particle, label in zip(self._particles, labels):
                if label == 0:
                    centroid[0] += particle[0]
                    centroid[1] += particle[1]
                    centroid[2] += particle[2]

            centroid /= len(self._particles)
            return True, centroid
        else:
            return False, []

    def plot(self, axes, orientation: bool = True):
        """Draws particles.

        Args:
            axes: Figure axes.
            orientation: Draw particle orientation.

        Returns:
            axes: Modified axes.

        """
        if orientation:
            dx = [math.cos(particle[2]) for particle in self._particles]
            dy = [math.sin(particle[2]) for particle in self._particles]
            axes.quiver(self._particles[:, 0], self._particles[:, 1],
                        dx, dy, color='b', scale=15, scale_units='inches')
        else:
            axes.plot(self._particles[:, 0],
                      self._particles[:, 1], 'bo', markersize=1)

        return axes

    def show(self, title: str = '', orientation: bool = True, display: bool = True,
             save_figure: bool = False, save_dir: str = 'img'):
        """Displays the current particle set on the map.

        Args:
            title: Plot title.
            orientation: Draw particle orientation.
            display: True to open a window to visualize the particle filter evolution in real-time. Time consuming.
            save_figure: True to save figure to a .png file.
            save_dir: Image save directory.

        """
        figure = self._figure
        axes = self._axes
        axes.clear()

        axes = self._map.plot(axes)
        axes = self.plot(axes, orientation)

        axes.set_title(title + ' (Iteration #' + str(self._iteration) + ')')
        figure.tight_layout()  # Reduce white margins

        if display:
            plt.show(block=False)
            # Wait for 0.1 ms or the figure won't be displayed
            plt.pause(0.001)

        if save_figure:
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)

            file_name = str(self._iteration).zfill(
                4) + ' ' + title.lower() + '.jpg'
            file_path = os.path.join(save_dir, file_name)
            figure.savefig(file_path)

    def _init_particles(self, particle_count: int) -> np.ndarray:
        """Draws N random valid particles.

        The particles are guaranteed to be inside the map and
        can only have the following orientations [0, pi/2, pi, 3pi/2].

        Args:
            particle_count: Number of particles.

        Returns: A numpy array of tuples (x, y, theta).

        """
        particles = np.zeros((particle_count, 3), dtype=object)

        for particle in particles:
            x = np.random.uniform(low=self._map.bounds()[0], high=self._map.bounds()[2])
            y = np.random.uniform(low=self._map.bounds()[1], high=self._map.bounds()[3])

            while not self._map.contains(point=(x, y)):
                x = np.random.uniform(low=self._map.bounds()[0], high=self._map.bounds()[2])
                y = np.random.uniform(low=self._map.bounds()[1], high=self._map.bounds()[3])

            particle[0] = x
            particle[1] = y
            particle[2] = random.choices([0, math.pi, math.pi / 2, 3 * math.pi / 2], [0.25, 0.25, 0.25, 0.25])[0]

        return particles

    @staticmethod
    def _init_sensor_polar_coordinates(sensors: List[Tuple[float, float, float]]) -> Tuple[List[float], List[float]]:
        """Converts the robots sensor location and orientation to polar coordinates wrt to the robot's coordinate frame.

        Args:
            sensors: Robot sensors location [m] and orientation [rad] (x, y, theta).

        Return:
            ds: List of magnitudes [m].
            phi: List of angles [rad].

        """
        ds = [math.sqrt(sensor[0] ** 2 + sensor[1] ** 2) for sensor in sensors]
        phi = [math.atan2(sensor[1], sensor[0]) for sensor in sensors]

        return ds, phi

    def _sense(self, particle: Tuple[float, float, float]) -> List[float]:
        """Obtains the predicted measurement of every sensor given the robot's location.

        Args:
            particle: Particle pose (x, y, theta) in [m] and [rad].

        Returns: List of predicted measurements; inf if a sensor is out of range.

        """
        rays = self._sensor_rays(particle)

        z_hat = []

        for ray in rays:
            _, distance = self._map.check_collision(ray, compute_distance=True)
            z_hat.append(distance)

        return z_hat

    @staticmethod
    def _gaussian(mu: float, sigma: float, x: float) -> float:
        """Computes the value of a Gaussian.

        Args:
            mu: Mean.
            sigma: Standard deviation.
            x: Variable.

        Returns:
            float: Gaussian.

        """

        # prob = (1 / (sqrt(2 * math.pi * (sigma ** 2)))) * math.exp(-0.5 * ((x - mu)**2) / (sigma ** 2))
        # The normalization factor is disposbale in this case
        return math.exp(-0.5 * (x - mu / sigma) ** 2)

    def _measurement_probability(self, measurements: List[float], particle: Tuple[float, float, float]) -> float:
        """Computes the probability of a set of measurements given a particle's pose.

        If a measurement is unavailable (usually because it is out of range), it is replaced with twice the sensor range
        to perform the computation. This value has experimentally been proven valid to deal with missing measurements.
        Nevertheless, it might not be the optimal replacement value.

        Args:
            measurements: Sensor measurements [m].
            particle: Particle pose (x, y, theta) in [m] and [rad].

        Returns:
            float: Probability.

        """

        particle_measurements = self._sense(particle=particle)
        prob = 1

        for real_measurement, particle_measurement in zip(measurements, particle_measurements):
            if real_measurement > self._sensor_range:
                real_measurement = self._sensor_range * 2

            if particle_measurement > self._sensor_range:
                particle_measurement = self._sensor_range * 2

            prob *= self._gaussian(mu=particle_measurement,
                                   sigma=self._sense_noise, x=real_measurement)
        return prob

    def _sensor_rays(self, particle: Tuple[float, float, float]) -> List[List[Tuple[float, float]]]:
        """Determines the simulated sensor ray segments for a given particle.

        Args:
            particle: Particle pose (x, y, theta) in [m] and [rad].

        Returns: Ray segments.
                 Format: [[(x0_begin, y0_begin), (x0_end, y0_end)], [(x1_begin, y1_begin), (x1_end, y1_end)], ...]

        """
        x = particle[0]
        y = particle[1]
        theta = particle[2]

        # Convert sensors to world coordinates
        xw = [x + ds * math.cos(theta + phi)
              for ds, phi in zip(self._ds, self._phi)]
        yw = [y + ds * math.sin(theta + phi)
              for ds, phi in zip(self._ds, self._phi)]
        tw = [sensor[2] for sensor in self._sensors]

        rays = []

        for xs, ys, ts in zip(xw, yw, tw):
            x_end = xs + self._sensor_range * math.cos(theta + ts)
            y_end = ys + self._sensor_range * math.sin(theta + ts)
            rays.append([(xs, ys), (x_end, y_end)])

        return rays


def test():
    """Function used to test the ParticleFilter class independently."""
    import time
    from robot_p3dx import RobotP3DX

    # Measurements from sensors 1 to 8 [m]
    measurements = [
        (0.3618, 0.4895, 0.8337, float('inf'), float(
            'inf'), 0.9343, float('inf'), float('inf')),
        (0.3618, 0.4895, 0.8337, float('inf'), float(
            'inf'), 0.8430, float('inf'), float('inf')),
        (0.3618, 0.4895, 0.8337, float('inf'), float(
            'inf'), 0.8430, float('inf'), float('inf')),
        (0.3618, 0.4895, 0.8337, float('inf'), float(
            'inf'), 0.8430, 0.8582, float('inf')),
        (0.3618, 0.4895, 0.8337, float('inf'), float(
            'inf'), 0.8430, 0.7066, float('inf')),
        (0.3618, 0.4895, 0.8337, float('inf'), float(
            'inf'), 0.8430, 0.5549, float('inf')),
        (0.3618, 0.4895, 0.8337, float('inf'), float(
            'inf'), 0.8430, 0.4957, float('inf')),
        (0.3618, 0.4895, 0.8337, float('inf'), float(
            'inf'), 0.8430, 0.4957, float('inf')),
        (0.3618, 0.4895, 0.8337, float('inf'), float(
            'inf'), 0.8430, 0.4957, float('inf')),
        (0.3618, 0.4895, 0.8337, float('inf'), float(
            'inf'), 0.8430, 0.4957, float('inf')),
        (0.3618, 0.4895, 0.8337, float('inf'),
         float('inf'), 0.8430, 0.4957, 0.3619),
        (0.3618, 0.4895, 0.8337, float('inf'),
         float('inf'), 0.8430, 0.4957, 0.3619),
        (0.3618, 0.4895, 0.8337, float('inf'), float(
            'inf'), float('inf'), 0.4957, 0.3619),
        (0.3618, 0.4895, 0.8337, float('inf'), float(
            'inf'), float('inf'), 0.4957, 0.3619),
        (0.3618, 0.4895, 0.8337, float('inf'), float(
            'inf'), float('inf'), 0.4957, 0.3619),
        (0.3618, 0.4895, 0.8337, float('inf'), float(
            'inf'), float('inf'), 0.4957, 0.3619),
        (0.3618, 0.4895, 0.8337, float('inf'), float(
            'inf'), float('inf'), float('inf'), 0.3619),
        (0.3618, 0.4895, 0.8337, float('inf'), float(
            'inf'), float('inf'), float('inf'), 0.3619),
        (0.3618, 0.4895, 0.8337, float('inf'), float(
            'inf'), float('inf'), float('inf'), 0.3619),
        (0.3618, 0.4895, 0.8337, float('inf'), float(
            'inf'), float('inf'), float('inf'), 0.3619),
        (0.3618, 0.4895, 0.8337, float('inf'), float(
            'inf'), 0.9920, float('inf'), float('inf')),
        (0.3618, 0.4895, 0.8337, float('inf'), float(
            'inf'), 0.8795, float('inf'), float('inf')),
        (0.3832, 0.6021, float('inf'), float('inf'),
         1.2914, 0.9590, float('inf'), float('inf')),
        (0.4207, 0.7867, float('inf'), float('inf'),
         0.9038, float('inf'), float('inf'), 0.5420),
        (0.4778, float('inf'), float('inf'), float('inf'),
         0.8626, float('inf'), float('inf'), 0.3648),
        (0.5609, float('inf'), float('inf'), 0.9514,
         0.9707, float('inf'), float('inf'), 0.3669),
        (0.6263, float('inf'), float('inf'), 0.8171,
         0.8584, float('inf'), float('inf'), 0.4199),
        (0.6918, float('inf'), 0.9942, 0.6828,
         0.7461, float('inf'), float('inf'), 0.5652),
        (0.7572, 0.9544, 0.9130, 0.5485, 0.6338,
         float('inf'), float('inf'), 0.7106),
        (0.8226, 0.8701, 0.8319, 0.4142, 0.5215,
         float('inf'), float('inf'), 0.8559),
        (0.8880, 0.7858, 0.7507, 0.2894, 0.4092,
         float('inf'), float('inf'), float('inf')),
        (0.9534, 0.7016, 0.6696, 0.2009, 0.2969,
         float('inf'), float('inf'), float('inf')),
        (float('inf'), 0.6173, 0.5884, 0.1124,
         0.1847, 0.4020, float('inf'), float('inf')),
        (0.9789, 0.5330, 0.1040, 0.0238, 0.0724, 0.2183, float('inf'), float('inf'))]

    # Wheel angular speed commands (left, right) [rad/s]
    motions = [(0, 0), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1),
               (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1,
                                                                1), (1, 1), (0.5, 0), (0.5, 0), (0.5, 0), (0.5, 0),
               (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]

    dt = 1  # Sampling time [s]

    m = Map('map_pf.json', sensor_range=RobotP3DX.SENSOR_RANGE,
            compiled_intersect=True, use_regions=True)
    pf = ParticleFilter(
        m, RobotP3DX.SENSORS[:8], RobotP3DX.SENSOR_RANGE, particle_count=1000)

    for u, z in zip(motions, measurements):
        # Solve differential kinematics
        v = (u[0] + u[1]) * RobotP3DX.WHEEL_RADIUS / 2
        w = (u[1] - u[0]) * RobotP3DX.WHEEL_RADIUS / RobotP3DX.TRACK

        # Move
        start = time.perf_counter()
        pf.move(v, w, dt)
        move = time.perf_counter() - start

        start = time.perf_counter()
        pf.show('Move', save_figure=True)
        plot_move = time.perf_counter() - start

        # Sense
        start = time.perf_counter()
        pf.resample_v2(z)
        sense = time.perf_counter() - start

        start = time.perf_counter()
        pf.show('Sense', save_figure=True)
        plot_sense = time.perf_counter() - start

        # Display timing results
        print(
            'Particle filter: {0:6.3f} s  =  Move: {1:6.3f} s  +  Sense: {2:6.3f} s   |   Plotting: {3:6.3f} s  =  Move: {4:6.3f} s  +  Sense: {5:6.3f} s'.format(
                move +
                sense,
                move,
                sense,
                plot_move +
                plot_sense,
                plot_move,
                plot_sense))


# This "strange" function is only called if this script (particle_filter.py) is the program's entry point.
if __name__ == '__main__':
    try:
        test()
    except KeyboardInterrupt:  # Press Ctrl+C to gracefully stop the program
        pass
