import ctypes as ct
import json
import math
import numpy as np
import os
import platform

from matplotlib import pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon, LineString
from typing import List, Tuple


class Map:
    """Class to perform operations on metric maps."""

    def __init__(self, json_file: str, sensor_range: float, use_regions: bool = True, compiled_intersect: bool = True):
        """Map class initializer.

        Args:
            json_file: JSON file with the coordinates of the external boundary and the internal hole vertices.
            sensor_range: Sensor measurement range [m].
            use_regions: Split the map in regions to reduce the number of comparisons required to find intersections.
            compiled_intersect: Use compiled intersect library for increased performance.

        """
        # Load map from JSON file
        with open(json_file, "r") as read_file:
            data = json.load(read_file)

        boundary = data['metric']['boundary']
        holes = data['metric']['holes']

        # Create a polygon map
        self._map_polygon = Polygon(boundary, holes=holes)

        # Create a segment map
        self._map_segments = []

        for i in range(len(boundary) - 1):
            self._map_segments.append([boundary[i], boundary[i + 1]])

        self._map_segments.append([boundary[-1], boundary[0]])

        for hole in holes:
            for i in range(len(hole) - 1):
                self._map_segments.append([hole[i], hole[i + 1]])

            self._map_segments.append([hole[-1], hole[0]])

        # Create a grid map
        try:
            map_size = data['grid']['size']
            obstacles = data['grid']['obstacles']

            self._grid_map = np.zeros(map_size, np.int8)

            for obstacle in obstacles:
                self._grid_map[tuple(obstacle)] = 1
        except KeyError:
            self._grid_map = None

        # Performance optimizations
        self._sensor_range = sensor_range
        self._intersect = self._init_intersect() if compiled_intersect else None
        self._region_segments = self._init_regions() if use_regions else None

    def bounds(self) -> Tuple[float, float, float, float]:
        """Coordinates of a bounding box that contains the map.

        Returns:
            x_min: Bottom left corner x coordinate [m].
            y_min: Bottom left corner y coordinate [m].
            x_max: Top right corner x coordinate [m].
            y_max: Top right corner y coordinate [m].

        """
        return self._map_polygon.bounds

    def check_collision(self, segment: List[Tuple[float, float]], compute_distance: bool = False) -> \
            Tuple[Tuple[float, float], float]:
        """Determines if a segment intersects with the map.

        Args:
            segment: Sensor ray or motion trajectory in the format [(start), (end)].
            compute_distance: True to compute the distance between the robot and the intersection point.

        Returns:
            intersection: Closest collision point (x, y) [m].
            distance: Distance to the obstacle [m]. inf if not computed.

        """
        intersections = []
        distance = float('inf')
        index = 0

        try:
            if self._region_segments is not None:
                r, c = self._xy_to_rc(segment[0])

                if r < 0 or c < 0:  # Consider negative indices out or range
                    raise IndexError

                map_segments = self._region_segments[r][c]
            else:
                map_segments = self._map_segments

            if self._intersect is not None:
                xi = ct.c_double(0.0)
                yi = ct.c_double(0.0)
                xp = ct.POINTER(ct.c_double)(xi)
                yp = ct.POINTER(ct.c_double)(yi)

                for map_segment in map_segments:
                    found = self._intersect.segment_intersect(
                        xp, yp,
                        segment[0][0], segment[0][1], segment[1][0], segment[1][1],
                        map_segment[0][0], map_segment[0][1], map_segment[1][0], map_segment[1][1]
                    )

                    if found:
                        intersections.append((xi.value, yi.value))
            else:
                from intersect import Intersect

                intersect = Intersect()

                for map_segment in map_segments:
                    pt = intersect.segment_intersect(segment, map_segment)

                    if pt is not None:
                        intersections.append(pt)

            if (compute_distance and intersections) or len(intersections) > 1:
                distances = [math.sqrt((pt[0] - segment[0][0]) ** 2 + (pt[1] - segment[0][1]) ** 2) for pt in intersections]
                index = int(np.argmin(distances))
                distance = distances[index]
        except IndexError:
            pass  # Sensor rays may be outside the map even though the center of the robot is within it.

        intersection = intersections[index] if intersections else []

        return intersection, distance

    def contains(self, point: Tuple[float, float]) -> bool:
        """Determines whether a point is within the map limits.

        Args:
            point: (x, y) coordinates to check.

        Returns:
            bool: True if the point is inside the map; False otherwise.

        """
        pt = Point(point[0], point[1])

        return self._map_polygon.contains(pt)

    @property
    def grid_map(self) -> np.ndarray:
        """Grid map getter.

        Returns:
            A 2D matrix containing 1 in cells with obstacles and 0 elsewhere. None if not available.

        """
        return self._grid_map

    def plot(self, axes):
        """Draws the map.

           Args:
               axes: Figure axes.

           Returns:
               axes: Modified axes.

        """
        x_min, y_min, x_max, y_max = self.bounds()

        major_ticks = np.arange(math.floor(min(x_min, y_min)), math.ceil(max(x_max, y_max)) + 0.01, 1)
        minor_ticks = np.arange(math.floor(min(x_min, y_min)), math.ceil(max(x_max, y_max)) + 0.01, 0.5)

        axes.set_xticks(major_ticks)
        axes.set_xticks(minor_ticks, minor=True)
        axes.set_yticks(major_ticks)
        axes.set_yticks(minor_ticks, minor=True)

        axes.set_xlim(math.floor(x_min), math.ceil(x_max))
        axes.set_ylim(math.floor(y_min), math.ceil(y_max))
        axes.grid(which='both', alpha=0.33, linestyle='dashed', zorder=1)
        axes.set(xlabel='x [m]', ylabel='y [m]')

        # Plot map
        x, y = self._map_polygon.exterior.xy
        axes.plot(x, y, color='black', alpha=1, linewidth=3, solid_capstyle='round', zorder=2)

        for interior in self._map_polygon.interiors:
            x, y = interior.xy
            axes.plot(x, y, color='black', alpha=1, linewidth=3, solid_capstyle='round', zorder=2)

        return axes

    def show(self, figure_number: int = 1, title: str = 'Map', block: bool = True,
             figure_size: Tuple[float, float] = (7, 7)):
        """Displays the map in a figure.

        Args:
            figure_number: Any existing figure with the same value will be overwritten.
            title: Plot title.
            block: True to stop program execution until the figure window is closed.
            figure_size: Figure window dimensions.

        """
        figure, axes = plt.subplots(1, 1, figsize=figure_size, num=figure_number)
        axes = self.plot(axes)
        axes.set_title(title)
        figure.tight_layout()  # Reduce white margins

        plt.show(block=block)
        plt.pause(0.0001)  # Wait for 0.1 ms or the figure won't be displayed

    def show_regions(self, figure_number: int = 1, title: str = 'Map regions', block: bool = True,
                     figure_size: Tuple[float, float] = (7, 7)):
        """Displays the map segments that belong to each region.

        Args:
            figure_number: Any existing figure with the same value will be overwritten.
            title: Plot title.
            block: True to stop program execution until the figure window is closed.
            figure_size: Figure window dimensions.

        """
        x_min, y_min, x_max, y_max = self.bounds()
        rows, cols = self._region_segments.shape
        major_ticks = np.arange(min(x_min, y_min), max(x_max, y_max) + 0.01, 1)

        if rows <= 5 and cols <= 5:
            label_size = 8
            map_line_width = 2
            marker_size = 5
            figure, axes = plt.subplots(rows, cols, figsize=figure_size, num=figure_number, sharex=True, sharey=True)
        else:
            label_size = 5
            map_line_width = 1.75
            marker_size = 1.5
            figure, axes = plt.subplots(rows, cols, figsize=figure_size, num=figure_number, sharex=True,
                                        sharey=True, gridspec_kw={'hspace': 0, 'wspace': 0})

        for ax in axes.flat:
            ax.set_xlabel('x [m]', fontsize='small')
            ax.set_ylabel('y [m]', fontsize='small')
            ax.label_outer()  # Hide x labels and tick labels for top plots and y ticks for right plots.

            ax.set_xticks(major_ticks)
            ax.set_yticks(major_ticks)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.tick_params(axis='x', labelsize=label_size, rotation=90)
            ax.tick_params(axis='y', labelsize=label_size)

            ax.grid(which='both', alpha=0.33, linestyle='dashed', zorder=1)

        figure.suptitle(title)
        figure.tight_layout()  # Reduce white margins

        for y in np.arange(y_max - 0.5, y_min, -1):
            for x in np.arange(x_min + 0.5, x_max):
                circle = Point(x, y).buffer(self._sensor_range + 1 / math.sqrt(2))
                cx, cy = circle.exterior.xy

                r, c = m._xy_to_rc((x, y))
                axes[r, c].plot(x, y, 'bo', markersize=marker_size)
                axes[r, c].plot(cx, cy, color='green', alpha=1, linewidth=1, linestyle='dashed', zorder=3)

                for s in m._region_segments[r][c]:
                    lx, ly = LineString(s).xy
                    axes[r, c].plot(lx, ly, color='black', linewidth=map_line_width, solid_capstyle='round', zorder=2)

        plt.show(block=block)

    def _init_intersect(self) -> ct.CDLL:
        """Loads a C library to compute intersections faster.

        Returns:
            intersect: An object to call functions in the library.
        """
        library_names = {
            'Windows': 'libintersect.dll',
            'Darwin': 'libintersect.dylib',
            'Linux': 'libintersect.so'
        }

        library_path = os.path.join(os.path.dirname(__file__), library_names[platform.system()])
        intersect = ct.CDLL(library_path)

        # Initialize function arguments and return value types
        intersect.segment_intersect.restype = ct.c_bool
        intersect.segment_intersect.argtypes = [
            ct.POINTER(ct.c_double),   # xi
            ct.POINTER(ct.c_double),   # yi
            ct.c_double, ct.c_double,  # x0, y0
            ct.c_double, ct.c_double,  # x1, y1
            ct.c_double, ct.c_double,  # x2, y2
            ct.c_double, ct.c_double]  # x3, y3

        return intersect

    def _init_regions(self) -> np.ndarray:
        """Divides the map in 1x1 m squares and finds the potentially visible segments.

        This function can be further improved by considering occlusions.

        Returns:
            region_segments: A 2D matrix that contains the segments for each region.
        """
        # Obtain map dimensions
        x_min, y_min, x_max, y_max = self.bounds()
        map_rows, map_cols = math.ceil(y_max - y_min), math.ceil(x_max - x_min)

        # Precomputed constants to convert from (x, y) to (row, col) faster
        self._XC = math.floor(map_cols / 2.0)
        self._YR = map_rows - math.ceil(map_rows / 2.0)

        # Find the segments visible from each region
        region_segments = np.zeros((map_rows, map_cols), dtype=list)

        for y in np.arange(y_max - 0.5, y_min, -1):
            for x in np.arange(x_min + 0.5, x_max):
                circle = Point(x, y).buffer(self._sensor_range + 1 / math.sqrt(2))
                segments = []

                for segment in self._map_segments:
                    line = LineString(segment)

                    if line.intersects(circle) and not line.touches(circle):
                        segments.append(segment)

                r, c = self._xy_to_rc((x, y))
                region_segments[r][c] = segments

        return region_segments

    def _xy_to_rc(self, xy: Tuple[float, float]) -> Tuple[int, int]:
        """Converts (x, y) coordinates of a metric map to (row, col) coordinates of a grid map.

        Args:
            xy: (x, y) [m].

        Returns:
            rc: (row, col) starting from (0, 0) at the top left corner.

        """
        x = math.floor(xy[0])
        y = math.ceil(xy[1])

        row = int(self._YR - y)
        col = int(x + self._XC)

        return row, col


if __name__ == '__main__':
    # Display the full map and its regions
    m = Map('map_project.json', sensor_range=1.0, compiled_intersect=False)
    m.show(figure_number=1, block=False, figure_size=(8, 8))
    m.show_regions(figure_number=2, figure_size=(8, 8))
