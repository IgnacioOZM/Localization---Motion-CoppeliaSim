from hashlib import new
import math
import numpy as np
import os

from map import Map
from matplotlib import pyplot as plt
from typing import Dict, List, Tuple
from copy import deepcopy
from collections import defaultdict


class Planning:
    """Class to plan the optimal path to a given location."""

    def __init__(self, map_object: Map, action_costs: Tuple[float, float, float, float]):
        """Planning class initializer.

        Args:
            map_object: Map of the environment.
            action_costs: Cost of of moving one cell left, right, up, and down.

        """
        self._map = map_object

        self._actions = np.array([
            (-1, 0),  # Move one cell left
            (0, 1),   # Move one cell up
            (1, 0),   # Move one cell right
            (0, -1)   # Move one cell down
        ])

        self._action_costs = action_costs

    def a_star(self, start: Tuple[float, float], goal: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Computes the optimal path to a given goal location using the A* algorithm.

        Args:
            start: Initial location in (x, y) format.
            goal: Destination in (x, y) format.

        Returns:
            Path to the destination. The first value corresponds to the initial location.

        """
        # Check start and goal
        if not(self._map.contains(start) and self._map.contains(goal)):
            raise Exception('Not inside')

        # Compute the heuristic map
        self._heuristic_map = self._compute_heuristic(goal)

        # Oppend and close lists
        open_dict = {start: (self._heuristic_map[self._xy_to_rc(start)], 0)}
        closed_dict = {}
        ancestors = {}

        step = 1
        while open_dict:
            # TODO: Explain what this line does
            current_node = min(open_dict, key=lambda k: open_dict.get(k)[0])
            f, g = open_dict.pop(current_node)

            if current_node == goal:
                print(f'Number of setps: {step}')
                return self._reconstruct_path(start, goal, ancestors)

            neighbors = [(current_node[0] - 1, current_node[1] + 0),
                         (current_node[0] + 0, current_node[1] + 1),
                         (current_node[0] + 1, current_node[1] - 0),
                         (current_node[0] + 0, current_node[1] - 1)]

            for i, neighbor in enumerate(neighbors):
                if not self._map.contains(neighbor) or neighbor in closed_dict:
                    continue
                g_new = g + self._action_costs[i]
                f_new = g_new + self._heuristic_map[self._xy_to_rc(neighbor)]
                open_dict[neighbor] = (f_new, g_new)
                ancestors[neighbor] = current_node

            closed_dict[current_node] = (f, g)
            step += 1
        return

    @staticmethod
    def smooth_path(path, data_weight: float = 0.1, smooth_weight: float = 0.1, tolerance: float = 1e-6) -> \
            List[Tuple[float, float]]:
        """Computes a smooth trajectory from a Manhattan-like path.

        Args:
            path: Non-smoothed path to the goal (start location first).
            data_weight: The larger, the more similar the output will be to the original path.
            smooth_weight: The larger, the smoother the output path will be.
            tolerance: The algorithm will stop when after an iteration the smoothed path changes less than this value.

        Returns: Smoothed path (initial location first) in (x, y) format.

        """
        newpath = [list(element) for element in path]

        change = tolerance

        while change >= tolerance:
            change = 0
            for i in range(1, len(path) - 1):
                for j in range(2):
                    ori = newpath[i][j]
                    newpath[i][j] = newpath[i][j] + \
                        data_weight * (path[i][j] - newpath[i][j])

                    newpath[i][j] = newpath[i][j] + smooth_weight * \
                        (newpath[i + 1][j] + newpath[i - 1][j] - 2 * newpath[i][j])
                    change += abs(ori - newpath[i][j])

        return newpath

    @staticmethod
    def plot(axes, path: List[Tuple[float, float]], smoothed_path: List[Tuple[float, float]] = ()):
        """Draws a path.

        Args:
            axes: Figure axes.
            path: Path (start location first).
            smoothed_path: Smoothed path (start location first).

        Returns:
            axes: Modified axes.

        """
        x_val = [x[0] for x in path]
        y_val = [x[1] for x in path]

        axes.plot(x_val, y_val)  # Plot the path
        # Draw blue circles in every intermediate cell
        axes.plot(x_val[1:-1], y_val[1:-1], 'bo', markersize=4)

        if smoothed_path:
            x_val = [x[0] for x in smoothed_path]
            y_val = [x[1] for x in smoothed_path]

            axes.plot(x_val, y_val, 'y')  # Plot the path
            # Draw yellow circles in every intermediate cell
            axes.plot(x_val[1:-1], y_val[1:-1], 'yo', markersize=4)

        # Draw a red square at the start location
        axes.plot(x_val[0], y_val[0], 'rs', markersize=7)
        # Draw a green star at the goal location
        axes.plot(x_val[-1], y_val[-1], 'g*', markersize=12)

        return axes

    def show(self, path, smoothed_path=(), figure_number: int = 1, title: str = 'Path', block: bool = False,
             figure_size: Tuple[float, float] = (7, 7), save_figure: bool = False, save_dir: str = 'img'):
        """Displays a given path on the map.

        Args:
            path: Path (start location first).
            smoothed_path: Smoothed path (start location first).
            figure_number: Any existing figure with the same value will be overwritten.
            title: Plot title.
            blocking: True to stop program execution until the figure window is closed.
            figure_size: Figure window dimensions.
            save_figure: True to save figure to a .png file.
            save_dir: Image save directory.

        """
        figure, axes = plt.subplots(
            1, 1, figsize=figure_size, num=figure_number)
        axes = self._map.plot(axes)
        axes = self.plot(axes, path, smoothed_path)
        axes.set_title(title)
        figure.tight_layout()  # Reduce white margins

        plt.show(block=block)
        plt.pause(0.0001)  # Wait for 0.1 ms or the figure won't be displayed

        if save_figure:
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)

            file_name = str(title.lower() + '.png')
            file_path = os.path.join(save_dir, file_name)
            figure.savefig(file_path)

    def _compute_heuristic(self, goal: Tuple[float, float]) -> np.ndarray:
        """Creates an admissible heuristic.

        Args:
            goal: Destination location in (x,y) coordinates.

        Returns:
            Admissible heuristic.

        """
        goal = self._xy_to_rc(goal)
        matrix_map = self._map._grid_map
        heuristic_map = np.empty(matrix_map.shape, dtype=np.int8)

        for index, _ in np.ndenumerate(heuristic_map):
            row, col = index
            heuristic_map[index] = abs(goal[0] - row) + abs(goal[1] - col)

        return heuristic_map

    def _compute_naive(self, goal: Tuple[float, float]) -> np.ndarray:
        """Creates an admissible heuristic.

        Args:
            goal: Destination location in (x,y) coordinates.

        Returns:
            Admissible heuristic.

        """
        matrix_map = self._map._grid_map
        heuristic_map = np.empty(matrix_map.shape, dtype=np.int8)
        for row in range(matrix_map.shape[0]):
            for col in range(matrix_map.shape[1]):
                heuristic_map[row, col] = 0
        return heuristic_map

    def _reconstruct_path(self, start: Tuple[float, float], goal: Tuple[float, float],
                          ancestors: Dict[Tuple[int, int], Tuple[int, int]]) -> List[Tuple[float, float]]:
        """Computes the trajectory from the start to the goal location given the ancestors of a search algorithm.

        Args:
            start: Initial location in (x, y) format.
            goal: Goal location in (x, y) format.
            ancestors: Matrix that contains for every cell, None or the (x, y) ancestor from which it was opened.

        Returns: Path to the goal (start location first) in (x, y) format.

        """
        path = [goal]

        while ancestors:
            new_coord = ancestors.pop(path[0])
            path.insert(0, new_coord)
            if new_coord == start:
                return path
        return

    def _xy_to_rc(self, xy: Tuple[float, float]) -> Tuple[int, int]:
        """Converts (x, y) coordinates of a metric map to (row, col) coordinates of a grid map.

        Args:
            xy: (x, y) [m].

        Returns:
            rc: (row, col) starting from (0, 0) at the top left corner.

        """
        map_rows, map_cols = np.shape(self._map.grid_map)

        x = round(xy[0])
        y = round(xy[1])

        row = int(map_rows - (y + math.ceil(map_rows / 2.0)))
        col = int(x + math.floor(map_cols / 2.0))

        return row, col

    def _rc_to_xy(self, rc: Tuple[int, int]) -> Tuple[float, float]:
        """Converts (row, col) coordinates of a grid map to (x, y) coordinates of a metric map.

        Args:
            rc: (row, col) starting from (0, 0) at the top left corner.

        Returns:
            xy: (x, y) [m].

        """
        map_rows, map_cols = np.shape(self._map.grid_map)
        row, col = rc

        x = col - math.floor(map_cols / 2.0)
        y = map_rows - (row + math.ceil(map_rows / 2.0))

        return x, y


def test():
    """Function used to test the Planning class independently."""
    m = Map('map_project.json', sensor_range=1.0, compiled_intersect=False, use_regions=False)

    start = (-4.0, -4.0)
    goal = (4.0, 4.0)
    action_costs = (1.0, 1.0, 1.0, 1.0)

    planning = Planning(m, action_costs)
    path = planning.a_star(start, goal)
    smoothed_path = planning.smooth_path(path, data_weight=0.2, smooth_weight=0.1)
    planning.show(path, smoothed_path, block=True)


if __name__ == '__main__':
    test()
