from typing import Tuple
from navigation import Navigation
from robot_p3dx import RobotP3DX
from particle_filter import ParticleFilter
from planning import Planning
from map import Map
import math
import os
import sim
import time


# Ignore Fortran Ctrl+C override (set before loading SciPy)
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'


def create_robot(client_id: int, x: float, y: float, theta: float):
    current_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_path, 'p3dx.ttm')

    rc, out_ints, _, _, _ = sim.simxCallScriptFunction(client_id, 'Maze', sim.sim_scripttype_childscript, 'createRobot', [
    ], [x, y, theta], [model_path], "", sim.simx_opmode_blocking)
    robot_handle = out_ints[0]

    return rc, robot_handle


def goal_reached(robot_handle: int, goal: Tuple[float, float], localized: bool, tolerance: float = 0.1) -> bool:
    distance = float('inf')

    if localized:
        _, position = sim.simxGetObjectPosition(
            client_id, robot_handle, -1, sim.simx_opmode_buffer)
        distance = math.sqrt(
            (position[0] - goal[0]) ** 2 + (position[1] - goal[1]) ** 2)

    return distance < tolerance


if __name__ == '__main__':
    # Connect to CoppeliaSim
    sim.simxFinish(-1)  # Close all pending connections
    client_id = sim.simxStart('127.0.0.1', 19997, True, True, 2000, 5)

    if client_id == -1:
        raise ConnectionError(
            'Could not connect to CoppeliaSim. Make sure the application is open.')

    # Start simulation
    sim.simxSynchronous(client_id, True)
    sim.simxStartSimulation(client_id, sim.simx_opmode_blocking)

    # Execute a simulation step to ensure the simulation has started
    sim.simxSynchronousTrigger(client_id)
    # Make sure the simulation step has finished
    sim.simxGetPingTime(client_id)

    # Initial and final locations
    start = (2, -3, -math.pi / 2)
    goal = (2, 2)

    # Create the robot
    _, robot_handle = create_robot(client_id, start[0], start[1], start[2])
    # Initialize real position streaming
    sim.simxGetObjectPosition(
        client_id, robot_handle, -1, sim.simx_opmode_streaming)

    # Execute a simulation step to get initial sensor readings
    sim.simxSynchronousTrigger(client_id)
    # Make sure the simulation step has finished
    sim.simxGetPingTime(client_id)

    # Write initialization code here
    dt = 0.05
    dt_count = 0
    steps = 0
    robot = RobotP3DX(client_id, dt)
    navigation = Navigation(dt)
    localized = True
    start_time = time.perf_counter()
    m = Map('map_project.json', sensor_range=RobotP3DX.SENSOR_RANGE,
            compiled_intersect=True, use_regions=True)
    pf = ParticleFilter(
        m, RobotP3DX.SENSORS, RobotP3DX.SENSOR_RANGE, particle_count=3000,
        figure_size=(4, 4))
    action_costs = (1.0, 1.0, 1.0, 1.0)
    planning = Planning(m, action_costs)
    try:
        while not goal_reached(robot_handle, goal, localized):
            # Write your control algorithm here
            z_us, z_v, z_w = robot.sense()
            v, w = navigation.explore(z_us, z_v, z_w)
            robot.move(v, w)

            # Particle filter
            # dt_count += dt
            # pf.move(v, w, dt)
            # if dt_count >= dt * 5:
            #     pf.show('Move', save_figure=False)
            #     pf.resample_v2(z_us)
            #     pf.show('Sense', save_figure=False)
            #     dt_count = 0

            if localized:
                path = planning.a_star(start, goal)
                smoothed_path = planning.smooth_path(
                    path, data_weight=0.1, smooth_weight=0.20)
                planning.show(path, smoothed_path, block=True)

        # Execute the next simulation step
        sim.simxSynchronousTrigger(client_id)
        # Make sure the simulation step has finished
        sim.simxGetPingTime(client_id)
        steps += 1

    except KeyboardInterrupt:  # Press Ctrl+C to break the infinite loop and gracefully stop the simulation
        pass

    # Display time statistics
    execution_time = time.perf_counter() - start_time
    print('\n')
    print('Simulated steps: {0:d}'.format(steps))
    print('Simulated time:  {0:.3f} s'.format(steps * dt))
    print('Execution time:  {0:.3f} s ({1:.3f} s/step)'.format(
        execution_time, execution_time / steps))
    print('')

    # Stop the simulation and close the connection
    sim.simxStopSimulation(client_id, sim.simx_opmode_blocking)
    # Make sure the stop simulation command had time to arrive
    sim.simxGetPingTime(client_id)
    sim.simxFinish(client_id)
