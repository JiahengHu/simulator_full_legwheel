
from simulation_runner import simulation_runner
import numpy as np

NUM_ENVS = 1
SIM_TIME_STEPS = 250

# sim_runner = simulation_runner(NUM_ENVS, show_GUI=True, record_video=True)
sim_runner = simulation_runner(NUM_ENVS, show_GUI=True, record_video=True, gui_speed_factor =5)
robot_names = ['www', 'lww', 'lwl', 'lll']
# robot_names = ['www', 'wlw', 'llw']

# terrain_block_height = sim_runner.MAX_BLOCK_HEIGHT_LOW
# terrain_block_height = sim_runner.MAX_BLOCK_HEIGHT_HIGH
terrain_block_heights =  np.linspace(
                    sim_runner.MAX_BLOCK_HEIGHT_LOW+0.01,
                    sim_runner.MAX_BLOCK_HEIGHT_HIGH, len(robot_names))
# terrain = sim_runner.randomize_terrains()
terrain_block_distance = (sim_runner.MIN_BLOCK_DISTANCE_LOW + sim_runner.MIN_BLOCK_DISTANCE_HIGH)/2
terrain_block_distance -= 0.1

for i in range(len(robot_names)):
    robot_name = robot_names[i]
    terrain_block_height = terrain_block_heights[i]
    terrain = sim_runner.randomize_terrains(
        terrain_block_height=terrain_block_height,
        terrain_block_distance = terrain_block_distance)
    sim_runner.load_robots(robot_name, randomize_xyyaw=False)
    rewards = sim_runner.run_sims(n_time_steps=SIM_TIME_STEPS, allow_early_stop=False)
