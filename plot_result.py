import matplotlib.pyplot as plt
import numpy as np

r_max_list = np.load("plot_data/r_max.npy")
r_var_list = np.load("plot_data/r_var.npy")
reward_list = -np.load("plot_data/r_mean.npy")
energy_list = np.load("plot_data/r_energy.npy")
# plt.scatter(r_max_list, r_var_list)
# plt.xlabel("max_distance")
# plt.ylabel("distance_variance")
# plt.show()

pareto = np.array([0, 1, 9, 7])

plt.scatter(reward_list, energy_list, label='all designs')
plt.plot(reward_list[pareto], energy_list[pareto], 'r', label='pareto-front')


plt.xlabel("negative_mean_distance")
plt.ylabel("energy_cost_per_distance")
plt.legend()
plt.show()