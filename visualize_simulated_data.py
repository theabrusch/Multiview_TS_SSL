from src.datasets.simulated_data import cpc_data_simulator, multiview_data_simulator, finetuning_simulator
import matplotlib.pyplot as plt
import numpy as np
plot = False
def plot_data(data, sources, emission_matrices, data_to_plot, source_state = None, plotline = True):
    data_sources, vars, titles = [data, emission_matrices, sources], [data.shape[2], 1, sources.shape[2]], ['Data', 'Mixing matrix', 'Sources']

    fig = plt.figure(figsize=(15, len(data_to_plot)*5))
    subfigs = fig.subfigures(len(data_to_plot), 1, squeeze=False)

    for k in range(len(data_to_plot)):
        subfig_ax = subfigs[k][0].subfigures(1, 3)
        for i in range(3):
            ax = subfig_ax[i].subplots(vars[i], 1, squeeze=False)
            subfig_ax[i].suptitle('{}'.format(titles[i]))
            if i == 1:
                if len(data_sources[i].shape) > 2:
                    pos = ax[0][0].imshow(data_sources[i][data_to_plot[k], :, :], aspect = 'auto')
                else:
                    pos = ax[0][0].imshow(data_sources[i], aspect = 'auto')
                fig.colorbar(pos, ax=ax[0][0])
                continue
            for j in range(vars[i]):
                if source_state is not None:
                    if j == source_state and i == 2:
                        ax[j][0].plot(data_sources[i][data_to_plot[k], :, j], color='r')
                    else:
                        ax[j][0].plot(data_sources[i][data_to_plot[k], :, j])
                # put vertical line at the middel of the time axis
                if plotline:
                    ax[j][0].axvline(x=data_sources[i].shape[1]/2, color='r', linestyle='--')
    plt.tight_layout()
    plt.show()

n_states = 1000
sigma = 0.5
fs = 100
length = 1

# Simulate cpc data
groups_of_dep_var = 5*[2]
n_sources = len(groups_of_dep_var)*[3]
simulator = cpc_data_simulator(n_sources, groups_of_dep_var, n_states, sigma, fs, length*2)

train, sources, emission = simulator.generate(2, random_emission_matrix=False, random_settings=True, return_sources=True)

plot_data(train, sources, emission, [0, 1])

# Simulate multiview data
n_sources = 10
groups_of_dep_var = 10
simulator = multiview_data_simulator(n_sources, groups_of_dep_var, n_states, sigma, fs, length*2)
train, sources, emission = simulator.generate(2, random_emission_matrix=False, random_settings=True, return_sources=True)

if plot:
    plot_data(train, sources, emission, [0, 1])

# simulate multiview data for finetuning
n_sources = [8]
groups_of_dep_var = [10]
simulator = finetuning_simulator('simulated_multiview', n_sources, groups_of_dep_var, n_states, sigma, fs, length)
data, sources, emisison, states = simulator.generate(2, return_sources=True)

source_state = simulator.y_state

plot_data(data, sources, emisison, [0, 1], source_state, False)