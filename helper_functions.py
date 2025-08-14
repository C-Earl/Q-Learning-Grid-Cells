from collections import defaultdict

import numpy as np
import torch

# Generate random weights for the grid cell to reservoir connections
def generate_grid_out_weights(in_size, out_size, sparsity, w_range):
  wmin, wmax = w_range
  w = np.zeros(in_size * out_size)
  num_ones = int(sparsity * in_size * out_size)
  w[:num_ones] = 1
  np.random.shuffle(w)
  w *= wmax   # TODO: NOTE: currently ignores min range, all synapses start with same strength
  w = w.reshape(in_size, out_size)
  return w


# Generate random weights for the reservoir to output connections
# Cumulative sum of weights for each pre-synaptic neuron (row) is equal to pre_synaptic_magnitude
def generate_res_out_weights(in_size, out_size, sparsity, w_range):
  wmin, wmax = w_range
  w = np.zeros(in_size * out_size)
  num_ones = int(sparsity * in_size * out_size)
  w[:num_ones] = 1
  np.random.shuffle(w)
  w *= wmax  # TODO: NOTE: currently ignores min range, all synapses start with same strength
  w = w.reshape(in_size, out_size)
  return w


# Determine which pre-synaptic neuron is most relevant
# Return index of top n most relevant (non-zero) neurons
def relevant_neurons(spike_train: torch.tensor, threshold: int = 4):
  total_spikes = spike_train.sum(axis=1)
  top_indices = np.where(total_spikes > threshold)[0]
  sorted_indices = top_indices[np.argsort(total_spikes[top_indices])][::-1]
  returned_indices = []
  firing_rates = []
  for ind in sorted_indices:
    if total_spikes[ind] > threshold:
      returned_indices.append(ind)
      firing_rates.append(total_spikes[ind])
  return returned_indices, firing_rates


def get_active_neurons(maze_size, spike_trains, threshold, verbose=False):
  active_neurons = {}
  # used_neurons = defaultdict(list)
  for i in range(maze_size[0]):  # Calculate relevant neurons for each position
    for j in range(maze_size[1]):
      top_neurons, firing_rates = relevant_neurons(spike_trains[i, j], threshold=threshold)
      active_neurons[(i, j)] = (top_neurons, firing_rates)
      if verbose:
        print(f"Position: ({i, j}), \n\tTop GCs: {top_neurons}, \n\tFiring Rates: {firing_rates}")
      # Find overlaps in relevant neurons
      # for neuron in top_neurons:
      #   used_neurons[neuron].append((i, j))

  return active_neurons
