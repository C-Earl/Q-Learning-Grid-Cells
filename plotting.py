import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from collections import defaultdict
from matplotlib import gridspec
import numpy as np
import os
import shutil

from Grid_Cells import GC_Population
from Association import Association
from Environment import Grid_Cell_Maze_Environment
from helper_functions import relevant_neurons, get_active_neurons

import matplotlib
matplotlib.use("Agg")

def plot_training_history(history, save_file, p1_len, p2_len, p3_len, p4_len, p5_len):
  # Determine start position based on the provided lengths
  # p1_start = 0
  # p1_end = 0+p1_len
  p2_start = p1_len
  p2_end = p2_start+p2_len
  p3_start = p2_end
  p3_end = p3_start+p3_len
  p4_start = p3_end
  p4_end = p4_start+p4_len
  p5_start = p4_end
  p5_end = p5_start+p5_len

  FULL_SAVE_FILE = os.path.join("./saves/plots", save_file)

  # Extract episode lengths and average delta Q values
  ep_lengths = [len(ep) for ep in history]
  delta_Qs = [[ep[i][4] if ep[i][4] != -1 else -0.001 for i in range(len(ep)-1)] for ep in history]
  avg_delta_q = [np.mean(ep) for ep in delta_Qs]

  fig = plt.figure(figsize=(8, 6))
  gs = gridspec.GridSpec(2, 1)
  ax1 = fig.add_subplot(gs[0, 0])
  ax2 = fig.add_subplot(gs[1, 0])

  # Plot episode lengths
  ax1.set_title("Learning Curve")
  ax1.set_xlabel("Episode")
  ax1.set_ylabel("Steps")
  ax1.plot(ep_lengths, color='blue', label="Warmup")    # Plot whole history in blue, overlay with red
  ax1.plot(np.arange(p2_start, p2_end), ep_lengths[p2_start:p2_end], color='red', label="Active Learning")
  ax1.plot(np.arange(p4_start, p5_end), ep_lengths[p4_start:p5_end], color='red')
  ax1.axvline(x=p3_start-1, color='g', linestyle='--', label="Start Position Change")
  ax1.axvline(x=p5_start-1, color='g', linestyle='--')
  ax1.legend(loc="lower left")

  # Plot average delta Q
  ax2.set_title("Average Delta Q")
  ax2.set_xlabel("Episode")
  ax2.set_ylabel("Average Delta Q")
  ax2.plot(avg_delta_q, color='purple', label="Average Delta-Q")
  ax2.axvline(x=p3_start-1, color='g', linestyle='--', label="Start Position Change")
  ax2.axvline(x=p5_start-1, color='g', linestyle='--')
  ax2.legend()
  plt.tight_layout()
  plt.savefig(FULL_SAVE_FILE, dpi=200)


def plot_active_grid_cells(active_neurons: dict, num_gc: int, save_file, maze_env: Grid_Cell_Maze_Environment=None, verbose=False):
  maze_size = maze.get_shape()
  OVERLAP_FULL_SAVE_FILE = os.path.join("./saves/plots", save_file)

  # Cosine similarity of two positions
  active_list = list(active_neurons.items())
  overlap_matrix = np.zeros((len(active_list), len(active_list)))
  for i in range(len(active_list)):
    for j in range(i + 1, len(active_list)):
      # vec1 = np.zeros(num_gc)
      # vec1[active_list[i][1][0]] = 1
      # vec2 = np.zeros(num_gc)
      # vec2[active_list[j][1][0]] = 1
      # overlap_matrix[i, j] = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
      pos1, data1 = active_list[i]
      pos2, data2 = active_list[j]
      if pos1 != pos2:
        overlap = set(data1[0]) & set(data2[0])
        if len(overlap) >= 2:
          if verbose:
            print(f"Positions {pos1} and {pos2} share {overlap} Grid Cells.")
          overlap_matrix[i, j] = len(overlap)

  # Plot relevant neurons per position
  active_neurons_matrix = np.zeros((maze_size[0], maze_size[1]))
  for (i, j), (neurons, _) in active_neurons.items():
    active_neurons_matrix[i, j] = len(neurons)
  vmin = int(0)
  vmax = int(np.max(active_neurons_matrix))
  fig, axs = plt.subplots(1, 2, figsize=(14, 7))
  im = axs[0].imshow(active_neurons_matrix, cmap='Blues', interpolation='nearest',
                  vmin=vmin, vmax=vmax,)
  cbar = fig.colorbar(im, ax=axs[0], shrink=0.8)
  cbar.set_ticks(np.arange(vmin, vmax + 1, 2))
  axs[0].set_title("Number of Active Grid Cells per Position")
  axs[0].set_xlabel("X Position")
  axs[0].set_ylabel("Y Position")
  if maze_env:
    maze_env.plot(ax=axs[0], path_color=None, line_width=5)

  # Plot overlap matrix
  im = axs[1].imshow(overlap_matrix, cmap='Blues', interpolation='nearest',
                  vmin=0, vmax=vmax,)
  fig.tight_layout(w_pad=5.0)
  cbar = fig.colorbar(im, ax=axs[1], shrink=0.8)
  cbar.set_ticks(np.arange(vmin, vmax + 1, 2))
  plt.title("Number of Overlapping Active Grid Cells")
  plt.xlabel("Position 1")
  plt.ylabel("Position 2")
  plt.savefig(OVERLAP_FULL_SAVE_FILE, dpi=500)


def plot_active_assoc_cells(active_neurons: dict, save_file, maze_env: Grid_Cell_Maze_Environment=None, verbose=False):
  maze_size = maze.get_shape()
  OVERLAP_FULL_SAVE_FILE = os.path.join("./saves/plots", save_file)

  # Check if any two positions share 2 or more active neurons
  active_list = list(active_neurons.items())
  overlap_matrix = np.zeros((len(active_list), len(active_list)))
  for i in range(len(active_list)):
    for j in range(i + 1, len(active_list)):
      pos1, data1 = active_list[i]
      pos2, data2 = active_list[j]
      if pos1 != pos2:
        overlap = set(data1[0]) & set(data2[0])
        if len(overlap) >= 2:
          if verbose:
            print(f"Positions {pos1} and {pos2} share {overlap} Assoc Cells.")
          overlap_matrix[i, j] = len(overlap)

  # Plot relevant neurons per position
  active_neurons_matrix = np.zeros((maze_size[0], maze_size[1]))
  for (i, j), (neurons, _) in active_neurons.items():
    active_neurons_matrix[i, j] = len(neurons)
  vmin = int(0)
  vmax = int(np.max(active_neurons_matrix))
  fig, axs = plt.subplots(1, 2, figsize=(14, 7))
  im = axs[0].imshow(active_neurons_matrix, cmap='Reds', interpolation='nearest',
                     vmin=vmin, vmax=vmax, )
  axs[0].set_title("Number of Active Assoc. Cells per Position")
  axs[0].set_xlabel("X Position")
  axs[0].set_ylabel("Y Position")
  if maze_env:
    maze_env.plot(ax=axs[0], path_color=None, line_width=5)

  # Plot overlap matrix
  im = axs[1].imshow(overlap_matrix, cmap='Reds', interpolation='nearest',
                     vmin=vmin, vmax=vmax, )
  fig.tight_layout(w_pad=5.0)
  cbar = fig.colorbar(im, ax=axs, shrink=0.8)
  cbar.set_ticks(np.arange(vmin, vmax + 1, 5))
  plt.title("Number of Overlapping Active Assoc. Cells")
  plt.xlabel("Position 1")
  plt.ylabel("Position 2")
  plt.savefig(OVERLAP_FULL_SAVE_FILE, dpi=500)


def plot_position_spikes(position, grid_cells: GC_Population, assoc_cells: Association, save_file: str):
  POSITION_FULL_SAVE_FILE = os.path.join("./saves/plots", save_file)
  fig, axs = plt.subplots(1, 2, figsize=(14, 7))

  # Plot grid cells spikes
  gc_spikes = grid_cells.maze_spike_trains[position]
  axs[0].bar(np.arange(gc_spikes.shape[0]), gc_spikes.sum(axis=1), color='blue')
  axs[0].set_ylim(0, 9)
  axs[0].set_title(f"Grid Cells Spikes at Position {position}")
  axs[0].set_ylabel("Time (ms)")
  axs[0].set_xlabel("Neuron ID")

  # Plot association cells spikes
  assoc_spikes = assoc_cells.maze_spike_trains[position]
  ssum = assoc_spikes.sum(axis=0)
  axs[1].bar(np.arange(assoc_spikes.shape[1]), ssum, color='red')
  axs[1].set_ylim(0, max(ssum)+1)
  axs[1].set_title(f"Association Cells Spikes at Position {position}")
  axs[1].set_xlabel("Time (s)")
  axs[1].set_ylabel("Neuron ID")

  plt.tight_layout()
  plt.savefig(POSITION_FULL_SAVE_FILE, dpi=500)


def plot_position_active_cells(position, active_GC: dict, active_assoc: dict, grid_cells: GC_Population, assoc_cells: Association, save_file: str, top_n=10):
  POSITION_FULL_SAVE_FILE = os.path.join("./saves/plots", save_file)

  gc_spike_trains = grid_cells.maze_spike_trains[position]
  top_n_active_gc = active_GC[position][0][:top_n]
  position_active_gc_spike_trains = gc_spike_trains[top_n_active_gc, :]

  assoc_spike_trains = assoc_cells.maze_spike_trains[position]
  top_n_active_assoc = active_assoc[position][0][:top_n]
  position_active_assoc_spike_trains = assoc_spike_trains.T[top_n_active_assoc, :]

  # Plot active top_n grid cell spike trains
  fig, axs = plt.subplots(1, 2, figsize=(14, 7))
  axs[0].imshow(position_active_gc_spike_trains, cmap='binary', aspect='auto', interpolation='nearest',)
  y_vals = np.arange(0.5, position_active_gc_spike_trains.shape[0], 1)
  axs[0].hlines(y=y_vals, xmin=0, xmax=position_active_gc_spike_trains.shape[1], colors='white', linewidth=5)
  axs[0].set_title(f"Active Grid Cells at Position {position}")
  axs[0].set_xlabel("Time (ms)")
  axs[0].set_ylabel("Neuron ID")
  y_labels = top_n_active_gc
  y_ticks = np.arange(0, len(y_labels), 1)
  x_ticks = np.arange(0, position_active_gc_spike_trains.shape[1]+1, 100)
  axs[0].set_yticks(y_ticks, labels=y_labels)
  axs[0].set_xticks(x_ticks)
  axs[0].set_xlim(-10, position_active_gc_spike_trains.shape[1]+10)

  # Plot active association cells
  axs[1].imshow(position_active_assoc_spike_trains, cmap='binary', aspect='auto', interpolation='nearest',)
  y_vals = np.arange(0.5, position_active_assoc_spike_trains.shape[0], 1)
  axs[1].hlines(y=y_vals, xmin=0, xmax=position_active_assoc_spike_trains.shape[1], colors='white', linewidth=5)
  axs[1].set_title(f"Active Assoc Cells at Position {position}")
  axs[1].set_xlabel("Time (ms)")
  axs[1].set_ylabel("Neuron ID")
  y_labels = top_n_active_gc
  y_ticks = np.arange(0, len(y_labels), 1)
  x_ticks = np.arange(0, position_active_assoc_spike_trains.shape[1]+1, 100)
  axs[1].set_yticks(y_ticks, labels=y_labels)
  axs[1].set_xticks(x_ticks)
  axs[1].set_xlim(-10, position_active_gc_spike_trains.shape[1]+10)

  plt.tight_layout()
  plt.savefig(POSITION_FULL_SAVE_FILE, dpi=500)


def plot_maze(maze_env: Grid_Cell_Maze_Environment, agent_coords, agent_img, save_file: str):
  MAZE_FULL_SAVE_FILE = os.path.join("./saves/plots", save_file)
  fig, ax = plt.subplots(figsize=(7, 7))
  maze_env.plot(ax=ax, agent_coords=agent_coords, agent_img=agent_img)
  ax.set_title("Maze Environment")
  ax.set_xlabel("X Position")
  ax.set_ylabel("Y Position")
  plt.savefig(MAZE_FULL_SAVE_FILE, dpi=500)


def plot_gc_grids(maze_env: Grid_Cell_Maze_Environment, grid_cell_inds, grid_cell_colors, grid_cells: GC_Population, save_file: str):
  GRID_FULL_SAVE_FILE = os.path.join("./saves/plots", save_file)
  fig, ax = plt.subplots(figsize=(7, 7))
  maze_env.plot(ax=ax)
  ax.set_title("Grid Cell Grids")
  ax.set_xlabel("X Position")
  ax.set_ylabel("Y Position")

  plot_x_range = (-maze_env.width*2, maze_env.width*2)
  plot_y_range = (-maze_env.height*2, maze_env.height*2)
  for i, c in zip(grid_cell_inds, grid_cell_colors):
    grid_cell = grid_cells.cell_by_index(i)
    grid_cell.plot_peaks(plot_x_range, plot_y_range, color=c, ax=ax)
    grid_cell.plot_closest_contour((4,4), ax=ax)

  plt.xlim(-0.5, maze_env.width-0.5)
  plt.ylim(-0.5, maze_env.height-0.5)

  plt.savefig(GRID_FULL_SAVE_FILE, dpi=500)


def plot_cell_voltage(voltages: np.ndarray, gc_spikes: np.ndarray, assoc_spikes: np.ndarray, position: tuple, asc_cells: np.ndarray,
                      grid_cells: np.ndarray, colors, threshold: int, time_range: tuple = None):
  position_voltages = voltages[position]

  if time_range is None:
    time_range = [0, position_voltages.shape[0]]

  for c in asc_cells:
    GRID_FULL_SAVE_FILE = os.path.join("./saves/plots/voltage_plots", f"pos_{position}_cell_{c}_trange_{time_range}.png")
    cell_voltage = position_voltages.T[c][time_range[0]:time_range[1]]
    grid_spikes = gc_spikes[position].T[:, time_range[0]:time_range[1]]
    asc_spikes = assoc_spikes[position].T[:, time_range[0]:time_range[1]]

    # Adjust voltages to represent spikes
    cell_voltage[np.where(asc_spikes[c] == 1)] = -30  # Spike voltage

    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(2, 1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax1.plot(cell_voltage, color='blue')
    ax1.margins(x=0)
    ax1.set_title(f"Membrane Voltage of Cell {c} at Position {position}")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Membrane Voltage (mV)")
    ax1.axhline(y=threshold, color='red', linestyle='--', linewidth=1)
    ax1.set_xticks(np.arange(0, cell_voltage.shape[0]+1, 100))
    ax1.set_ylim(-71, -43)

    ax2.imshow(grid_spikes, aspect='auto', cmap="binary")
    ax2.set_title(f"Raster Plot of Grid Cells")
    ax2.set_xlabel("Time (ms)")
    for gc, c in zip(grid_cells, colors):
      spike_inds = np.where(grid_spikes[gc] >= 1)[0]
      y = np.full(len(spike_inds), gc)
      plt.scatter(spike_inds, y, color=c, s=3)
    ax2.margins(x=0)
    ax2.set_xticks(np.arange(0, cell_voltage.shape[0] + 1, 100))

    plt.tight_layout()
    plt.savefig(GRID_FULL_SAVE_FILE, dpi=500)
    plt.clf()

  return

def plot_training_history_avg(histories, save_file, p1_len, p2_len, p3_len, p4_len, p5_len):
  # Determine start position based on the provided lengths
  p1_start = 0
  p1_end = 0+p1_len
  p2_start = p1_len
  p2_end = p2_start+p2_len
  p3_start = p2_end
  p3_end = p3_start+p3_len
  p4_start = p3_end
  p4_end = p4_start+p4_len
  p5_start = p4_end
  p5_end = p5_start+p5_len

  FULL_SAVE_FILE = os.path.join("./saves/plots", save_file)

  # Extract episode lengths and average delta Q values
  ep_lengths = np.array([[len(ep) for ep in h] for h in histories])
  avg_delta_q = np.array([[
    np.mean([ep[i][4] if ep[i][4] != -1 else -0.001 for i in range(len(ep)-1)])
    for ep in h] for h in histories])

  fig = plt.figure(figsize=(8, 6))
  gs = gridspec.GridSpec(2, 1)
  ax1 = fig.add_subplot(gs[0, 0])
  ax2 = fig.add_subplot(gs[1, 0])

  # Plot episode lengths
  length_mean = np.mean(ep_lengths, axis=0)
  length_std = np.std(ep_lengths, axis=0)
  ax1.set_title("Average Learning Curve")
  ax1.set_xlabel("Episode")
  ax1.set_ylabel("Average Steps")
  ax1.plot(np.arange(p1_start, p1_end+1), length_mean[p1_start:p1_end+1], color='blue', label="Warmup")
  ax1.fill_between(np.arange(p1_start, p1_end+1),
                   length_mean[p1_start:p1_end+1] - length_std[p1_start:p1_end+1],
                   length_mean[p1_start:p1_end+1] + length_std[p1_start:p1_end+1],
                   color='blue', alpha=0.3)
  ax1.plot(np.arange(p2_start, p2_end), length_mean[p2_start:p2_end], color='red', label="Active Learning")
  ax1.fill_between(np.arange(p2_start, p2_end),
                   length_mean[p2_start:p2_end] - length_std[p2_start:p2_end],
                   length_mean[p2_start:p2_end] + length_std[p2_start:p2_end],
                   color='red', alpha=0.3)
  ax1.plot(np.arange(p3_start-1, p3_end+1), length_mean[p3_start-1:p3_end+1], color='blue', label="Warmup")
  ax1.fill_between(np.arange(p3_start-1, p3_end+1),
                   length_mean[p3_start-1:p3_end+1] - length_std[p3_start-1:p3_end+1],
                   length_mean[p3_start-1:p3_end+1] + length_std[p3_start-1:p3_end+1],
                   color='blue', alpha=0.3)
  ax1.plot(np.arange(p4_start, p5_end), length_mean[p4_start:p5_end], color='red')
  ax1.fill_between(np.arange(p4_start, p5_end),
                   length_mean[p4_start:p5_end] - length_std[p4_start:p5_end],
                   length_mean[p4_start:p5_end] + length_std[p4_start:p5_end],
                   color='red', alpha=0.3)
  # plt.fill_between(x, mean - std, mean + std, color='blue', alpha=0.3, label='±1 Std Dev')
  ax1.axvline(x=p3_start-1, color='g', linestyle='--', label="Start Position Change")
  ax1.axvline(x=p5_start-1, color='g', linestyle='--')
  ax1.legend(loc="lower left")

  # Plot average delta Q
  dq_mean = np.mean(avg_delta_q, axis=0)
  dq_std = np.std(avg_delta_q, axis=0)
  ax2.set_title("Average Delta Q")
  ax2.set_xlabel("Episode")
  ax2.set_ylabel("Average Delta Q")
  ax2.plot(dq_mean, color='purple', label="Average Delta-Q")
  ax2.fill_between(np.arange(len(dq_mean)),
                   dq_mean - dq_std,
                   dq_mean + dq_std,
                   color='purple', alpha=0.3)
  ax2.axvline(x=p3_start-1, color='g', linestyle='--', label="Start Position Change")
  ax2.axvline(x=p5_start-1, color='g', linestyle='--')
  ax2.legend()
  plt.tight_layout()
  plt.savefig(FULL_SAVE_FILE, dpi=100)


def plot_GC_module(grid_cells, maze, save_file='gc_modules.png'):
  FULL_SAVE_FILE = os.path.join("./saves/plots", save_file)

  fig = plt.figure(figsize=(8, 8))
  gs = gridspec.GridSpec(2, 2)
  ax1 = fig.add_subplot(gs[0, 0])
  ax2 = fig.add_subplot(gs[1, 0])
  ax3 = fig.add_subplot(gs[0, 1])
  ax4 = fig.add_subplot(gs[1, 1])

  module_inds = [0]
  colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'yellow', 'magenta', 'teal', 'lime', 'navy', 'maroon']
  for i, ax in zip(module_inds, [ax1]):
    gc_module = grid_cells.modules[i]
    maze.plot(ax=ax)
    # for i, gc in enumerate(gc_module.grid_cells[0:]):
    #   gc.plot_grid_lines([-30, +30], [-30, +30], color=colors[i], ax=ax)
    for i, gc in enumerate(gc_module.grid_cells[0:]):
      gc.plot_peaks([-30, +30], [-30, +30], color=colors[i], ax=ax)

    ax.set_xlim(-0.5, maze.width - 0.5)
    ax.set_ylim(-0.5, maze.height - 0.5)

  plt.tight_layout()
  plt.savefig(FULL_SAVE_FILE, dpi=300)

if __name__ == '__main__':
  with open('saves/grid_cells/7_7_grid_spikes.pkl', 'rb') as f:
    grid_cells = pkl.load(f)
  # with open('./saves/assoc_cells/7_7_assoc_spikes.pkl', 'rb') as f:
  #   assoc_cells = pkl.load(f)
  with open('./saves/mazes/7_7_maze.pkl', 'rb') as f:
    maze = pkl.load(f)
  # with open('./saves/history/7_7_history.pkl', 'rb') as f:
  #   history = pkl.load(f)

  plot_GC_module(grid_cells, maze, save_file='7_7_gc_modules.png')

  # choice_location = (4, 4)
  # active_assoc_cells = np.where(assoc_cells.maze_spike_trains[choice_location].T.sum(1) > 6)[0]
  # active_grid_cells = np.where(grid_cells.maze_spike_trains[choice_location].sum(1) > 6)[0]
  # gc_colors = plt.cm.get_cmap('tab20', len(active_grid_cells)).colors
  #
  # if os.path.exists("./saves/plots/voltage_plots"):
  #   shutil.rmtree("./saves/plots/voltage_plots")
  # os.mkdir("./saves/plots/voltage_plots")
  # plot_cell_voltage(assoc_cells.maze_voltages, grid_cells.maze_spike_trains.transpose(0, 1, 3, 2),
  #                   assoc_cells.maze_spike_trains, choice_location,
  #                   asc_cells=active_assoc_cells[0:10], grid_cells=active_grid_cells, colors=gc_colors, threshold=-45)
  #
  # plot_cell_voltage(assoc_cells.maze_voltages, grid_cells.maze_spike_trains.transpose(0, 1, 3, 2),
  #                   assoc_cells.maze_spike_trains, choice_location,
  #                   asc_cells=[1678], grid_cells=active_grid_cells, colors=gc_colors, threshold=-45, time_range=(100, 300))
  #
  #
  # active_gc = get_active_neurons(maze.get_shape(), grid_cells.maze_spike_trains, threshold=6)
  # active_assoc = get_active_neurons(maze.get_shape(), assoc_cells.maze_spike_trains.numpy().transpose(0, 1, 3, 2), threshold=6)
  # plot_position_active_cells(choice_location, active_gc, active_assoc, grid_cells, assoc_cells, '7_7_position_active_cells.png')
  # plot_active_grid_cells(active_gc, grid_cells.n_cells, '7_7_GC_analysis.png', maze_env=maze)
  # plot_active_assoc_cells(active_assoc, '7_7_AC_analysis.png', maze_env=maze)
  # plot_position_spikes(choice_location, grid_cells, assoc_cells, '7_7_position_spikes.png')
  # plot_maze(maze, choice_location, './mouse.png', '7_7_maze.png')
  # plot_training_history(history, '7_7_history.png', 25, 5, 25, 5, 5)
  # plot_gc_grids(maze, [99, 90, 81, 80], ['blue', 'green', 'red', 'purple'], grid_cells, '7_7_gc_grids.png')

