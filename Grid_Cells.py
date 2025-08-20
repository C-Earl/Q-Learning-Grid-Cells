from scipy.stats import multivariate_normal
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import math

# https://stackoverflow.com/questions/74519927/best-way-to-rotate-and-translate-a-set-of-points-in-python
def rotate_matrix(a):
  return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


class Grid_Cell:
  def __init__(self, x_offset, y_offset, rotation, scale=1, sharpness=1, max_firing_rate=8):
    self.x_offset = x_offset    # Offset in x-direction
    self.y_offset = y_offset    # Offset in y-direction
    self.rotation = rotation    # Rotation in radians
    self.scale = scale      # How far apart peaks are
    self.sharpness = sharpness  # How 'sharp' distribution for firing peaks are
    self.max_activity = max_firing_rate  # Max firing rate (hz) of cell

    # Sharpness should not be below 1
    if self.sharpness < 1:
      raise ValueError(f"Sharpness should not be below 1; got {self.sharpness}")

    # Ensure range doesn't overlap other peaks too much
    d = (1/self.sharpness) * (self.scale * 0.5)  # PDF should be near 0 at roughly half-way point between peaks
    var = (d/3)**2  # 99.7% of values within 3 standard deviations
    self.cov = [[var, 0], [0, var]]

    # Save max activity for normalization
    self.max_activity = multivariate_normal.pdf([0, 0], [0, 0], self.cov)

  # Define base grid indices for hexagonal grid
  def base_grid_indices(self, x_range, y_range):
    grid_indices = np.mgrid[x_range[0]:x_range[1],
                            y_range[0]:y_range[1]].transpose(1, 2, 0).astype(float)
    return grid_indices

  def base_to_hex_transform(self, p):
    # Transform basic grid to hexagonal grid
    x_flatten = p.reshape(-1, p.shape[-1])[:, 0].copy()
    y_flatten = p.reshape(-1, p.shape[-1])[:, 1].copy()
    for i, y_ in enumerate(y_flatten):
      if y_ % 2 == 0:
        x_flatten[i] += 0.5

    x = x_flatten.reshape(p.shape[:-1])
    y = y_flatten.reshape(p.shape[:-1])
    y *= np.sqrt(0.75)
    return np.stack((x, y), axis=-1)

  # Translate from grid cell coordinates to plot coordinates
  def hex_to_plot_transform(self, p):
    x = p[0] * self.scale * math.cos(self.rotation) - p[1] * self.scale * math.sin(self.rotation) + self.x_offset
    y = p[0] * self.scale * math.sin(self.rotation) + p[1] * self.scale * math.cos(self.rotation) + self.y_offset
    return [x, y]

  # Translate from plot coordinates to grid cell coordinates
  def plot_to_hex_transform(self, p):
    x = ((p[0] - self.x_offset) * math.cos(self.rotation) + (p[1] - self.y_offset) * math.sin(self.rotation)) / self.scale
    y = ((p[1] - self.y_offset) * math.cos(self.rotation) - (p[0] - self.x_offset) * math.sin(self.rotation)) / self.scale
    return [x, y]

  # Determine range of grid indices for a given range of plot coordinates
  def grid_indices_range(self, x_range, y_range):
    # Transform plot coordinates to grid cell coordinates
    grid_ranges = [self.plot_to_hex_transform([x, y]) for x, y in [(x_range[0], y_range[0]),
                                                                   (x_range[1], y_range[0]),
                                                                   (x_range[0], y_range[1]),
                                                                   (x_range[1], y_range[1])]]
    grid_x_range = [min(grid_ranges, key=lambda p: p[0])[0], max(grid_ranges, key=lambda p: p[0])[0]]
    grid_y_range = [min(grid_ranges, key=lambda p: p[1])[1], max(grid_ranges, key=lambda p: p[1])[1]]
    # Ensure ranges are integers
    grid_x_range = [math.floor(grid_x_range[0]), math.ceil(grid_x_range[1])]
    grid_y_range = [math.floor(grid_y_range[0]), math.ceil(grid_y_range[1])]
    return grid_x_range, grid_y_range

  # Get closest firing peak to pos
  def find_closest_peak(self, pos):
    # Locate closest firing peak (relative to grid-cell coordinates)
    grid_x, grid_y = self.plot_to_hex_transform(pos)
    p1 = [math.floor(grid_x), math.floor(grid_y)]
    p2 = [math.floor(grid_x), math.ceil(grid_y)]
    p3 = [math.ceil(grid_x), math.floor(grid_y)]
    p4 = [math.ceil(grid_x), math.ceil(grid_y)]
    for p in [p1, p2]:
      if p[1] % 2 == 0:
        p[0] += 0.5
    for p in [p3, p4]:
      if p[1] % 2 == 0:
        p[0] += 0.5
    p1[1] *= np.sqrt(0.75)
    p2[1] *= np.sqrt(0.75)
    p3[1] *= np.sqrt(0.75)
    p4[1] *= np.sqrt(0.75)

    p1_t = self.hex_to_plot_transform(p1)
    p2_t = self.hex_to_plot_transform(p2)
    p3_t = self.hex_to_plot_transform(p3)
    p4_t = self.hex_to_plot_transform(p4)

    # Visual plots for peaks and position
    # self.plot_peaks([0, 10], [0, 10], "blue")
    # plt.plot(p1_t[0], p1_t[1], '.', color='brown')
    # plt.plot(p2_t[0], p2_t[1], '.', color='black')
    # plt.plot(p3_t[0], p3_t[1], '.', color='gray')
    # plt.plot(p4_t[0], p4_t[1], '.', color='pink')
    # plt.plot(pos[0], pos[1], '.', color='green')

    # Generate/Sample activity around closest firing peak
    peaks = np.array([p1_t, p2_t, p3_t, p4_t])
    distances = np.linalg.norm(peaks - pos, axis=1, ord=2)
    closest_peak = peaks[np.argmin(distances)]
    return closest_peak

  # Generate activity for a given position
  def activity(self, pos):
    closest_peak = self.find_closest_peak(pos)
    x_p, y_p = closest_peak
    mvn = multivariate_normal(mean=(x_p, y_p), cov=self.cov)
    activity = mvn.pdf(pos) / self.max_activity  # Normalize so all activity in [0, 1]
    if activity < 0.1:
      return 0
    else:
      return activity

  # Plot firing peaks for grid cell
  def plot_peaks(self, x_range, y_range, color='blue', contours=False, pos=None, fig=None, ax=None):
    # Indices relative to grid-cells
    # Do this to find range of firing peaks to plot
    grid_x_range, grid_y_range = self.grid_indices_range(x_range, y_range)
    grid_indices = self.base_grid_indices(grid_x_range, grid_y_range)
    hex_indices = self.base_to_hex_transform(grid_indices)
    x, y = self.hex_to_plot_transform([hex_indices[:, :, 0], hex_indices[:, :, 1]])
    plt_indices = np.stack((x, y), axis=-1)

    # Plot peaks
    x = plt_indices[:,:, 0].flatten()
    y = plt_indices[:,:, 1].flatten()
    plt.scatter(x, y, s=5, alpha=1, color=color)
    # x_ll, y_ll = plt_indices[1, 0]
    for i in range(plt_indices.shape[0]-1):
      for j in range(plt_indices.shape[1]-1):
        x, y = plt_indices[i, j]
        x_r, y_r = plt_indices[i, j+1]
        x_u, y_u = plt_indices[i+1, j]
        ax.plot([x, x_r], [y, y_r], '-', alpha=0.2, color=color, markersize=5)
        ax.plot([x, x_u], [y, y_u], '-', alpha=0.2, color=color, markersize=5)
        if grid_indices[i, j, 1] % 2 == 0:
          x_dr, y_dr = plt_indices[i+1, j+1]
          ax.plot([x, x_dr], [y, y_dr], '-', alpha=0.2, color=color, markersize=5)
          if j > 0:
            x_dl, y_dl = plt_indices[i+1, j-1]
            ax.plot([x, x_dl], [y, y_dl], '-', alpha=0.2, color=color, markersize=5)

    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])

    # Plot contour around peaks
    if contours:
      for i in range(plt_indices.shape[0]):
        for j in range(plt_indices.shape[1]):
          x, y = plt_indices[i, j]
          mvn = multivariate_normal(mean=(x, y), cov=self.cov)
          x_r = np.linspace(x - self.scale, x + self.scale, 100)
          y_r = np.linspace(y - self.scale, y + self.scale, 100)
          X, Y = np.meshgrid(x_r, y_r)
          Z = mvn.pdf(np.dstack((X, Y)))
          Z = Z / self.max_activity
          cont_map = ax.contour(X, Y, Z, levels=10)
      # fig.colorbar(cont_map)

    # Plot position
    if pos:
      ax.plot(pos[0], pos[1], 'o', color='red')
      # print('Activity:', self.activity(pos))

  def plot_closest_contour(self, pos, ax=None):
    if ax is None:
      fig, ax = plt.subplots()
    closest_peak = self.find_closest_peak(pos)
    x, y = closest_peak
    mvn = multivariate_normal(mean=(x, y), cov=self.cov)
    x_r = np.linspace(x - self.scale, x + self.scale, 100)
    y_r = np.linspace(y - self.scale, y + self.scale, 100)
    X, Y = np.meshgrid(x_r, y_r)
    Z = mvn.pdf(np.dstack((X, Y))) / self.max_activity
    levels = np.linspace(0.1, 1.0, 10)
    cont_map = ax.contour(X, Y, Z, levels=levels)
    return cont_map


# Module of Grid Cells
class GC_Module:
  def __init__(self, n_cells, x_offsets, y_offsets, rotations, scales, sharpnesses, max_firing_rates=None, colors=None):
    max_firing_rates = max_firing_rates if max_firing_rates is not None else [8] * n_cells
    self.grid_cells = [Grid_Cell(x_offsets[i], y_offsets[i],
                       rotations[i], scales[i], sharpnesses[i],
                       max_firing_rates[i]) for i in range(n_cells)]
    self.n_cells = n_cells
    self.x_offsets = x_offsets
    self.y_offsets = y_offsets
    self.rotations = rotations
    self.scales = scales
    self.sharpnesses = sharpnesses
    self.max_firing_rates = max_firing_rates
    if colors is None:
      self.colors = []
      for cmap_name in ['tab20', 'tab20b', 'tab20c', 'Set1', 'Set2', 'Set3', 'Paired', 'Pastel1', 'Pastel2',
      'Accent', 'Dark2']:
        cmap = plt.get_cmap(cmap_name)
        self.colors.extend([cmap(i) for i in np.linspace(0, 1, 20)])
    else:
      self.colors = colors

  # Generate Grid Cell activities for given position
  def activity(self, pos):
    return [gc.activity(pos) for gc in self.grid_cells]

  # Plot Grid Cell activity
  def plot_peaks(self, x_range, y_range, pos=None, contours=False, fig=None, ax=None):
    if ax is None:
      fig, ax = plt.subplots()

    for i, gc in enumerate(self.grid_cells):
      gc.plot_peaks(x_range, y_range, self.colors[i], pos=False, contours=False, fig=fig, ax=ax)
    
    if contours:
      for i, gc in enumerate(self.grid_cells):
        cont_map = gc.plot_closest_contour(pos, ax=ax)
      # fig.colorbar(cont_map)

    # Plot position
    if pos:
      ax.plot(pos[0], pos[1], 'o', color='red')
      ax.set_title('Activity: ' + str(self.activity(pos)))

    return ax


# Population of Modules
# Each module has same scale, rotation, sharpness, but varying offsets
class GC_Population:
  def __init__(self, num_modules, offsets_per_module, global_scale, scales, rotations, sharpnesses):
    self.num_modules = num_modules
    self.offsets_per_module = offsets_per_module
    self.global_scale = global_scale
    self.scales = scales
    self.rotations = rotations
    self.sharpnesses = sharpnesses
    self.modules = []
    self.cells_per_module = offsets_per_module**2
    self.maze_activities = None
    self.maze_spike_trains = None

    # Use for varying shades of same color per module
    base_colors =  [
      "#E6194B",  # Red
      "#3CB44B",  # Green
      "#FFE119",  # Yellow
      "#0082C8",  # Blue
      "#F58231",  # Orange
      "#911EB4",  # Purple
      "#46F0F0",  # Cyan
      "#F032E6",  # Magenta
      "#D2F53C",  # Lime
      "#FABEBE",  # Pink
      "#008080",  # Teal
      "#E6BEFF",  # Lavender
      "#AA6E28",  # Brown
      "#FFFAC8",  # Light Yellow
      "#800000",  # Maroon
      "#AAFFC3",  # Mint
      "#808000",  # Olive
      "#FFD8B1",  # Peach
      "#000080",  # Navy
      "#808080",  # Gray
      "#000000",  # Black
      "#FFFFFF",  # White
    ]

    for i, r in enumerate(rotations):
      for j in range(num_modules):
        # Create grid of uniformly spaced offsets
        x_offsets = []
        y_offsets = []
        s = scales[j] * global_scale
        offset_step_size = s / offsets_per_module
        base_x_offsets = []
        for k in range(1, offsets_per_module + 1):
          base_x_offsets.append((offset_step_size * k) + 20)    # to avoid universal overlap at origin
        base_y_offsets = base_x_offsets.copy()
        mod_x_offsets, mod_y_offsets = np.meshgrid(base_x_offsets, base_y_offsets)
        mod_x_offsets = mod_x_offsets.flatten()  # Transform into 1D arrays
        mod_y_offsets = mod_y_offsets.flatten()
        x_offsets.extend(mod_x_offsets)
        y_offsets.extend(mod_y_offsets)

        # Other parameters
        scale = [s] * self.cells_per_module
        rotation = [r] * self.cells_per_module
        sharpness = [sharpnesses] * self.cells_per_module

        b_color = mcolors.to_rgba(base_colors[j])   # Note: Colors will repeat for different rotations
        shades = [(b_color[0] * (k / self.cells_per_module),
                     b_color[1] * (k / self.cells_per_module),
                     b_color[2] * (k / self.cells_per_module),
                     1) for k in range(1, self.cells_per_module + 1)]

        # Create module
        self.modules.append(GC_Module(offsets_per_module**2, x_offsets, y_offsets, rotation, scale, sharpness, colors=['purple', 'red', 'green', 'blue', 'cyan', 'magenta', 'orange']))

    self.n_cells = sum([m.n_cells for m in self.modules])
    self.max_firing_rates = [8] * self.n_cells   # TODO: Make this scalable like in GC_Module

  # Generate Grid Cell activities for given position
  def activity(self, pos):
    activity = []
    for m in self.modules:
      activity.extend(m.activity(pos))
    return activity


  def plot_peaks(self, x_range, y_range, pos=None, contours=False, fig=None, ax=None):
    if ax is None:
      fig, ax = plt.subplots()

    for m in self.modules:
      ax = m.plot_peaks(x_range, y_range, pos=pos, contours=contours, fig=fig, ax=ax)

  def cell_by_index(self, index):
    # Get module and cell index from global index
    module_index = index // self.modules[0].n_cells
    cell_index = index % self.modules[0].n_cells
    return self.modules[module_index].grid_cells[cell_index]

  # Generate grid cell activity for each coordinate in the environment
  def grid_cell_activity_generator(self, maze_size):
    # Generate the spike activity for each coordinate in the environment
    x_range, y_range = maze_size
    activity = np.zeros((x_range, y_range, self.n_cells))
    for i in range(x_range):
      for j in range(y_range):
        pos = (i, j)
        a = self.activity(pos)
        activity[i, j] = a
    self.maze_activities = activity
    return activity

  def spike_train_generator(self, sim_time, max_firing_rates):
    # Note: gc_activity values in range of values [0, 1]
    time_denominator = 1000  # working in ms
    x_range, y_range, n_cells = self.maze_activities.shape
    spike_trains = np.zeros((*self.maze_activities.shape, sim_time))
    for i in range(x_range):
      for j in range(y_range):
        for k in range(n_cells):
          activity = self.maze_activities[i, j, k]  # in range [0, 1]
          max_freq = max_firing_rates[k]  # max firing rate for this grid cell
          spike_rate = activity * max_freq / time_denominator  # spike rate per ms
          spike_train = np.zeros(sim_time)
          if spike_rate != 0:
            step_size = int(1 / spike_rate)  # number of ms between spikes
            spike_train[::step_size] = 1
          spike_trains[i, j, k] = spike_train
    self.maze_spike_trains = spike_trains
    return spike_trains