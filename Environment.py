import random
from collections import deque

import matplotlib.patches
import numpy as np
from labyrinth.generate import DepthFirstSearchGenerator
from labyrinth.grid import Cell, Direction
from labyrinth.maze import Maze as _Maze
from labyrinth.solve import MazeSolver
from matplotlib.pyplot import plot as plt

import pickle as pkl
import matplotlib.pyplot as plt
from torch import optim


# Modified to allow for start cell setting
class Maze(_Maze):
  def __init__(self, width: int = 10, height: int = 10, generator: DepthFirstSearchGenerator = DepthFirstSearchGenerator()):
    self._start_cell = None  # Initialize start cell
    super().__init__(width, height, generator)
    self._start_cell = self[0, 0]  # Set default start cell

  @property
  def start_cell(self):
    if self._start_cell is None:
      return self[0, 0]  # Default start cell if not set

    return self._start_cell

  @start_cell.setter
  def start_cell(self, cell: Cell):
    self._start_cell = cell


class Maze_Environment():
  def __init__(self, width, height, trace_length=5):

    # Generate basic maze & solve
    self.width = width
    self.height = height
    self.maze = Maze(width=width, height=height, generator=DepthFirstSearchGenerator())
    self.solver = MazeSolver()
    self.path = self.solver.solve(self.maze)
    self.maze.path = self.path    # No idea why this is necessary
    self.agent_cell = self.maze.start_cell
    self.num_actions = 4
    self.path_history = []  # (state, reward, done, info)
    self.trace_length = trace_length

    # Add reward traces to cells
    self.reward_trace = self.calculate_reward_trace()

  # Paths of length 'trace_length' emanating from goal cell providing agent with reward
  def calculate_reward_trace(self):
    reward_trace = np.full((self.height, self.width), np.inf)
    goal = self.maze.end_cell.coordinates
    queue = deque([(goal, 0)])  # (cell coordinates, distance)
    visited = set()

    while queue:
      (x, y), dist = queue.popleft()
      if (x, y) in visited or dist > self.trace_length:
        continue
      visited.add((x, y))
      reward_trace[y, x] = dist

      for direction in Direction:
        if direction in self.maze[x, y].open_walls:
          neighbor = self.maze.neighbor(self.maze[x, y], direction)
          queue.append((neighbor.coordinates, dist + 1))

    # Normalize reward trace to go from large to small values
    max_dist = np.max(reward_trace[reward_trace != np.inf])
    reward_trace = max_dist - reward_trace
    reward_trace[reward_trace > self.trace_length] = 0  # Cut off at trace_length
    reward_trace[reward_trace == -np.inf] = 0  # Replace np.inf with 0
    return reward_trace.T

  def plot(self, agent_coords=None, path_color='blue', q_table: dict = None, agent_img=None, line_width: int = 1.5, state_behavior: np.ndarray = None, zorder=0, ax=None):
    if ax is None:
      fig, ax = plt.subplots()

    # Axis
    ax.set_xlim(-0.5, self.width - 0.5)
    ax.set_ylim(-0.5, self.height - 0.5)
    ax.set_xticks(np.arange(0, self.width, 1))
    ax.set_yticks(np.arange(0, self.height, 1))

    # Transpose agent coordinates (just how the maze is stored)
    if agent_coords is not None:
      agent_coords = agent_coords[::-1]

    # Box around maze
    ax.plot([-0.5, self.width-1+0.5], [-0.5, -0.5], color='black', linewidth=line_width, zorder=zorder+.1)
    ax.plot([-0.5, self.width-1+0.5], [self.height-1+0.5, self.height-1+0.5], color='black', linewidth=line_width, zorder=zorder+.1)
    ax.plot([-0.5, -0.5], [-0.5, self.height-1+0.5], color='black', linewidth=line_width, zorder=zorder+.1)
    ax.plot([self.width-1+0.5, self.width-1+0.5], [-0.5, self.height-1+0.5], color='black', linewidth=line_width, zorder=zorder+.1)

    # Plot maze
    for row in range(self.height):
      for column in range(self.width):
        # Path
        cell = self.maze[column, row]  # Transpose maze coordinates (just how the maze is stored)
        if cell == self.maze.start_cell:
          square = matplotlib.patches.Rectangle((row - 0.5, column - 0.5), 1, 1, linewidth=0, color='green', alpha=1, zorder=zorder)
          ax.text(row, column, 'START', color='white', fontsize=7, ha='center', va='center', fontweight='bold', zorder=zorder+.1)
          ax.add_patch(square)
        elif cell == self.maze.end_cell:
          square = matplotlib.patches.Rectangle((row - 0.5, column - 0.5), 1, 1, linewidth=0, color='purple', alpha=1, zorder=zorder)
          ax.text(row, column, 'GOAL', color='white', fontsize=7, ha='center', va='center', fontweight='bold', zorder=zorder+.1)
          ax.add_patch(square)
        elif cell in self.maze.path and path_color is not None:
          # ax.plot(row, column, marker='o', color=path_color, markersize=5)
          square = matplotlib.patches.Rectangle((row-0.5, column-0.5), 1, 1, linewidth=0, color=path_color, alpha=0.25, zorder=zorder)
          ax.add_patch(square)

        # Walls
        if Direction.S not in cell.open_walls:
          ax.plot([row-0.5, row+0.5], [column+0.5, column+0.5], color='black', linewidth=line_width, zorder=zorder+.1)
        if Direction.E not in cell.open_walls:
          ax.plot([row+0.5, row+0.5], [column-0.5, column+0.5], color='black', linewidth=line_width, zorder=zorder+.1)

        # Table
        if q_table:
          if (column, row) in q_table:
            q_values = q_table[(column, row)]
          else:
            q_values = np.zeros(self.num_actions) # Actions are N, E, S, W
          ax.text(row, column-0.4, f'{q_values[0]:.2f}S', ha='center', va='center', zorder=zorder)
          ax.text(row+0.4, column, f'{q_values[1]:.2f}E', ha='center', va='center', rotation=90, zorder=zorder)
          ax.text(row, column+0.4, f'{q_values[2]:.2f}N', ha='center', va='center', zorder=zorder)
          ax.text(row-0.4, column, f'{q_values[3]:.2f}W', ha='center', va='center', rotation=90, zorder=zorder)

        if state_behavior is not None:
          activity = state_behavior[column, row]
          ax.text(row, column+0.2, f'{activity[0]}', ha='center', va='center', zorder=zorder)
          ax.text(row+0.2, column, f'{activity[1]}', ha='center', va='center', zorder=zorder)
          ax.text(row, column-0.2, f'{activity[2]}', ha='center', va='center', zorder=zorder)
          ax.text(row-0.2, column, f'{activity[3]}', ha='center', va='center', zorder=zorder)

    # Plot agent
    if agent_img and agent_coords is not None:
      img = matplotlib.image.imread(agent_img)
      ax.imshow(img, extent=(agent_coords[0]-0.5, agent_coords[0]+0.5, agent_coords[1]-0.5, agent_coords[1]+0.5), zorder=zorder)
    elif agent_coords is not None:
      ax.plot(agent_coords[0], agent_coords[1], 'yo')

    return ax

  def reset(self):
    self.agent_cell = self.maze.start_cell
    self.path_history = []
    return self.agent_cell, {}

  def is_end(self, state):
    return self.maze[state[0], state[1]] == self.maze.end_cell

  def virtual_step(self, state, action):
    # Transform action into Direction
    if action == 0:
      action = Direction.N
    elif action == 1:
      action = Direction.E
    elif action == 2:
      action = Direction.S
    elif action == 3:
      action = Direction.W

    # Transform state into Cell
    state = self.maze[state[0], state[1]]

    # Check if action runs into wall
    if action not in state.open_walls:
      return state.coordinates, -1, False, {}

    # Simulate moving agent
    else:
      prev_cell = state
      next_state = self.maze.neighbor(state, action)
      if state == self.maze.end_cell:    # Check if agent has reached the end
        return next_state.coordinates, 1, True, {}
      else:
        prev_trace = self.reward_trace[prev_cell.coordinates]
        new_trace = self.reward_trace[next_state.coordinates]
        reward = 1 if new_trace > prev_trace else 0
        return next_state.coordinates, reward, False, {}

  # Takes action
  # Returns next state, reward, done, info
  def step(self, action):
    # Transform action into Direction
    if action == 0:
      action = Direction.N
    elif action == 1:
      action = Direction.E
    elif action == 2:
      action = Direction.S
    elif action == 3:
      action = Direction.W

    # Check if action runs into wall
    if action not in self.agent_cell.open_walls:
      self.path_history.append((self.agent_cell.coordinates, -1, False, action))
      return self.agent_cell, -1, False, {}

    # Move agent
    else:
      prev_cell = self.agent_cell
      self.agent_cell = self.maze.neighbor(self.agent_cell, action)
      if self.agent_cell == self.maze.end_cell:    # Check if agent has reached the end
        self.path_history.append((self.agent_cell.coordinates, 1, True, action))
        return self.agent_cell, 1, True, {}
      else:
        prev_trace = self.reward_trace[prev_cell.coordinates]
        new_trace = self.reward_trace[self.agent_cell.coordinates]
        reward = 1 if new_trace > prev_trace else 0
        self.path_history.append((self.agent_cell.coordinates, reward, False, action))
        return self.agent_cell, reward, False, {}

  def save(self, filename):
    with open(filename, 'wb') as f:
      pkl.dump(self, f)

  def get_shape(self):
    return self.maze.width, self.maze.height


class Grid_Cell_Maze_Environment(Maze_Environment):
  def __init__(self, width, height, in_spikes=None, trace_length=5, load_from=None):
    if load_from is not None:
      with open(load_from, 'rb') as f:
        super().__init__(width, height, trace_length)
        obj_data = pkl.load(f)
        self.__dict__.update(obj_data.__dict__)
    else:
      super().__init__(width, height, trace_length)

    self.reward_trace = self.calculate_reward_trace()
    self.samples = in_spikes

  # Returns:
  # - Spike train of grid cell corresponding to agent's position
  # - Reset coordinates (x, y)
  # - info: (empty)
  def reset(self):
    cell, info = super().reset()
    return self.state_to_grid_cell_spikes(cell), cell.coordinates, info

  # Move in maze
  def step(self, action):
    obs, reward, done, info = super().step(action)
    coords = obs.coordinates
    obs = self.state_to_grid_cell_spikes(obs)
    return obs, reward, done, coords

  # Return stored spike trains at coordinate location
  def state_to_grid_cell_spikes(self, cell):
    return self.samples[cell.coordinates]

  def set_spike_trains(self, spike_trains):
    self.samples = spike_trains


if __name__ == '__main__':
  np.random.seed(0)
  with open('saves/3000_res_10_10_maze.pkl', 'rb') as f:
    in_spikes = pkl.load(f)
  maze = Grid_Cell_Maze_Environment(10, 10, in_spikes, trace_length=0)
  fig, ax = plt.subplots(figsize=(10, 10))
  maze.plot(agent_coords=(4, 2), path_color='blue', agent_img='mouse.png', ax=ax)
  plt.savefig('maze_10_10_plot.png', dpi=200, bbox_inches='tight')
  # plt.clf()
  # maze.maze.start_cell = maze.maze[6, 2]
  # maze.maze.path = maze.solver.solve(maze.maze)
  # maze.plot(agent_coords=(0, 0), path_color='purple')
  # plt.savefig('maze_10_10_plot_alt.png', dpi=200)
  with open('maze_10_10.pkl', 'wb') as f:
    pkl.dump(maze, f)