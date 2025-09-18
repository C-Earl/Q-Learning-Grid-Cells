# Technical Imports #
import numpy as np
import os
import pickle as pkl
import torch
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import count

# Custom Objects & Functions #
from Environment import Grid_Cell_Maze_Environment
from Grid_Cells import GC_Population
from Association import Association
from STDP_Q_Learning import STDP_Q_Learning
from helper_functions import generate_grid_out_weights, generate_res_out_weights
from plotting import plot_training_history


def run(parameters: dict):
  ## Run Parameters ##
  ANIMATE_TRAINING = parameters['animate_training']
  VERBOSE_TRAINING = parameters['verbose_training']
  ONLY_PHASE_1 = parameters['only_phase_1']
  MAZE_FILE = parameters['maze_file']
  GRID_SPIKES_FILE = parameters['grid_spikes_file']
  ASSOC_SPIKES_FILE = parameters['assoc_spikes_file']
  HISTORY_FILE = parameters['history_file']
  HISTORY_PLOT_FILE = parameters['history_plot_file']
  MAZE_SIZE = parameters['maze_size']
  SCALES = parameters['scales']
  ROTATIONS = parameters['rotations']
  NUM_MODULES = parameters['num_modules']
  OFFSETS_PER_MODULE = parameters['offsets_per_module']
  GLOBAL_SCALE = parameters['global_scale']
  SHARPNESSES = parameters['sharpness']
  SIM_TIME = parameters['sim_time']
  EXC_SIZE = parameters['exc_size']
  OUT_SIZE = parameters['out_size']
  HYPERPARAMS = parameters['hyperparams']
  SPARSITIES = parameters['sparsities']
  RANGES = parameters['ranges']
  ALPHA = parameters['alpha']
  GAMMA = parameters['gamma']
  DECAY = parameters['decay']
  LR = parameters['lr']
  TRACE_LENGTH = parameters['trace_length']
  MAX_STEPS = parameters['max_steps']
  NUM_EPISODES = parameters['episodes']
  WARMUP_EPISODES = parameters['warmup_episodes']

  MAZE_FILE_PATH = os.path.join("./saves/mazes", MAZE_FILE)
  GRID_SPIKES_FILE_PATH = os.path.join("./saves/grid_cells", GRID_SPIKES_FILE)
  ASSOC_SPIKES_FILE_PATH = os.path.join("./saves/assoc_cells", ASSOC_SPIKES_FILE)
  HISTORY_FILE_PATH = os.path.join("./saves/history", HISTORY_FILE)


  #################### Phase 1: Initialization ####################
  # 1. Initialize Maze
  # 2. Initialize Grid Cell Spike Trains
  # 3. Initialize Assoc. Cell Spike Trains


  # Initialize Maze #
  if os.path.exists(MAZE_FILE_PATH):   # Load maze from file if exists
    print("Loading maze from file:", MAZE_FILE_PATH)
    with open(MAZE_FILE_PATH, 'rb') as f:
      maze_env = pkl.load(f)
  else:                                # Create new maze environment & save
    print("Creating new maze environment with size:", MAZE_SIZE)
    width = MAZE_SIZE[0]
    height = MAZE_SIZE[1]
    maze_env = Grid_Cell_Maze_Environment(width, height, trace_length=TRACE_LENGTH,)
    print("Saving maze environment to file:", MAZE_FILE_PATH)
    with open(MAZE_FILE_PATH, 'wb') as f:
      pkl.dump(maze_env, f)


  # Initialize Grid Cell Population & Spike Trains #
  if os.path.exists(GRID_SPIKES_FILE_PATH):  # Load grid cell spikes from file if exists
    print("Loading grid cell spikes from file:", GRID_SPIKES_FILE_PATH)
    with open(GRID_SPIKES_FILE_PATH, 'rb') as f:
      gc_population = pkl.load(f)
  else:
    print("Creating new grid cell population...")
    gc_population = GC_Population(NUM_MODULES, OFFSETS_PER_MODULE, GLOBAL_SCALE, SCALES, ROTATIONS, SHARPNESSES)
    gc_population.grid_cell_activity_generator(MAZE_SIZE)
    gc_population.spike_train_generator(sim_time=1000, max_firing_rates=gc_population.max_firing_rates)
    print("Saving grid cell spikes to file:", GRID_SPIKES_FILE_PATH)
    with open(GRID_SPIKES_FILE_PATH, 'wb') as f:
      pkl.dump(gc_population, f)


  # Initialize Assoc. Cell Population & Spike Trains #
  if os.path.exists(ASSOC_SPIKES_FILE_PATH):  # Load assoc. cell spikes from file if exists
    print("Loading association cell spikes from file:", ASSOC_SPIKES_FILE_PATH)
    with open(ASSOC_SPIKES_FILE_PATH, 'rb') as f:
      assoc_population = pkl.load(f)
  else:
    # Create SNN for association population
    print("Creating new association cell population...")
    n_cells = gc_population.n_cells
    w_in_exc = generate_grid_out_weights(n_cells, EXC_SIZE, SPARSITIES['in_exc'], RANGES['in_exc'])
    assoc_population = Association(
      in_size=n_cells,
      exc_size=EXC_SIZE,
      w_in_exc=w_in_exc,
      hyper_params=HYPERPARAMS, )

    # Generate per-position spike trains for association population
    assoc_spike_trains = torch.zeros(MAZE_SIZE[0], MAZE_SIZE[1], 1000, EXC_SIZE)
    assoc_voltages = torch.zeros(MAZE_SIZE[0], MAZE_SIZE[1], 1000, EXC_SIZE)
    gc_spike_trains = gc_population.maze_spike_trains
    for i in range(MAZE_SIZE[0]):
      for j in range(MAZE_SIZE[1]):
        exc_spikes, exc_voltage = assoc_population.get_spikes(gc_spike_trains[i, j], sim_time=1000)  # Run for 1 second
        assoc_spike_trains[i, j] = exc_spikes.squeeze(1)  # (time, exc)
        assoc_voltages[i, j] = exc_voltage.squeeze(1)  # (time, exc)

    # Save
    print("Saving association cell spikes to file:", ASSOC_SPIKES_FILE_PATH)
    assoc_population.set_maze_spike_trains(assoc_spike_trains)
    assoc_population.set_maze_voltages(assoc_voltages)
    with open(ASSOC_SPIKES_FILE_PATH, 'wb') as f:
      pkl.dump(assoc_population, f)
  #################### END PHASE 1 ####################

  if ONLY_PHASE_1:
    print("ONLY PHASE 1 flag is set; Exiting.")
    exit()

  #################### Phase 2: Training ####################
  # 1. Initialize SNN for Q-learning
  # 2. Train SNN using STDP and Q-learning
  # 3. Save trained SNN


  # Initialize SNN for Q-learning #
  print("Initializing SNN for Q-learning...")
  w_exc_out = generate_res_out_weights(EXC_SIZE, OUT_SIZE, SPARSITIES['exc_out'], RANGES['exc_out'])
  w_out_out = np.zeros((OUT_SIZE, OUT_SIZE))
  model = STDP_Q_Learning(
    in_size=EXC_SIZE,
    out_size=OUT_SIZE,
    w_exc_out=w_exc_out,
    w_out_out=w_out_out,
    exploration=1,
    alpha=ALPHA,
    gamma=GAMMA,
    num_actions=4,
    wmin=RANGES['exc_out'][0],
    wmax=RANGES['exc_out'][1],
    decay=DECAY,
    lr=LR,
    hyper_params=HYPERPARAMS,
  )
  maze_env.set_spike_trains(assoc_population.maze_spike_trains)

  if ANIMATE_TRAINING:
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(3, 3)
    maze_ax = fig.add_subplot(gs[0:3, -2:])
    weights_ax = fig.add_subplot(gs[0, 0])
    res_spikes_ax = fig.add_subplot(gs[1, 0])
    motor_spikes_ax = fig.add_subplot(gs[2, 0])

  def run_episode(animate=False, warmup=False, eps=1.0, decay=0.99):
    # state: spike trains of shape (exc, time)
    state, coords, _ = maze_env.reset()
    history = []
    for t in count():
      if animate and t > 0:
        maze_ax.clear()
        weights_ax.clear()
        res_spikes_ax.clear()
        motor_spikes_ax.clear()
        model.plot_weights(ax=weights_ax)
        model.plot_spikes(ax=res_spikes_ax, spikes=state, title="Association Spikes")
        model.plot_spikes(ax=motor_spikes_ax, spikes=out_spikes, title="Motor Spikes")
        maze_env.plot(coords, q_table=model.q_table, ax=maze_ax)
        plt.tight_layout()
        plt.pause(0.00001)

      ## Action Choice Logic ##
      # Warmup -> Epsilon-greedy exploration (only uses Q-Table, not SNN)
      # Non-warmup -> SNN-based action selection
      explore = eps > np.random.rand()
      if warmup:
        if explore:
          action = np.random.randint(model.num_actions)
        else:
          candidates = np.argwhere(model.q_table[coords] == model.q_table[coords].max())
          action = np.random.choice(candidates.flatten())   # Randomly select among max-value actions
      else:
        action, out_spikes = model.select_action(state, SIM_TIME, explore)

      ## Environment Step ##
      new_state, reward, terminated, new_coords = maze_env.step(action)
      delta_Q = model.Q_Learning(coords, action, reward, new_coords)

      ## Learning ##
      if not warmup:
        model.STDP_RL(np.sign(delta_Q), state, out_spikes, action)
        model.reset_state_variables()

      eps *= decay  # Decay epsilon for exploration

      ## Recording ##
      history.append((coords, action, reward, new_coords, delta_Q))
      if VERBOSE_TRAINING:
        print(f"Step {t+1}/{MAX_STEPS} - Reward: {reward:.2f} - Delta-Q {np.sign(delta_Q)} - exploring?: {explore} - Epsilon: {eps:.3f}")
      state = new_state
      coords = new_coords

      if terminated or t >= MAX_STEPS:
        if not warmup:
          print(f"Episode finished after {t} timesteps")
          print("Average Delta-Q:", np.mean([h[4] for h in history]))
        break

    return history, eps

  # Train SNN using STDP and Q-learning #
  universal_history = []
  ep_lengths = []

  # Q-Table warmup phase
  print("PHASE 1...")
  eps = 1.0
  for episode in range(WARMUP_EPISODES):
    history, eps = run_episode(False, warmup=True, eps=eps, decay=1)
    ep_lengths.append(len(history))
    universal_history.append(history)
  print("PHASE 1 completed")

  print("PHASE 2")
  epsilon = 1
  for episode in range(NUM_EPISODES):
    history, epsilon = run_episode(ANIMATE_TRAINING, eps=epsilon)
    ep_lengths.append(len(history))
    print(f"Episode {episode + 1}/{NUM_EPISODES} - Steps: {len(history)}")
    universal_history.append(history)
  print("PHASE 2 completed")

  maze_env.maze.start_cell = maze_env.maze.get_cell(6, 0)  # Different corner
  print("PHASE 3")
  eps = 1.0
  for episode in range(WARMUP_EPISODES):
    history, eps = run_episode(False, warmup=True, eps=eps, decay=1)
    ep_lengths.append(len(history))
    universal_history.append(history)
  print("PHASE 3 completed")

  print("PHASE 4")
  epsilon = 1
  for episode in range(NUM_EPISODES):
    history, epsilon = run_episode(ANIMATE_TRAINING, eps=epsilon)
    ep_lengths.append(len(history))
    print(f"Episode {episode + 1}/{NUM_EPISODES} - Steps: {len(history)}")
    universal_history.append(history)
  print("PHASE 4 completed")

  print("PHASE 5")
  maze_env.maze.start_cell = maze_env.maze.get_cell(0, 0)  # Different corner
  epsilon = 1
  for episode in range(NUM_EPISODES):
    history, epsilon = run_episode(ANIMATE_TRAINING, eps=1)
    ep_lengths.append(len(history))
    print(f"Episode {episode + 1}/{NUM_EPISODES} - Steps: {len(history)}")
    universal_history.append(history)
  print("PHASE 5 completed")

  print("Saving training history to file...")
  with open(HISTORY_FILE_PATH, "wb") as f:
    pkl.dump(universal_history, f)


if __name__ == '__main__':
  GC_scales = np.array([3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, ])
  np.random.seed(123)
  p = {
    'plot': False,
    'animate_training': False,
    'verbose_training': False,
    'only_phase_1': False,
    'maze_file': "7_7_maze.pkl",
    'grid_spikes_file': "7_7_grid_spikes.pkl",
    'assoc_spikes_file': "7_7_assoc_spikes.pkl",
    'spike_analysis_file': "7_7_spike_analysis.pkl",
    'history_file': "7_7_history.pkl",
    'history_plot_file': "7_7_history_plot.png",
    'maze_size': (7, 7),
    'num_modules': 5,
    'offsets_per_module': 4,
    'scales': GC_scales,
    'global_scale': 1,
    'rotations': [0, np.pi / 3, np.pi / 5, np.pi*3 / 5, np.pi / 7, np.pi*3 / 7, np.pi / 11],
    'sharpness': 1.5,  # Should *not* go below 1
    'sim_time': 1000,  # ms
    'exc_size': 3000,
    'inh_size': 1,
    'out_size': 100 * 4,
    'hyperparams': {
      "exc_refrac": 1,
      "exc_reset": -70,  # Base
      "exc_tc_decay": 20,  # AND decay for 20ms interval @ 15mv threshold
      "exc_tc_theta_decay": 10_000,
      "exc_theta_plus": 0,
      "exc_thresh": -45,  #
      "inh_refrac": 1,
      "inh_reset": -64,
      "inh_tc_decay": 10_000,
      "inh_tc_theta_decay": 10_000,
      "inh_theta_plus": 0,
      "inh_thresh": -60,
      "refrac_out": 0,
      "reset_out": -64,  # Base
      "tc_decay_out": 20,  # AND decay for 20ms interval @ 11mv threshold
      "tc_theta_decay_out": 10_000,
      "theta_plus_out": 0,
      "thresh_out": -49,
    },
    'ranges': {
      'in_exc': (0, 6.5),
      'in_inh': (0, 1),
      'exc_exc': (0, 1),
      'exc_inh': (0, 1),
      'inh_exc': (-1, 0),
      'inh_inh': (-1, 0),
      'exc_out': (0, 1),
      'out_out': (-1, -1),

    },
    'sparsities': {
      'in_exc': 0.12,
      'in_inh': 0.0,
      'exc_exc': 0.0,
      'exc_inh': 0.0,
      'inh_exc': 0.0,
      'inh_inh': 0.0,
      'exc_out': 0.2,
      'out_out': 0.0,
    },
    'alpha': 0.01,  # Q-Table learning rate
    'gamma': 0.99,  # Q-Table discount factor (how much future rewards are discounted)
    'decay': 0.5,  # Synaptic decay (UNUSED)
    'lr': 0.01,  # Weight update learning rate
    'trace_length': 0,
    'max_steps': 1000,
    'warmup_episodes': 25,
    'episodes': 5,
  }
  run(p)