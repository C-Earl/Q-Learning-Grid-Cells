import pickle as pkl
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

if __name__ == '__main__':
  with open('saves/3000_res_7_7_maze.pkl', 'rb') as f:
    association_data = pkl.load(f)

  with open('saves/grid_spikes.pkl', 'rb') as f:
    grid_spikes = pkl.load(f)

  with open('saves/voltage_spikes.pkl', 'rb') as f:
    association_voltages = pkl.load(f)

  with open('saves/synapses.pkl', 'rb') as f:
    synapses = pkl.load(f)

  target_asc_data = association_data[4, 2].numpy()
  target_gc_data = grid_spikes[4, 2].copy()

  # Identitfy active grid cells
  plt.figure(figsize=(7, 3.5))
  gc_activity = target_gc_data.sum(1)
  important_gc = np.argsort(gc_activity)[-10:]
  semi_important_gc = np.random.choice(np.arange(len(gc_activity)), replace=False, size=10)
  plt.bar(np.arange(len(gc_activity)-1), gc_activity[1:])
  plt.title("Grid Cell Activity at (4, 2)")
  plt.xlabel("Grid Cell Index")
  plt.ylabel("Spike Count")
  plt.tight_layout()
  plt.savefig("grid_cell_activity", dpi=200)
  print("Most active grid cells:", important_gc)

  # Identify active place cells
  plt.clf()
  plt.figure(figsize=(7, 3.5))
  pc_activity = target_asc_data.sum(0)
  important_pc = np.argsort(pc_activity)[-10:]
  semi_important_pc = np.random.choice(np.arange(len(pc_activity)), replace=False, size=10)
  plt.bar(np.arange(len(pc_activity)-1), pc_activity[1:])
  plt.title("Association Cell Activity at (4, 2)")
  plt.xlabel("Place Cell Index")
  plt.ylabel("Spike Count")
  plt.tight_layout()
  plt.savefig("place_cell_activity", dpi=200)
  print("Most active place cells:", np.argsort(pc_activity)[-10:])

  ## Associate firing behaviors ##
  # Important GC activity
  plt.figure(figsize=(7, 3.5))
  important_gc_activity = target_gc_data[important_gc, :]
  random_gc_activity = target_gc_data[semi_important_gc, :]
  plt.clf()
  plt.imshow(important_gc_activity, interpolation='nearest', aspect='auto')
  plt.title("High-Activity Grid Cell Spikes at (4, 2)")
  plt.xlabel("Time Step")
  plt.ylabel("Grid Cell Index")
  plt.yticks(ticks=np.arange(len(important_gc)), labels=important_gc)
  plt.xticks(ticks=np.arange(0, 1000, 100))
  plt.grid(alpha=0.5, linestyle='--', linewidth=0.5, color='gray')
  plt.savefig("important_grid_spike_plot", dpi=300, bbox_inches='tight')

  # Random GC activity
  plt.clf()
  plt.figure(figsize=(7, 3.5))
  plt.imshow(random_gc_activity, interpolation='nearest', aspect='auto')
  plt.title("Random Grid Cell Spikes at (4, 2)")
  plt.xlabel("Time Step")
  plt.ylabel("Grid Cell Index")
  plt.yticks(ticks=np.arange(len(semi_important_gc)), labels=semi_important_gc)
  plt.xticks(ticks=np.arange(0, 1000, 200))
  plt.savefig("random_grid_spike_plot", dpi=300, bbox_inches='tight')

  # Important association cell activity
  plt.clf()
  plt.figure(figsize=(7, 3.5))
  important_asc_activity = target_asc_data.T[important_pc, :]
  random_asc_activity = target_asc_data.T[semi_important_pc, :]
  plt.imshow(important_asc_activity, interpolation='nearest', aspect='auto')
  plt.title("High-Activity Association Cell Spikes at (4, 2)")
  plt.xlabel("Time Step")
  plt.ylabel("Association Cell Index")
  plt.yticks(ticks=np.arange(len(important_pc)), labels=important_pc)
  plt.savefig("important_place_spike_plot", dpi=300, bbox_inches='tight')

  # Random association cell activity
  plt.clf()
  plt.figure(figsize=(7, 3.5))
  plt.imshow(random_asc_activity, interpolation='nearest', aspect='auto')
  plt.title("Random 'Place Cell' Spikes at (4, 2)")
  plt.xlabel("Time Step")
  plt.ylabel("Association Cell Index")
  plt.yticks(ticks=np.arange(len(semi_important_pc)), labels=semi_important_pc)
  plt.savefig("random_place_spike_plot", dpi=300, bbox_inches='tight')

  # GC raster plot at (4, 2)
  plt.clf()
  plt.figure(figsize=(10, 5))
  plt.imshow(target_gc_data, interpolation='nearest', aspect='auto')
  plt.title("Grid Cell Spikes at (4, 2)")
  plt.xlabel("Time Step")
  plt.ylabel("Grid Cell Index")
  plt.savefig("grid_raster_plot", dpi=300, bbox_inches='tight')

  # High activity GC and AC
  plt.figure(figsize=(5, 5))
  important_synapses = synapses[np.ix_(important_gc, important_pc)]
  plt.clf()
  plt.imshow(important_synapses.T, interpolation='nearest', aspect='auto', cmap='plasma',vmin=0, vmax=1)
  plt.title("High Activity GC to High Activity AC Synapses")
  plt.xlabel("Grid Cell Index")
  plt.ylabel("Association Cell Index")
  plt.xticks(ticks=np.arange(len(important_gc)), labels=important_gc)
  plt.yticks(ticks=np.arange(len(important_pc)), labels=important_pc)
  plt.savefig('synaptic_connections', dpi=300, bbox_inches='tight')

  # High activity GC to random AC
  plt.figure(figsize=(5, 5))
  random_synapses = synapses[np.ix_(important_gc, semi_important_pc)]
  plt.clf()
  plt.imshow(random_synapses.T, interpolation='nearest', aspect='auto', cmap='plasma', vmin=0, vmax=1)
  plt.title("High Activity GC to Random AC Synapses")
  plt.xlabel("Grid Cell Index")
  plt.ylabel("Association Cell Index")
  plt.xticks(ticks=np.arange(len(important_gc)), labels=important_gc)
  plt.yticks(ticks=np.arange(len(semi_important_pc)), labels=semi_important_pc)
  plt.savefig('random_synaptic_connections', dpi=300, bbox_inches='tight')

  # Voltage of association cell #605
  from matplotlib.gridspec import GridSpec
  plt.figure(figsize=(6, 6))
  gs = GridSpec(2, 1)
  voltage_ax = plt.subplot(gs[0, 0])
  spike_ax = plt.subplot(gs[1, 0])
  mod_voltages = association_voltages[4, 2].T[605, 100:200]
  mod_voltages[86] = mod_voltages[85]+6.5 # Show spike for visualization
  voltage_ax.plot(np.arange(100, 200), association_voltages[4, 2].T[605, 100:200], label='Association Cell #605 Voltage', color='blue')
  voltage_ax.plot(np.arange(100, 200), -49 * np.ones(100), '--', label='Firing Threshold', color='red')
  voltage_ax.grid(alpha=0.5, linestyle='--', linewidth=0.5, color='gray')
  voltage_ax.set_ylim(-75, -45)
  voltage_ax.set_xlim(100, 200)
  voltage_ax.set_xticks(ticks=np.arange(100, 200, 10))
  voltage_ax.set_xlabel("Time Step")
  voltage_ax.set_ylabel("Voltage (mV)")
  voltage_ax.set_title("Voltage of Association Cell #605 at (4, 2)")

  important_gc_highlights = grid_spikes[4, 2][:, 100:200].copy()
  important_gc_highlights[important_gc] *= 100
  spike_ax.imshow(important_gc_highlights, vmin=0, vmax=100, aspect='auto', interpolation='nearest')
  spike_ax.grid(alpha=0.5, linestyle='--', linewidth=0.5, color='gray')
  spike_ax.set_xticks(ticks=np.arange(0, 100, 10), labels=np.arange(100, 200, 10))
  spike_ax.set_title("Grid Cell Spikes at (4, 2)")
  spike_ax.set_xlabel("Time Step")
  spike_ax.set_ylabel("Grid Cell Index")
  plt.tight_layout()
  plt.savefig("association_cell_voltage_1", dpi=300, bbox_inches='tight')

  plt.figure(figsize=(6, 6))
  gs = GridSpec(2, 1)
  voltage_ax = plt.subplot(gs[0, 0])
  spike_ax = plt.subplot(gs[1, 0])
  mod_voltages = association_voltages[4, 2].T[605, 300:400]
  mod_voltages[70] = mod_voltages[69]+6.5 # Show spike for visualization
  voltage_ax.plot(np.arange(300, 400), association_voltages[4, 2].T[605, 300:400], label='Association Cell #605 Voltage', color='blue')
  voltage_ax.plot(np.arange(300, 400), -49 * np.ones(100), '--', label='Firing Threshold', color='red')
  voltage_ax.grid(alpha=0.5, linestyle='--', linewidth=0.5, color='gray')
  voltage_ax.set_ylim(-75, -45)
  voltage_ax.set_xlim(300, 400)
  voltage_ax.set_xticks(ticks=np.arange(300, 400, 10))
  voltage_ax.set_xlabel("Time Step")
  voltage_ax.set_ylabel("Voltage (mV)")
  voltage_ax.set_title("Voltage of Association Cell #605 at (4, 2)")

  important_gc_highlights = grid_spikes[4, 2][:, 300:400].copy()
  important_gc_highlights[important_gc] *= 100
  spike_ax.imshow(important_gc_highlights, vmin=0, vmax=100, aspect='auto', interpolation='nearest')
  spike_ax.grid(alpha=0.5, linestyle='--', linewidth=0.5, color='gray')
  spike_ax.set_xticks(ticks=np.arange(0, 100, 10), labels=np.arange(300, 400, 10))
  spike_ax.set_title("Grid Cell Spikes at (4, 2)")
  spike_ax.set_xlabel("Time Step")
  spike_ax.set_ylabel("Grid Cell Index")
  plt.tight_layout()
  plt.savefig("association_cell_voltage_2", dpi=300, bbox_inches='tight')

  plt.figure(figsize=(6, 6))
  gs = GridSpec(2, 1)
  voltage_ax = plt.subplot(gs[0, 0])
  spike_ax = plt.subplot(gs[1, 0])
  mod_voltages = association_voltages[4, 2].T[605, 500:600]
  mod_voltages[52] = mod_voltages[51]+13 # Show spike for visualization
  voltage_ax.plot(np.arange(500, 600), association_voltages[4, 2].T[605, 500:600], label='Association Cell #605 Voltage', color='blue')
  voltage_ax.plot(np.arange(500, 600), -49 * np.ones(100), '--', label='Firing Threshold', color='red')
  voltage_ax.grid(alpha=0.5, linestyle='--', linewidth=0.5, color='gray')
  voltage_ax.set_ylim(-75, -45)
  voltage_ax.set_xlim(500, 600)
  voltage_ax.set_xticks(ticks=np.arange(500, 600, 10))
  voltage_ax.set_xlabel("Time Step")
  voltage_ax.set_ylabel("Voltage (mV)")
  voltage_ax.set_title("Voltage of Association Cell #605 at (4, 2)")

  important_gc_highlights = grid_spikes[4, 2][:, 500:600].copy()
  important_gc_highlights[important_gc] *= 100
  spike_ax.imshow(important_gc_highlights, vmin=0, vmax=100, aspect='auto', interpolation='nearest')
  spike_ax.grid(alpha=0.5, linestyle='--', linewidth=0.5, color='gray')
  spike_ax.set_xticks(ticks=np.arange(0, 100, 10), labels=np.arange(500, 600, 10))
  spike_ax.set_title("Grid Cell Spikes at (4, 2)")
  spike_ax.set_xlabel("Time Step")
  spike_ax.set_ylabel("Grid Cell Index")
  plt.tight_layout()
  plt.savefig("association_cell_voltage_3", dpi=300, bbox_inches='tight')


  plt.figure(figsize=(6, 6))
  gs = GridSpec(2, 1)
  voltage_ax = plt.subplot(gs[0, 0])
  spike_ax = plt.subplot(gs[1, 0])
  mod_voltages = association_voltages[4, 2].T[605, 700:800]
  mod_voltages[41] = mod_voltages[40]+6.5 # Show spike for visualization
  voltage_ax.plot(np.arange(700, 800), association_voltages[4, 2].T[605, 700:800], label='Association Cell #605 Voltage', color='blue')
  voltage_ax.plot(np.arange(700, 800), -49 * np.ones(100), '--', label='Firing Threshold', color='red')
  voltage_ax.grid(alpha=0.5, linestyle='--', linewidth=0.5, color='gray')
  voltage_ax.set_ylim(-75, -45)
  voltage_ax.set_xlim(700, 800)
  voltage_ax.set_xticks(ticks=np.arange(700, 800, 10))
  voltage_ax.set_xlabel("Time Step")
  voltage_ax.set_ylabel("Voltage (mV)")
  voltage_ax.set_title("Voltage of Association Cell #605 at (4, 2)")

  important_gc_highlights = grid_spikes[4, 2][:, 700:800].copy()
  important_gc_highlights[important_gc] *= 100
  spike_ax.imshow(important_gc_highlights, vmin=0, vmax=100, aspect='auto', interpolation='nearest')
  spike_ax.grid(alpha=0.5, linestyle='--', linewidth=0.5, color='gray')
  spike_ax.set_xticks(ticks=np.arange(0, 100, 10), labels=np.arange(700, 800, 10))
  spike_ax.set_title("Grid Cell Spikes at (4, 2)")
  spike_ax.set_xlabel("Time Step")
  spike_ax.set_ylabel("Grid Cell Index")
  plt.tight_layout()
  plt.savefig("association_cell_voltage_4", dpi=300, bbox_inches='tight')
