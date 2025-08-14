import numpy as np
import torch

from bindsnet.network import Network
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input, AdaptiveLIFNodes
from bindsnet.network.topology import MulticompartmentConnection
from bindsnet.network.topology_features import Weight


class Association(Network):
  def __init__(self,
               in_size: int,   # Number of input neurons
               exc_size: int,  # Number of excitatory neurons
               w_in_exc: np.ndarray,  # Input to excitatory weights
               hyper_params: dict,  # Dictionary of hyperparameters
               device: str = 'cpu'):
    super().__init__()

    ## Maze specific parameters ##
    self.maze_spike_trains = None
    self.maze_voltages = None

    ## Layers ##
    input = Input(n=in_size)
    assoc_exc = AdaptiveLIFNodes(
      n=exc_size,
      thresh=hyper_params['exc_thresh'],
      theta_plus=hyper_params['exc_theta_plus'],
      refrac=hyper_params['exc_refrac'],
      reset=hyper_params['exc_reset'],
      tc_theta_decay=hyper_params['exc_tc_theta_decay'],
      tc_decay=hyper_params['exc_tc_decay'],
      traces=True,
    )
    self.add_layer(input, name='input')
    self.add_layer(assoc_exc, name='assoc_exc')

    ## Layer monitor ##
    exc_monitor = Monitor(assoc_exc, ["s", "v"], device=device)
    self.add_monitor(exc_monitor, name='assoc_monitor_exc')
    self.exc_monitor = exc_monitor

    ## Connections ##
    in_exc_wfeat = Weight(name='in_exc_weight_feature', value=torch.Tensor(w_in_exc),)
    in_exc_conn = MulticompartmentConnection(
      source=input, target=assoc_exc,
      device=device, pipeline=[in_exc_wfeat],
    )

    self.add_connection(in_exc_conn, source='input', target='assoc_exc')

    ## Migrate ##
    self.to(device)

  def set_maze_spike_trains(self, spike_trains):
    self.maze_spike_trains = spike_trains

  def set_maze_voltages(self, voltages):
    self.maze_voltages = voltages

  # Expect input_train to be an ndarray of shape (#-Grid-Cells, time)
  def get_spikes(self, input_train: np.ndarray, sim_time):
    input_train = torch.Tensor(input_train.T).unsqueeze(1)    # Reshape to (time, 1, #-Grid-Cells)
    self.run(inputs={'input': input_train}, time=sim_time,)
    exc_spikes = self.exc_monitor.get('s')
    exc_voltage = self.exc_monitor.get('v')
    return exc_spikes, exc_voltage
