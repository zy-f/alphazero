import mcts
import network
import torch
import numpy as np
from utils import *

class Storage(object):
    def __init__(self, net_config, mcts_config, game_class, buffer_len=int(1e5)): 
        self.net_config = net_config
        self.mcts_config = mcts_config
        self.board_cls = game_class
        self.n_networks = 0
        self.networks = []
        self.dataset = []
        self.dataset_buffer_len = buffer_len
    
    def latest_network(self):
        net = network.Network(self.net_config)
        if len(self.networks) > 0:
            self.net_config.weights_filepath = self.networks[-1]
            net.load_state_dict(torch.load(self.net_config.weights_filepath))
        return net
    
    def save_network(self, network):
        filename = f"net_files/temp_{self.n_networks%5}.pth"
        torch.save(network.net.state_dict(), filename)
        self.n_networks += 1
        if len(self.networks) < 5:
            self.networks.append(filename)
        else:
            self.networks = networks[1:]+[filename]
    
    def save_sub_dataset(self, dataset):
        removable_idxs = max(0, len(self.dataset)+len(dataset)-self.dataset_buffer_len)
        self.dataset = self.dataset[removable_idxs:] + dataset
    
    def get_batches(self):
        data_idxs = np.array(range(len(self.dataset)))
        np.random.shuffle(data_idxs)

        boards, policies, values = map(np.array, list(zip(*self.dataset)))
        bsz = self.net_config.bsz
        for k in range( (len(self.dataset)//bsz)-1 ):
            inp, pi, z = ( np.take(x, data_idxs[k*bsz:(k+1)*bsz], axis=0) for x in (boards, policies, values) )
            inp, pi, z = map(torch.from_numpy, (inp,pi,z))
            yield {'inputs':inp, 'pis':pi, 'zs':z}
        return