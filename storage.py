import mcts
import network
import torch
from utils import *

class Storage(object):
    def __init__(self, net_config, mcts_config, game_class, buffer_len=int(1e5)): 
        self.net_config = net_config
        self.mcts_config = mcts_config
        self.board_cls = game_class
        self.n_networks = 0
        self.networks = []
        self.games = []
        self.game_buffer_len = buffer_len
    
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
    
    def save_games(self, dataset):
        removable_game_idxs = max(0, len(self.games)+len(dataset)-self.game_buffer_len)
        self.games = self.games[removable_game_idxs:] + dataset
    
    def get_batches(self):
        for _ in range(self.net_config.n_batches):
            data = np.random.choice(np.array(self.games), size=self.net_config.bsz)
            inp = torch.from_numpy(data[:,0])
            pi = torch.from_numpy(data[:,1])
            z = torch.from_numpy(data[:,2])
            yield {'inputs':inp, 'pis':pi, 'zs':z}
        return