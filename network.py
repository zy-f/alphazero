import torch
from tqdm import tqdm
import storage
from alphaNet import *

class Network(object):
    def __init__(self, net_config):
        self.epochs = net_config.n_epochs
        self.net = net_config.net_cls(**net_config.net_params)
        if net_config.weights_filepath is not None:
            self.net.load_weights(net_config.weights_filepath)
        self.loss_func = net_config.loss_cls()
        try:
            self.loss_func = net_config.loss_cls().eval()
        except:
            pass
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=net_config.optim_lr)
    
    # runs 1 epoch of training
    def train_epochs(self, storage):
        if len(storage.games) < bsz*n_batches:
            return
        self.train()
        for _ in tqdm(self.epochs):
            sum_loss = 0.0
            iters = 0
            for batch in tqdm(storage.get_batches()):
                inp, pi, z = batch['inputs'], batch['pis'], batch['zs']
                p, v = self.net(inp)
                loss_size = self.loss_func(pi, p, z, v)
                optimizer.zero_grad()
                loss_size.backward()
                optimizer.step()
                
                sum_loss += loss_size.item()
                iters += 1
            print(f"LOSS={sum_loss/iters}")
    
    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def predict(self, inp):
        inp = torch.from_numpy(inp)
        p, v = self.net(inp)
        policy = {(k//3,k%3):p[k] for k in range(len(p))}
        return policy, v