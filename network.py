import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
import storage
from alphaNet import *
import numpy as np

class Network(object):
    def __init__(self, net_config):
        self.epochs = net_config.n_epochs
        self.net = net_config.net_cls(**net_config.net_params)
        if net_config.weights_filepath is not None:
            self.net.load_weights(net_config.weights_filepath)
        self.loss_func = net_config.loss_cls()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=net_config.optim_lr, weight_decay=net_config.weight_decay)
        try:
            self.loss_func = net_config.loss_cls().eval()
        except:
            pass
    
    # runs 1 epoch of training
    def train_epochs(self, storage):
        if len(storage.dataset) < storage.net_config.bsz:
            return
        self.train()
        print("Training Neural Net:")
        for epoch in range(self.epochs):
            sum_loss = 0.0
            iters = 0
            for batch in storage.get_batches(): #desc=f'Training Epoch {epoch+1}/{self.epochs}'
                inp, pi, z = batch['inputs'], batch['pis'], batch['zs']
                p, v = self.net(inp)
                loss_size = self.loss_func(pi, p, z, v)
                self.optimizer.zero_grad()
                loss_size.backward()
                self.optimizer.step()
                
                sum_loss += loss_size.item()
                iters += 1
            if (epoch+1) % (max(1,self.epochs//10)) == 0:
                print(f"EPOCH={epoch+1}/{self.epochs}, LOSS={sum_loss/iters}")

    def predict(self, inp):
        self.eval()
        inp = torch.from_numpy(inp)
        p, v = self.net(inp)
        v = v.detach().numpy().squeeze()
        p = p.detach().exp().numpy().squeeze()
        # print("PRED_V="+str(v))
        return p, v
    
    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()
    
    def __str__(self):
        return str(self.net)