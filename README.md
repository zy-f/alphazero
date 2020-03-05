# alphazero
Personal implementation of DeepMind's AlphaZero in Pytorch

- developed for tictactoe
- development for connect 4 planned

To train and play:
1) Install library requirements (via `pip3 install -r requirements.txt`)
2) Run alphaZero.py (ensure correct hyperparameters, see below)
3) Change opponent network to saved network name in playGames.py
4) Run playGames.py

If you only want to play, skip steps 2 and 3.

Tic-tac-toe AlphaZero parameters:
```
self.game_board = board.T3Board
storage_buffer_len = int(1e5)

net_params = {
    'board_layers': self.game_board.encoding_layers,
    'board_size': self.game_board.board_size,
    'action_space': self.game_board.action_space,
    'n_filters': 128,
    'n_hidden': 128,
    'n_res': 2
}

net_kwargs = {
    'net_cls': alphaNet.AlphaNet, 
    'net_params': net_params, 
    'loss_cls': alphaNet.AlphaLoss, 
    'optim_lr': 1e-3,
    'weight_decay': 0,
    'weights_filepath': pretrained_path,
    'n_epochs': 10,
    'bsz': 64,
    'save_cutoff': .5
}

mcts_kwargs = {
    'n_games': 5,
    'n_sims_per_game_step': 25,
    'temp_threshold': 9,
    'dirichlet_noise_alpha': .15,
    'verbose': True
}
```

Connect 4 AlphaZero Parameters (WIP)
```
self.game_board = board.C4Board
storage_buffer_len = int(1e4)

net_params = {
    'board_layers': self.game_board.encoding_layers,
    'board_size': self.game_board.board_size,
    'action_space': self.game_board.action_space,
    'n_filters': 64, #64
    'n_hidden': 64, #64
    'n_res': 5
}

net_kwargs = {
    'net_cls': alphaNet.AlphaNet, 
    'net_params': net_params, 
    'loss_cls': alphaNet.AlphaLoss, 
    'optim_lr': 1e-3,
    'weight_decay': 0,
    'weights_filepath': pretrained_path,
    'n_epochs': 10,
    'bsz': 64,
    'save_cutoff': .501
}

mcts_kwargs = {
    'n_games': 5,
    'n_sims_per_game_step': 30,
    'temp_threshold': 25,
    'dirichlet_noise_alpha': .15,
    'verbose': True
}
```