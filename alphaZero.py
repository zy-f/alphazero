import mcts
from storage import Storage
import network
import alphaNet
import board
from utils import *

def alphazero():
    game_board = board.T3Board

    net_params = {
        'n_filters': 32, 
        'n_hidden':32, 
        'n_res':10
    }

    net_kwargs = {
            'net_cls': alphaNet.AlphaNet, 
            'net_params': net_params, 
            'loss_cls': alphaNet.AlphaLoss, 
            'optim_lr': 2e-2,
            'weights_filepath': None,
            'n_epochs': 100,
            'bsz': 32,
            'n_batches': 8
    }

    mcts_kwargs = {
        'n_games': 25,
        'n_sims_per_game': 100,
        'c_puct_decay_n': 10
    }
    net_config = Config(**net_kwargs)
    mcts_config = Config(**mcts_kwargs)
    storage = Storage(net_config, mcts_config, game_board, buffer_len=int(1e5))
    net = storage.latest_network()

    while True:
        loops = 0
        try:
            mcts.self_play(storage)
            net.train_epochs(storage)
            if loops % 100 == 0:
                self_play_game()
            loops += 1
        except KeyboardInterrupt:
            return storage.latest_network()

def self_play_game(network):
    board = board.T3Board()
    while board.end_state() == 0:
        action = mcts.play_learned_action(network, board, n_sim=100)
        board.play(action)
        print(board)


if __name__ == '__main__':
    alphazero()