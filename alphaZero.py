import mcts
from storage import Storage
import network
import alphaNet
import board
from utils import *

class AlphaZero:
    def __init__(self):
        self.game_board = board.T3Board
        
        net_params = {
            'board_layers': self.game_board.encoding_layers,
            'board_size': self.game_board.board_size,
            'action_space': self.game_board.action_space,
            'n_filters': 512, 
            'n_hidden': 512, 
            'n_res': 2
        }

        net_kwargs = {
                'net_cls': alphaNet.AlphaNet, 
                'net_params': net_params, 
                'loss_cls': alphaNet.AlphaLoss, 
                'optim_lr': 1e-3,
                'weight_decay': .1,
                'weights_filepath': None,
                'n_epochs': 5,
                'bsz': 64
        }

        mcts_kwargs = {
            'n_games': 25,
            'n_sims_per_game_step': 25,
            'c_puct_decay_n': 10
        }

        net_config = Config(**net_kwargs)
        mcts_config = Config(**mcts_kwargs)
        self.storage = Storage(net_config, mcts_config, self.game_board, buffer_len=int(1e5))

    def alpha_zero(self):
        net = self.storage.latest_network()
        loops = 0
        while True:
            try:
                print(f'\x1b[1;30;41mALPHAZERO LOOP {loops+1}\x1b[0m')
                mcts.self_play(self.storage)
                net.train_epochs(self.storage)
                if loops % 10 == 0:
                    for k in range(5):
                        print(f'Game {k+1}')
                        self.self_play_game(self.storage.latest_network())
                loops += 1
            except KeyboardInterrupt:
                return self.storage.latest_network()

    def self_play_game(self, network):
        game_board = self.game_board()
        while game_board.end_state() == 0:
            action = mcts.play_learned_action(network, game_board, n_sim=100)
            game_board.play(action)
            print(game_board)


if __name__ == '__main__':
    az = AlphaZero()
    az.alpha_zero()