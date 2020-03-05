import mcts
from storage import Storage
import network
import alphaNet
import board
from utils import *
import torch
from tqdm import tqdm
import os

class AlphaZero:
    def __init__(self, pretrained_path=None):

        self.game_board = board.C4Board
        storage_buffer_len = int(1e5)
        
        net_params = {
            'board_layers': self.game_board.encoding_layers,
            'board_size': self.game_board.board_size,
            'action_space': self.game_board.action_space,
            'n_filters': 128, #64
            'n_hidden': 128, #64
            'n_res': 3
        }

        net_kwargs = {
                'net_cls': alphaNet.AlphaNet, 
                'net_params': net_params, 
                'loss_cls': alphaNet.AlphaLoss, 
                'optim_lr': 1e-3,
                'weight_decay': 0,
                'weights_filepath': pretrained_path,
                'n_epochs': 10,
                'bsz': 32,
                'save_cutoff': .501
        }

        mcts_kwargs = {
            'n_games': 5,
            'n_sims_per_game_step': 30,
            'temp_threshold': 20,
            'dirichlet_noise_alpha': .08,
            'verbose': True
        }

        net_config = Config(**net_kwargs)
        mcts_config = Config(**mcts_kwargs)
        self.storage = Storage(net_config, mcts_config, self.game_board, buffer_len=storage_buffer_len)

        self.n_display_games = 2
        self.n_compare_games = 10

    def run_alpha_zero(self):
        # cleanup old files
        for dirfile in os.listdir('net_files'):
            if "temp" in dirfile:
                os.remove(f'net_files/{dirfile}')
        
        net = self.storage.latest_network()
        self.storage.save_network(net)
        print(net)
        loops = 0
        while True:
            net = self.storage.latest_network()
            try:
                print(f'\x1b[1;30;41m ALPHAZERO LOOP {loops+1} (CTRL+C TO STOP) \x1b[0m')
                mcts.self_play(self.storage)
                net.train_epochs(self.storage)
                if self.compare_networks([self.storage.latest_network(), net]):
                    self.storage.save_network(net)
                if (loops+1) % 5 == 0:
                    for k in range(self.n_display_games):
                        print(f'Game {k+1}')
                        self.self_play_game([self.storage.latest_network(),self.storage.latest_network()], print_game=True)
                loops += 1
            except KeyboardInterrupt:
                return self.storage.latest_network()

    # networks = [old_net, new_net]
    def compare_networks(self, networks):
        new_wins = []
        for k in tqdm(range(self.n_compare_games), desc='Comparing Networks'):
            turn_winner = self.self_play_game(networks[::(-1)**k], print_game=(k%2==0 and k<4))
            # 1 point to new net for winning, .5 points for draw, 0 points for losing
            new_wins.append( [.5, 0, 1][ turn_winner*((-1)**k) ] )
        print(f"Results: {new_wins}\nNew winrate: {sum(new_wins)/self.n_compare_games}") 
        return sum(new_wins)/self.n_compare_games >= self.storage.net_config.save_cutoff

    def self_play_game(self, networks, print_game=False):
        game_board = self.game_board()
        turn = 0
        while game_board.end_state() is None:
            action = mcts.get_learned_action(networks[turn%2], game_board, n_sim=self.storage.mcts_config.n_sims_per_game_step, noise=self.storage.mcts_config.dirichlet_noise_alpha, print_state=(print_game and self.storage.mcts_config.verbose))
            game_board.play(action)
            if print_game:
                print(game_board)
        return game_board.end_state()
    
    def play_vs_human(self, board, print_thinking=False):
        if not hasattr(self, 'final_net'):
            self.final_network = self.storage.latest_network()
        return mcts.get_learned_action(self.final_network, board, n_sim=self.storage.mcts_config.n_sims_per_game_step, noise=self.storage.mcts_config.dirichlet_noise_alpha, print_state=print_thinking)

if __name__ == '__main__':
    az = AlphaZero()
    final_net = az.run_alpha_zero()
    save_file = input('Save as? (ctrl-c to quit without saving) ')
    torch.save(final_net.net.state_dict(), f'net_files/final/{save_file}.pth')