from alphaZero import AlphaZero
from board import T3Board
import argparse

def play_t3(opponent_network):
    print("===Tic-tac-toe VS AlphaZero===")
    az = AlphaZero(pretrained_path=f'net_files/final/{opponent_network}.pth')
    replay = True

    while replay:
        player_turn = 0
        while player_turn not in [1,2]:
            try:
                player_turn = input("Play first or second (1=first, 2=second)? ")
                player_turn = int(player_turn)
            except:
                if 'q' in str(player_turn):
                    return
                else:
                    print("Type 'q' to quit")
        print(f"Player is {['O','X'][player_turn-1]}")
        player_turn = [1,-1][player_turn-1] # 1 = first, -1 = second
        
        board = T3Board()
        while board.end_state() is None:
            print(board)
            if board.player == player_turn:
                action = None
                while not board.play(action)[0]:
                    try:
                        action = input("Select play (type as 'r c' [e.g. '0 0' plays the top left square]): ")
                        action = [int(x) for x in action.split(' ')]
                    except:
                        if 'q' in str(action):
                            return
                        else:
                            print("Type 'q' to quit")
            else:
                board.play(az.play_vs_human(board, print_thinking=False))
        print(board)
        replay = 'y' in input("Play again? ")


if __name__ == '__main__':
    play_t3(opponent_network='goodNet_t3')
