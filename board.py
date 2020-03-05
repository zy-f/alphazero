import numpy as np
from copy import deepcopy

class Board(object):
    action_space = 0
    board_size = 0
    encoding_layers = 1

    def __init__(self, history=None):
        self.history = history or []
        self.player = [1,-1][len(history)%2] if history is not None else 1
        self.board = None

    def play(self, action, in_place=True):
        pass

    def end_state(self):
        pass

    def encoded(self):
        pass

    def legal_actions(self):
        pass
    
    def idx_action_map(self):
        pass

    def clone(self):
        return self.__class__(history=list(self.history), board=np.copy(self.board))
    
    def symmetries(self, pi):
        pass

class T3Board(Board):
    action_space = 9
    board_size = 9
    encoding_layers = 1

    def __init__(self, history=None, board=None):
        super(T3Board,self).__init__(history=history)
        self.board = board if board is not None else np.zeros((3,3), dtype=int)
    
    # action is an (r,c) tuple
    def play(self, action, in_place=True):
        try:
            action = tuple(action)
            board = np.copy(self.board)
            if board[action] == 0:
                board[action] = self.player
                if in_place:
                    self.history.append(action)
                    self.board = board
                    self.player *= -1
                    return (True, None)
                else:
                    return (True, board)
            return (False, None)
        except:
            return (False, None)
    
    # checks absolute end state (None=game not ended, 0=draw, 1=player 1 win, -1=player 2 win)
    def end_state(self):
        # horizontal/vertical checking
        if (np.sum(self.board,axis=0)**2 == 9).any() or (np.sum(self.board,axis=1)**2 == 9).any():
            return -self.player

        # diagonal checking
        if sum(self.board.diagonal(0))**2 == 9 or sum(np.fliplr(self.board).diagonal(0))**2 == 9:
            return -self.player

        # draw checking
        if not (self.board == 0).any():
            return 0
        return None

    def encoded(self):
        encoded = np.zeros((self.__class__.encoding_layers,3,3), dtype=np.float32)
        encoded[0,:,:] = self.board*self.player
        return encoded
    
    def symmetries(self, pi):
        board = self.encoded()
        pi = pi.reshape(3,3)
        stack = np.insert(board,1,pi,axis=0)
        out = []
        stack_to_tuple = lambda arr: (arr[:-1], arr[-1].reshape(-1))
        for k in range(4):
            stack = np.rot90(stack, k=k, axes=(-1,-2))
            out += [stack_to_tuple(stack), stack_to_tuple(np.fliplr(stack))]
        return out

    # returns a boolean array of length action_space
    def legal_actions(self):
        return self.board.reshape(-1) == 0
    
    def idx_action_map(self):
        mapping = [(k//3, k%3) for k in range(self.__class__.action_space)]
        return mapping

    def __str__(self):
        str_player = ['X','O'][int(self.player/2+.5)]
        str_board = self.board.astype(str)
        str_board = '\n'.join(['|'.join(str_board[k]) for k in range(len(str_board))])
        str_board = str_board.replace('-1','X').replace('1','O').replace('0',' ')
        is_won = self.end_state()
        if is_won is None:
            return f"\n{str_board}\n{str_player}'s turn"
        out_state_string = ['Draw','Player O Wins','Player X Wins'][int(is_won)]
        return f"\n{str_board}\n{out_state_string}"


class C4Board(Board):
    action_space = 7
    board_size = 42
    encoding_layers = 1

    def __init__(self, history=None, board=None):
        super(C4Board,self).__init__(history=history)
        self.board = board if board is not None else np.zeros((6,7), dtype=int)

    # action is an int from 0-6
    def play(self, action, in_place=True):
        try:
            board = np.copy(self.board)
            if (board[:,action]==0).any():
                lowest_dex = 0
                for k in range(6):
                    if board[k,action] != 0:
                        break
                    lowest_dex = k
                board[lowest_dex, action] = self.player
                if in_place:
                    self.history.append(action)
                    self.board = board
                    self.player *= -1
                    return (True, None)
                else:
                    return (True, board)
            return (False, None)
        except:
            return (False, None)

    def end_state(self):
        lines = []
        # rows
        lines += [self.board[k,:] for k in range(self.board.shape[0])]
        # columns
        lines += [self.board[:,k] for k in range(self.board.shape[1])]
        # forward diagonals \
        lines += [np.diagonal(self.board, offset=k) for k in range(-2, 4)]
        # back diagonals /
        lines += [np.diagonal(np.fliplr(self.board), offset=k) for k in range(-2, 4)]
        for line in lines:
            if np.sum(line==1)<4 and np.sum(line==-1)<4:
                continue
            repeats=0
            for k in range(1,len(line)):
                if line[k] == line[k-1]:
                    repeats += 1
                    if repeats > 2:
                        return line[k]
                else:
                    repeats=0
        if not (self.board==0).any():
            return 0
        return None

    def encoded(self):
        encoded = np.zeros((self.__class__.encoding_layers,6,7), dtype=np.float32)
        encoded[0,:,:] = self.board*self.player
        return encoded

    def legal_actions(self):
        return (self.board==0).any(axis=0)
    
    def idx_action_map(self):
        return range(7)
    
    def symmetries(self, pi):
        return [(self.encoded(), pi), (np.flip(self.encoded(), axis=-1), np.flip(pi))]
    
    def __str__(self):
        str_player = ['+','o'][int(self.player/2+.5)]
        str_board = self.board.astype(str)
        str_board = '\n'.join(['|'.join(str_board[k]) for k in range(len(str_board))])
        str_board = str_board.replace('-1','+').replace('1','o').replace('0',' ')
        is_won = self.end_state()
        if is_won is None:
            return f"\n{str_board}\n{str_player}'s turn"
        out_state_string = ['Draw','Player o Wins','Player + Wins'][int(is_won)]
        return f"\n{str_board}\n{out_state_string}"