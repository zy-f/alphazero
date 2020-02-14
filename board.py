import numpy as np

class Board(object):
    def __init__(self, history):
        self.history = history or []
        self.player = [1,-1][len(history)%2] if history else 1

    def play(self, action):
        pass

    def check_end(self):
        pass

    def encoded(self):
        pass

    def legal_actions(self):
        pass

    def clone(self):
        pass
        
    def rotations(self):
        pass

class T3Board(Board):
    def __init__(self, history=None, board=None):
        super(T3Board,self).__init__(history)
        self.board = board if board is not None else np.zeros((3,3), dtype=int)
    
    # action is an (r,c) tuple
    def play(self, action):
        if self.board[action] == 0:
            self.history.append(action)
            self.board[action] = self.player
            self.player *= -1
            return True
        else:
            return False
    
    """
    "AlphaZero is provided with perfect knowledge of the game rules. These are used during
    MCTS, to simulate the positions resulting from a sequence of moves, to determine game
    termination, and to score any simulations that reach a terminal state." (AlphaZero 14)
    """
    def check_end(self):
        # horizontal/vertical checking
        if (np.sum(self.board,axis=0)**2 == 9).any() or (np.sum(self.board,axis=1)**2 == 9).any():
            return -self.player

        # diagonal checking
        if sum(self.board.diagonal(0))**2 == 9 or sum(np.fliplr(self.board).diagonal(0))**2 == 9:
            return -self.player

        # draw checking
        if not (self.board == 0).any():
            return 0.5
        return 0

    """
    "The M feature planes are composed of binary feature planes indicating the presence of the player’s pieces, 
    with one plane for each piece type, and a second set of planes indicating the presence of the opponent’s 
    pieces...There are an additional L constant-valued input planes denoting the player’s colour, the move
    number, and the state of special rules..." (AlphaZero 14-15)
    """
    """
    AlphaZero and AlphaGo Zero discuss the importance of including past board states in the network input.
    However, this is because repetitions are illegal/otherwise relevant in Go/Chess/Shogi, whereas tictactoe
    is "fully observable soley from the current [board state]" (AlphaGo supplement)
    """
    def encoded(self):
        encoded = np.zeros((3,3,3), dtype=int)
        # M planes
        encoded[:,:,0] = (self.board==1) # O's
        encoded[:,:,1] = (self.board==-1) # X's

        # L planes
        encoded[:,:,2] = self.player # current player (L1)
        return encoded

    def legal_actions(self):
        x,y = np.where(self.board == 0)
        actions = list(zip(list(x),list(y)))
        return actions

    def clone(self):
        return T3Board(history=self.history, board=self.board)

    def __str__(self):
        str_player = ['X','O'][int(self.player/2+.5)]
        str_board = self.board.astype(str)
        str_board = '\n'.join(['|'.join(str_board[k]) for k in range(len(str_board))])
        str_board = str_board.replace('-1','X').replace('1','O').replace('0',' ')
        return f"{str_player}'s turn:\n{str_board}"