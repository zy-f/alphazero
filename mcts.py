from utils import *
from tqdm import tqdm, trange
import numpy as np

class Node(object):
    """
    "Each node s in the search tree contains edges (s, a) for all legal actions a ∈ A(s).
    Each edge stores a set of statistics,
        { N(s,a), W(s,a), Q(s,a), P(s,a) }
    where N(s,a) is the visit count, W(s,a) is the total action value, Q(s,a) is the mean
    action value and P(s, a) is the prior probability of selecting that edge" (AlphaGo Zero supplement)
    """
    def __init__(self, p=0, parent=None, action=None):
        self.n = 0
        self.p = p
        self.w = 0
        self.q = 0
        self.parent = parent
        self.action = action
        self.children = []
        self.is_expanded = False
    
    def expand(self, node_params):
        for p, action in node_params:
            # print(f"action={action}, p={p}")
            self.children.append( Node(p=p, parent=self, action=action) ) 
        self.is_expanded = True
    
    def update(self, value):
        self.n += 1
        self.w += value
        self.q = self.w/self.n
    
    def __str__(self):
        return f"action={self.action}, p={self.p}, q={self.q}, n={self.n}, w={self.w}"

def select_child(node, c_puct=1):
        # print(node.children)
        N, Q, P = map( np.array, zip(*[ [child.n, child.q, child.p] for child in node.children ]) )
        # print(P)
        U = c_puct * P * np.sqrt(np.sum(N)) / (1+N)
        # print(f"Q={Q}\nU={U}\nQ+U={Q+U}")
        return node.children[np.argmax(Q+U)]

def select_leaf(root, c_puct):
    node = root
    search_path = [root]
    while node.is_expanded:
        node = select_child(node, c_puct=c_puct)
        # print(node)
        search_path.append(node)
    return search_path

def evaluate(network, board):
    # NEED TO DO BOARD ROTATIONS N STUFF SOMEWHERE
    network.eval()
    encoded = board.encoded()
    encoded = np.expand_dims(encoded, axis=0)
    pi, v = network.predict(encoded)
    return pi, v

def backup(leaf, v):
    node = leaf
    while node.parent is not None:
        node.update(v)
        # print(node)
        v *= -1
        node = node.parent
    # root
    node.update(v)

def add_dirichlet_noise(root):
    child_ps = np.array([child.p for child in root.children])
    noise_distrib = np.random.gamma(.1, 1, len(root.children))
    child_ps = .75*child_ps + .25*noise_distrib
    for k, child_p in enumerate(child_ps):
        root.children[k].p = child_p
    return root

def search(board, network, n_sim=100, c_puct=1, add_noise=True):
    root = Node()
    default_prob = board.legal_actions().astype(float)
    default_prob /= np.sum(default_prob)
    default = zip(default_prob, board.idx_action_map())
    root.expand(default)
    if add_noise:
        root = add_dirichlet_noise(root)
    # print(np.array([child.p for child in root.children]))
    
    for _ in range(n_sim):
        temp_board = board.clone()
        # print("SELECT")
        search_path = select_leaf(root, c_puct) # select
        leaf = search_path[-1]
        for node in search_path[1:]:
            temp_board.play(node.action)
        # print(temp_board)
        # print("EVAL")
        pi, v = evaluate(network, temp_board) # evaluate

        action_ps = pi * temp_board.legal_actions()
        if sum(action_ps) > 0:
            action_ps /= sum(action_ps)
            node_params = list(zip(action_ps, board.idx_action_map()))
            leaf.expand(node_params) # expand
            backup(leaf, -v) # backup
        else:
            z = temp_board.end_state()
            # if temp_board.player != board.player:
            #     z = -z
            backup(leaf, -z*board.player)
    # print(board)
    # print(board.player, [(child.q, child.n, child.w) for child in root.children])
    return root

def get_pi(node, tau, mask):
    N = np.array([child.n for child in node.children])
    # print(N)
    N *= mask
    # print(N)
    if tau == 0:
        pi = np.zeros(N.shape)
        pi[np.argmax(N)] = 1
        return pi
    pi = N**(1/tau)
    pi /= np.sum(pi)
    return pi

def self_play(storage):
    mcts_config = storage.mcts_config
    network = storage.latest_network()
    dataset = []
    for _ in trange(mcts_config.n_games, desc='MCTS Data Collection'):
        tau = 1
        c_puct = .5
        board = storage.board_cls()
        s_arr = []
        pi_arr = []
        v_arr = []
        z = None
        plays = 0
        while z is None:

            # ROTATIONS ARE MISSING
            node = search(board, network, n_sim=mcts_config.n_sims_per_game_step, c_puct=c_puct, add_noise=True)

            pi = get_pi(node, tau, mask=board.legal_actions())
            for b, p in board.rotations(pi):
                s_arr.append(b)
                pi_arr.append(p)
                v_arr.append(board.player)
            
            action_idx = np.random.choice(range(len(pi)),p=pi)
            action = board.idx_action_map()[action_idx]
            board.play(action)
            plays += 1
            
            z = board.end_state()
        
        v_arr = z * np.array(v_arr).astype(float)
        # if z == 0:
        #     v_arr += .5
        dataset += list(zip(s_arr, pi_arr, v_arr))
        # print(dataset[-2:])
    storage.save_sub_dataset(dataset)

def play_learned_action(network, board, n_sim=25, print_state=False):
    node = search(board, network, n_sim=n_sim, c_puct=.5, add_noise=True)
    pi = get_pi(node, tau=1, mask=board.legal_actions())
    funky, board_value = evaluate(network, board)
    actions = board.idx_action_map()
    if print_state:
        print("VALUE:", board_value)
        print(list(zip(actions,pi,funky)))
    act_idx = np.argmax(pi)
    return actions[act_idx]
    
