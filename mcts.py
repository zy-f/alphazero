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
        self.children = {}
        self.is_expanded = False
    
    def expand(self, action_dict):
        for action, p in action_dict.items():
            self.children[action] = Node(p=p, parent=self, action=action)
        self.is_expanded = True
    
    def update(self, value):
        self.n += 1
        self.w += value
        self.q = self.w/self.n

def select_child(node, c_puct=1):
        # print(node.children)
        actions, N, Q, P = zip(*[ [child.action, child.n, child.q, child.p] for child in node.children.values() ])
        N, Q, P = map(np.array, (N,Q,P))
        U = c_puct * P * np.sqrt(np.sum(N)) / (1+N)
        best_action = actions[np.argmax(Q+U)]
        return node.children[best_action]

def select_leaf(root, c_puct):
    node = root
    search_path = [root]
    while node.is_expanded:
        node = select_child(node, c_puct=1)
        search_path.append(node)
    return search_path

def evaluate(network, board):
    # NEED TO DO BOARD ROTATIONS N STUFF SOMEWHERE
    encoded = board.encoded()
    encoded = np.expand_dims(encoded, axis=0)
    pi, v = network.predict(encoded)
    return pi, v

def backup(leaf, v):
    node = leaf
    while node.parent is not None:
        node.update(v)
        v *= -1
        node = node.parent
    # root
    node.update(v)

def add_dirichlet_noise(root):
    child_ps = [ [action, child.p] for action, child in root.children.items()]
    noise_distrib = np.random.gamma(.03, 1, len(root.children))
    for k in range(len(child_ps)):
        child_ps[k][1] = .75*child_ps[k][1] + .25*noise_distrib[k]
    for action, child_p in child_ps:
        root.children[action].p = child_p
    return root

def search(board, network, n_sim=100, c_puct=1, learning=True):
    network.eval()
    root = Node()
    for _ in range(n_sim):
        if root.is_expanded and learning:
            root = add_dirichlet_noise(root)
        search_path = select_leaf(root, c_puct) # select
        leaf = search_path[-1]
        for node in search_path[1:]:
            board.play(node.action)
        pi, v = evaluate(network, board) # evaluate
        legal_actions = board.legal_actions()
        legal_action_idxs = board.actions_to_idxs(legal_actions)
        action_dict = {a:pi[ legal_action_idxs[a] ] for a in legal_actions}
        if len(action_dict) > 0:
            leaf.expand(action_dict)
        backup(leaf, v)
    return root

def get_pi(node, tau):
    actions, N = zip(*[ [child.action, child.n] for child in node.children.values() ])
    N = np.array(N)
    pi = N**(1/tau)
    pi /= np.sum(pi)
    return list(actions), pi

def self_play(storage):
    mcts_config = storage.mcts_config
    network = storage.latest_network()
    dataset = []
    for _ in trange(mcts_config.n_games, desc='MCTS Data Collection'):
        tau = 1
        board = storage.board_cls()
        s_arr = []
        pi_arr = []
        v = 0
        plays = 0
        while v == 0:
            c_puct = 1 if plays < mcts_config.c_puct_decay_n else 1e-3
            # ROTATIONS ARE MISSING
            s_arr.append(board.encoded())
            node = search(board.clone(), network, n_sim=mcts_config.n_sims_per_game_step, c_puct=c_puct, learning=True)

            actions, pi = get_pi(node, tau)
            action_idx = np.random.choice(range(len(pi)),p=pi)
            action = actions[action_idx]
            
            pi_temp = np.zeros(storage.board_cls.action_space)
            legal_action_idxs = board.actions_to_idxs(actions)
            for k in range(len(actions)):
                pi_temp[ legal_action_idxs[actions[k]] ] = pi[k]
            pi_arr.append(pi_temp)
            
            board.play(action)
            plays += 1
            
            v = board.end_state()
        if v**2 < 1:
            v_arr = [.5]*len(pi_arr) 
        else:
            v_arr = np.array( [(-1)**(k%2) for k in range(len(pi_arr))] )
            if v == -1:
                v_arr = -v_arr 
        dataset += list(zip(s_arr, pi_arr, v_arr))
    storage.save_sub_dataset(dataset)

def play_learned_action(network, board, n_sim=100):
    node = search(board.clone(), network, n_sim=100, c_puct=1e-3, learning=False)
    actions, pi = get_pi(node, 1)
    act_idx = np.argmax(pi)
    return actions[act_idx]
    
