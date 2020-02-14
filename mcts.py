from utils import *
from tqdm import tqdm, trange

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
        for action, p in action_dict:
            self.children[action] = Node(p=p, parent=self, action=action)
        self.is_expanded = True
    
    def update(self, value):
        self.n += 1
        self.w += value
        self.q = self.w/self.n

def select_child(node, c_puct=1):
        actions, N, Q, P = ( np.array(k) for k in zip(*[ [child.action, child.n, child.q, child.p] for child in node.children ]) )
        U = c_puct * P * np.sqrt(np.sum(N)) / (1+N)
        best_action = actions[np.argmax(Q+U)]
        return node.children[best_action]

def select_leaf(root, c_puct, add_noise=True):
    node = add_noise(root) if add_noise else root
    search_path = [root]
    while node.is_expanded:
        node = select_child(node, c_puct=1)
        search_path.append(node)
    return search_path

def evaluate(network, board):
    # NEED TO DO BOARD ROTATIONS N STUFF SOMEWHERE
    pi, v = network(board.encoded())
    return pi, v

def backup(leaf, v):
    node = leaf
    while node.parent is not None:
        node.update(v)
        v *= -1
        node = node.parent
    # root
    node.update(v)

def add_noise(root):
    root = [0.75*child.p + 0.25*np.random.dirichlet(np.zeros([len(root.children)], dtype=np.float32)+192) for child in root.children.keys()]
    return root

def search(board, network, n_sim=100, c_puct=1, learning=True):
    network.eval()
    root = Node()
    for _ in range(n_sim):
        search_path = select_leaf(root, c_puct, add_noise=(not learning)) # select
        leaf = search_path[-1]
        for node in search_path:
            board.play(node.action)
        pi, v = evaluate(network, board) # evaluate
        legal_actions = board.legal_actions()
        action_dict = {a:pi[ 3*a[0]+a[1] ] for a in legal_actions}
        leaf.expand(action_dict)
        backup(leaf, v)
    return root

def pi(node, tau):
    actions, N = ( np.array(k) for k in zip(*[ [child.action, child.n] for child in node.children ]) )
    pi = N**(1/tau)
    pi /= np.sum(pi)
    return actions, pi

def self_play(storage):
    mcts_config = storage.mcts_config
    network = storage.latest_network()
    dataset = []
    for _ in trange(mcts_config.n_games):
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
            node = search(board.clone(), network, n_sim=mcts_config.n_sims_per_game, c_puct=c_puct, learning=True)
            actions, pi = pi(node, tau)
            pi_arr.append({actions[k]: pi[k] for k in range(len(pi))})
            action = np.random.choice(actions,p=pi)
            board.play(action)
            plays += 1
            v = board.end_state()
        if v**2 < 1:
            v_arr = [.5]*len(pi_arr) 
        else:
            v_arr = [(-1)**(k%2) for k in range(len(pi_arr))]
        dataset += list(zip(s_arr, pi_arr, v_arr))
    storage.save_games(dataset)

def play_learned_action(network, board, n_sim=100):
    node = search(board.clone(), network, n_sim=100, c_puct=1e-3, learning=False)
    actions, pi = pi(node, 1)
    act_idx = np.argmax(pi)
    return actions[act_idx]
    
