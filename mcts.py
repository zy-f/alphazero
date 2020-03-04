from utils import *
from tqdm import tqdm, trange
import numpy as np

"""
"Each node s in the search tree contains edges (s, a) for all legal actions a ∈ A(s).
Each edge stores a set of statistics,
    { N(s,a), W(s,a), Q(s,a), P(s,a) }
where N(s,a) is the visit count, W(s,a) is the total action value, Q(s,a) is the mean
action value and P(s, a) is the prior probability of selecting that edge" (AlphaGo Zero supplement)
"""
class Node(object):
    def __init__(self, state=None):
        self.in_edges = {}
        self.out_edges = []
        self.state = state
        self.is_expanded = False
    
    def expand(self, edge_params):
        for p, action, child_node in edge_params:
            # print(f"action={action}, p={p}")
            self.out_edges.append( Edge(parent_node=self, action=action, child_node=child_node, p=p) ) 
        self.is_expanded = True
    
    def __str__(self):
        edges = '\n'.join([str(edge) for edge in self.out_edges])
        return f"state:\n{self.state}\n{edges}"

class Edge(object):
    def __init__(self, parent_node, action, child_node, p=0):
        self.parent_node = parent_node
        self.action = tuple(action)
        self.child_node = child_node
        self.n = 0
        self.w = 0
        self.q = 0
        if child_node is None:
            self.p = 0
            self.w = -float('inf')
            self.q = -float('inf')
        else:
            self.p = p
            self.child_node.in_edges[self.action] = self
    
    def update(self, value):
        self.n += 1
        self.w += value
        self.q = self.w/self.n
    
    def __str__(self):
        return f"{self.action} => n={self.n}, q={self.q}, p={self.p}"

def select_child(node, c_puct=1):
        # print(node)
        N, Q, P = map( np.array, zip(*[ [edge.n, edge.q, edge.p] for edge in node.out_edges ]) )
        # print(P)
        U = c_puct * P * np.sqrt(np.sum(N)) / (1+N)
        # print(f"Q={Q}\nU={U}\nQ+U={Q+U}")
        best_edge = node.out_edges[np.argmax(Q+U)]
        return best_edge.child_node, best_edge.action

def select_leaf(root, c_puct):
    node = root
    search_path = []
    while node.is_expanded:
        node, action = select_child(node, c_puct=c_puct)
        # print(node)
        search_path.append(action)
    return node, search_path

def evaluate(network, board):
    network.eval()
    encoded = board.encoded()
    encoded = np.expand_dims(encoded, axis=0)
    pi, v = network.predict(encoded)
    return pi, v

def backup(leaf, action_path, v):
    node = leaf
    for act in action_path[::-1]:
        v *= -1
        edge = node.in_edges[act]
        edge.update(v)
        # print(edge)
        node = edge.parent_node

def add_dirichlet_noise(node):
    child_ps = np.array([edge.p for edge in node.out_edges])
    noise_distrib = np.random.gamma(.1, 1, len(node.out_edges))
    child_ps = .75*child_ps + .25*noise_distrib
    for k, child_p in enumerate(child_ps):
        node.out_edges[k].p = child_p
    return node

def get_child_nodes(board, actions, known_states):
    out = []
    for action in actions:
        is_legal, new_board_state = board.play(action, in_place=False)
        if is_legal:
            str_board_state = repr(new_board_state.tolist())
            if str_board_state not in known_states.keys():
                known_states[str_board_state] = Node(state=new_board_state)
            out.append( known_states[str_board_state] )
        else:
            out.append(None)
    return out, known_states 

def search(board, network, n_sim=100, c_puct=1, add_noise=True, verbose=False):
    root = Node(state=board.board)
    known_states = {repr(board.board.tolist()): root}
    for _ in range(n_sim):
        temp_board = board.clone()
        # print("SELECT")
        leaf, search_path = select_leaf(root, c_puct) # select
        for act in search_path:
            temp_board.play(act)
        # print(temp_board)
        # print("EVAL")
        pi, v = evaluate(network, temp_board) # evaluate

        action_ps = pi * temp_board.legal_actions()
        z = temp_board.end_state()
        if sum(action_ps) > 0 and z is None:
            action_ps /= sum(action_ps)
            child_nodes, known_states = get_child_nodes(temp_board, board.idx_action_map(), known_states)
            edge_params = list(zip(action_ps, board.idx_action_map(), child_nodes))
            leaf.expand(edge_params) # expand
            if add_noise and len(search_path) < 1:
                root = add_dirichlet_noise(root)
                add_noise = False
            backup(leaf, search_path, v) # backup
        elif z is not None:
            backup(leaf, search_path, z*temp_board.player)
    if verbose:
        print(root)
    return root

def get_pi(node, tau):
    N = np.array([edge.n for edge in node.out_edges])
    actions = [edge.action for edge in node.out_edges]
    # print(N)
    if tau == 0:
        pi = np.zeros(N.shape)
        pi[np.argmax(N)] = 1
        return actions, pi
    pi = N**(1/tau)
    pi /= np.sum(pi)
    # print(list(zip(actions, pi)))
    return actions, pi

def self_play(storage):
    mcts_config = storage.mcts_config
    network = storage.latest_network()
    dataset = []
    for _ in trange(mcts_config.n_games, desc='MCTS Data Collection'):
        tau = 1
        c_puct = 1
        board = storage.board_cls()
        s_arr = []
        pi_arr = []
        v_arr = []
        z = None
        plays = 0
        while z is None:

            # ROTATIONS ARE MISSING
            node = search(board, network, n_sim=mcts_config.n_sims_per_game_step, c_puct=c_puct, add_noise=True, verbose=False)#(plays>5))

            actions, pi = get_pi(node, tau)
            for b, p in board.rotations(pi):
                s_arr.append(b)
                pi_arr.append(p)
                v_arr.append(board.player)
            
            action_idx = np.random.choice(range(len(pi)),p=pi)
            action = actions[action_idx]
            board.play(action)
            plays += 1
            
            z = board.end_state()
        
        v_arr = z * np.array(v_arr).astype(float)
        # if z == 0:
        #     v_arr += .5
        dataset += list(zip(s_arr, pi_arr, v_arr))
    storage.save_sub_dataset(dataset)
    # storage.print_dataset_chunk()

def get_learned_action(network, board, n_sim=25, print_state=False):
    node = search(board, network, n_sim=n_sim, c_puct=1, add_noise=True, verbose=print_state)
    actions, pi = get_pi(node, tau=1)
    funky, board_value = evaluate(network, board)
    if print_state:
        print("VALUE:", board_value)
        print(list(zip(actions,pi,funky)))
    act_idx = np.argmax(pi)
    return actions[act_idx]
