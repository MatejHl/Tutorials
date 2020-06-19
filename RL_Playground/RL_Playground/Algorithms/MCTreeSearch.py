import numpy as np

print("WARNING: We assume that actions are only indices of some ordering of real actions")

class Node:
    """
    Attributes:
    -----------
    id : str
        nodes id. Is the same as state id.

    state : GameState, object
        state of the node.

    playerTurn : int
        id of player, that is on turn.

    edges : list
        list of edges coming out from the node in direction root -> leaf.
    """
    def __init__(self, state):
        self.id = state.id
        self.state = state
        self.playerTurn = state.playerTurn
        self.edges = []

    def isLeaf(self):
        return not len(self.edges) > 0



class Edge:
    """
    Attributes:
    -----------
    id : str
        combination of inNode.id and outNode.id

    inNode : Node, object
        start node

    outNode : Node, object
        end node

    action : 
        BLA

    playerTurn : int
        id of the player that took the action. inNode playerTurn.

    attrs : dict
        attributes of the edges. These are [N, W, Q, p], where
            N - is number of paths passing through edge
            W - is total value. It is used for convinience in order to not have to save value for every node.
            Q - action-value fanction value. It is given as W / N
            p - action probability (prior probability of action calculated when inNodde was visited)
        So when initializing edge, the attrs are N = 0, W = 0, Q = 0, p = pi(a|s), where pi is policy (model)
    """
    def __init__(self, inNode, outNode, action, action_prior):
        self.id = inNode.id + '|' + outNode.id
        self.inNode = inNode
        self.outNode = outNode
        self.action = action
        self.playerTurn = self.inNode.state.playerTurn
        self.attrs = {'N': 0,
                      'W': 0,
                      'Q': 0,
                      'p': action_prior}



class MCTreeSearch:
    """
    Attributes:
    -----------
    root : Node, object
        root node from which the tree search begins.

    tree : dict
        set of nodes in the graph. Each node has output 
        edges in node.edges 
        tree[node.id] = node


    Notes:
    ------
    From book Reinforcement Learning: An Introduction by Sutton and Barto [1]:
    "In more detail, each iteration of a basic version of MCTS consists of the following 
    four steps as illustrated in Figure 8.10:
        1.Selection. 
            Starting at the root node, a tree policy based on the action values attached 
            to the edges of the tree traverses the tree to select a leaf node.
        2.Expansion. 
            On some iterations (depending on details of the application), the tree is 
            expanded from the selected leaf node by adding one or more child nodes 
            reached from the selected node via unexplored actions.
        3.Simulation.
            From the selected node, or from one of its newly-added child nodes (if any), 
            simulation of a complete episode is run with actions selected by the rolloutpolicy. 
            The result is a Monte Carlo trial with actions selected first by the tree policy and 
            beyond the tree by the rollout policy.
        4.Backup.
            The return generated by the simulated episode is backed up to update,or to initialize, 
            the action values attached to the edges of the tree traversed by the tree policy in this 
            iteration of MCTS. No values are saved for the states and actions visited by the rollout 
            policy beyond the tree. Figure 8.10 illustrates this by showing a backup from the terminal 
            state of the simulated trajectory directly to thestate–action node in the tree where the 
            rollout policy began (though in general, theentire return over the simulated trajectory 
            is backed up to this state–action node)."

    Technically, if if you imagine undirected graph, than this is not tree, because 
    different action-state pairs can lead to the same state.

    Ref:
    ----
    [1] @book{10.5555/3312046,
         author = {Sutton, Richard S. and Barto, Andrew G.},
         title = {Reinforcement Learning: An Introduction},
         year = {2018},
         isbn = {0262039249},
         publisher = {A Bradford Book},
         address = {Cambridge, MA, USA}}
    """
    def __init__(self, root, config):
        self.root = root
        self.tree = dict()

        ### UCB formula
        self.UCBversion = config.UCBversion
        self.eps = config.EPS # 0.2
        self.alpha = config.ALPHA # 0.8
        if self.UCBversion == 'DeepMind':
            # Params for UCB from [3] (See maxUCB_edge below)
            self.pb_c_base = config.PB_C_BASE # 19652
            self.pb_c_init = config.PB_C_INIT # 1.25
            self.c_puct = None

        elif self.UCBversion == 'dirichlet':
            # Params for UCB from [4] (See maxUCB_edge below)
            self.pb_c_base = None
            self.pb_c_init = None
            self.c_puct = config.C_PUCT # 1.0



    def moveToLeaf(self):
        """
        forward pass. Starts in root node, goes forward and chooses next 
        node by the rule. All edges from non leaf nodes are in the tree.

        Parameters:
        -----------
        None

        Returns:
        --------
        currentNode : Node, object
            leaf node.

        search_path : list
            path that was taken to reach leaf node.

        done : bool
            True if game ended.

        terminate_value : float
            terminate_value is reward if game ends. if it is not used it has 
            arbitrary value (0) and it is not used. If the game ended then no 
            model evalution is needed and instead estimated action-value is a reward.

        Notes:
        ------
        assumtion is that policy and action-values for both players are calculated 
        by the same model. (self-play)
        """
        currentNode = self.root
        search_path = []

        done = 0
        terminate_value = 0

        while not currentNode.isLeaf():
            simulationEdge = self.maxUCB_edge(currentNode)
            newState, terminate_value, done = currentNode.state.takeAction(simulationEdge.action)
            currentNode = simulationEdge.outNode
            search_path.append(simulationEdge)
        
        return currentNode, terminate_value, done, search_path
    


    def evaluateLeaf(self, leaf, terminate_value, done, search_path, get_prediction):
        """
        get prior probabilities of actions and value of leaf.state and
        add to tree new leaf nodes, that can be reached. New leaf nodes 
        values and action distribution is not evaluated.

        Parameters:
        -----------
        get_prediction : object
            function that inside inference model in leaf.state and returns:
                value - of the leaf.state
                probs - probability distribution of actions. Actions that are 
                        not allowed should have neglitiable probability

        terminate_value : float
            value of the end-game node. If done == 0 then terminate_value will 
            not be used.

        done : bool
            if the game ended.

        search_path : list
            path from root to the leaf. This is not used and it will be only 
            propagated further.

        Returns:
        --------
        value : float
            value that will be backpropagated in backFill

        search_path : list
            path from root to the leaf. This is directly propagated to output.

        """
        if done == 0:
            value, probs = get_prediction(leaf.state)
            
            allowedActions = leaf.state.allowedActions
            probs = probs[allowedActions] # not allowed actions have neglitiable prob, so they can be discarded.

            # Add potential future leafs to tree.
            for action, action_prior in zip(allowedActions, probs):
                
                newState, _, _ = leaf.state.takeAction(action)

                if newState.id not in self.tree:
                    node = Node(newState)
                    self.addNode(node)
                else:
                    node = self.tree[newState.id]

                newEdge = Edge(leaf, node, action, action_prior)
                leaf.edges.append(newEdge)

        else:
            value = terminate_value


        return value, search_path



    def backFill(self, leaf, value, search_path):
        """
        backward pass. Propagates action-value form leaf node (this is the same 
        as value of the node) along the path.

        Parameters:
        -----------
        leaf : Node, object
            leaf node

        value : float
            action-value that will be propagated through search_path. This is 
            calculated by model or it is end-game reward.

        search_path : list
            list of edges that where visited when getting from root to leaf.

        Returns:
        --------
        None
        """        
        currentPlayer = leaf.state.playerTurn

        for edge in search_path:

            if edge.playerTrun == currentPlayer:
                direction = 1.0
            else:
                direction = -1.0

            edge.stats['N'] = edge.stats['N'] + 1
            edge.stats['W'] = edge.stats['W'] + direction * value
            edge.stats['Q'] = edge.stats['W'] / edge.stats['N']

        return None



    def addNode(self, node):
        """
        add node to tree.
        """
        self.tree[node.id] = node



    def maxUCB_edge(self, currentNode):
        """
        choose next action as argmax(UCB_score), where UCB score 
        is defined either in [3] (UCBversion == 'DeepMind') or 
        [4] (UCBversion == 'dirichlet')

        Parameters:
        -----------
        currentNode : Node, object
            node from which we are choosing action.

        Returns:
        --------
        simulationEdge : Edge, object
            edge that maximazes UCB score.

        Notes:
        ------
        See [1] for parameter chioce and Dirichlet distribution reasoning.

        Ref:
        ----
        [1] https://medium.com/oracledevs/lessons-from-alphazero-part-3-parameter-tweaking-4dceb78ed1e5
        [2] https://web.stanford.edu/~surag/posts/alphazero.html
        [3] https://gist.github.com/erenon/cb42f6656e5e04e854e6f44a7ac54023
        [4] https://medium.com/applied-data-science/how-to-build-your-own-alphazero-ai-using-python-and-keras-7f664945c188
        """
        maxUCB = -9999
        N_currNode = sum((edge.stats['N'] for edge in currentNode.edges.values())) # parent.visit_count in [3]
        if currentNode == self.root:
            # eps can decrease with increasing N_currNode because we already explored space.
            nu = np.random.dirichlet([self.alpha]*len(currentNode.edges))
            eps = self.eps
        else:
            nu = [0.0]*len(currentNode.edges)
            eps = 0.0
            

        if self.UCBversion == 'DeepMind':
            # UCB form DeepMind pseudocode [3]
            _pb_c = math.log((N_currNode + self.pb_c_base + 1)/self.pb_c_base) + self.pb_c_init
        elif self.UCBversion == 'dirichlet':
            # UCB from [4]
            _pb_c = self.c_puct


        for idx, edge in enumerate(currentNode.edges):
            pb_c = _pb_c * np.sqrt(N_currNode) / (edge.stats['N'] + 1)
            UCB = edge.stats['Q'] + pb_c * ((1.0 - eps)*edge.stats['p'] + eps*nu[idx])
            if UCB > maxUCB:
                maxUCB = UCB
                simulationEdge = edge

        return simulationEdge