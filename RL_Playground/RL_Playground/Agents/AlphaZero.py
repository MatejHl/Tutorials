import random
import tensorflow as tf
import numpy as np

from Algorithms.MCTreeSearch import MCTreeSearch, Node

class Agent:
    """
    """
    def __init__(self, name, state_space_size, action_space_size, model, optimizer, config):
        self.name = name
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size

        self.model = model

        self.tau = config.INIT_TAU

        self.mcts_n_simulations = config.MCTS_N_SIMULATIONS
        self.batch_size = config.BATCH_SIZE
        self.train_steps = config.TRAIN_STEPS
        self.weight_decay = config.WEIGHT_DECAY

        self.optimizer = optimizer

        self.config = config

        self.mcts = None

    def simulate(self):
        """
        get new leaf node (node that was not seen before), evaluate 
        its value and backpropagate value to other nodes along path 
        to the leaf node.
        
        Returns:
        --------
        None
        """
        # Get new leaf node:
        leaf, terminate_value, done, search_path = self.mcts.moveToLeaf()

        # Evaluate leaf node:
        value, search_path = self.mcts.evaluateLeaf(leaf, terminate_value, done, search_path, self.get_preds)

        # Backpropagate along the path:
        self.mcts.backFill(leaf, value, search_path)

        return None

    @tf.function
    def _get_preds(self, model, input, idx_to_keep):
        """
        run model to get value of current leaf node and 
        distribution for action to take from this leaf node.
        No action will be taken just yet, distribution and 
        possible states will be added to the tree, but values of
        possible states are not calculated.
        """
        value, logits = model(input) #.__call__(input)
        logits = tf.squeeze(logits)

        # Set logits for not allowed actions to -100
        
        to_keep = tf.gather(logits, idx_to_keep, axis = -1)
        idx_to_keep = tf.expand_dims(idx_to_keep, -1)
        sp_logits = tf.sparse.SparseTensor(idx_to_keep, to_keep, tf.shape(logits, out_type=tf.int64))
        logits = tf.sparse.to_dense(sp_logits, default_value = -100.0)

        probs = tf.nn.softmax(logits, name='softmax')

        return value, probs


    def get_preds(self, state):
        """
        """
        input = self.convertToModelInput([state])
        idx_to_keep = tf.convert_to_tensor(state.allowedActions, dtype = tf.int64)
        value, probs = self._get_preds(self.model, input, idx_to_keep)

        return value.numpy(), probs.numpy()


    def chooseAction(self):
        """
        action probability distribution for root 
        node is already influencing visit count.
        In case of non-determistic transotion probabilities 
        (same action leads to multiple states), mean action_value
        is calculated and N is summed for the same action.
        """
        pi = np.zeros(self.action_space_size, dtype = np.integer)
        action_values = np.zeros(self.action_space_size, dtype = np.float32)
        _n = np.zeros(self.action_space_size, dtype = np.integer)

        for edge in self.mcts.root.edges:
            # In case that action leads to multiple states, N for action is summed.
            pi[edge.action] = pi[edge.action] + edge.stats['N'] 
            action_values[edge.action] = action_values[edge.action] + edge.stats['Q']
            _n[edge.action] = _n[edge.action] + 1

        _n[_n==0] = 1
        action_values = np.divide(action_values, _n) # get mean action_value

        if self.tau == 0:
            actions = np.argwhere(pi == max(pi)).flatten()
            pi = np.zeros(self.action_space_size, dtype = np.integer)
            pi[actions] = 1.0/len(actions)
            action = np.random.choice(actions)
        else:
            pi = np.power(pi, 1.0/self.tau)
            pi = pi/(np.sum(pi))
            action = np.random.choice(np.arange(self.action_space_size), p = pi)

        value = action_values[action]

        return action, value, pi


    def act(self, state):
        """
        Returns:
        --------
        action : 
            action to take
        """

        if self.mcts is None or state.id not in self.mcts.tree:
            self.buildMCTS(state)
        else:
            self.changeRootMCTS(state)

        for sim in range(self.mcts_n_simulations):
            self.simulate()

        action, value, pi = self.chooseAction()
        return action, pi

    @tf.function
    def _train_step(self, model, batch_inputs, batch_values, batch_policy, opt, weight_decay):
        with tf.GradientTape as tape:
            value, logits = model(batch_inputs)
            loss_value = tf.math.reduce_mean(tf.math.squared_difference(value, batch_values),
                                             name = 'value_loss')
            loss_policy = tf.nn.softmax_cross_entropy_with_logits(labels = batch_policy, 
                                                                  logits = logits, 
                                                                  axis = -1, 
                                                                  name = 'policy_loss')
            loss = loss_value + loss_policy # Loss without regularization

            for weights in model.get_weights(): # Adding regularization
                loss += weight_decay * tf.nn.l2_loss(weights)

        grad = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grad, model.trainable_variables))
        return loss_value, loss_policy, loss


    def train(self, longMemory, ckpt):
        if self.optimizer is None:
            raise ValueError('optimizer is None.')

        for step in range(self.train_steps):
            batch = random.sample(longMemory, min(self.batch_size, len(longMemory)))
            
            batch_inputs = self.convertToModelInput([row['state'] for row in batch])
            
            batch_values = np.array([row['value'] for row in batch])
            batch_policy = np.array([row['policy'] for row in batch])
            
            loss_value, loss_policy, loss = self._train_step(model = self.model, 
                                                             batch_inputs = batch_inputs, 
                                                             batch_values = batch_values, 
                                                             batch_policy = batch_policy, 
                                                             opt = self.optimizer, 
                                                             weight_decay = self.weight_decay)
            tf.summary.scalar('loss_value', loss_value, step = ckpt.step)
            tf.summary.scalar('loss_policy', loss_policy, step = ckpt.step)
            tf.summary.scalar('loss', loss, step = ckpt.step)
            ckpt.step.assign_add(1)

        return None

    def buildMCTS(self, state):
        """
        """
        self.root = Node(state)
        self.mcts = MCTreeSearch(self.root, self.config)
        return None

    def changeRootMCTS(self, state):
        self.mcts.root = self.mcts.tree[state.id]
        return None

    def convertToModelInput(self, states):
        """
        This is to transform raw output of state to correct shape and type for model.
        It is used in Agent class.
        """
        return tf.cast(tf.stack([tf.reshape(state.trinary, shape = self.model.board_shape) for state in states]), dtype = tf.float32, name = 'convertToModelInput')