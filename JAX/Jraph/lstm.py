import collections
import numpy as np

import jax
import jax.numpy as jnp
import haiku as hk

import jraph


NUM_NODES = 5
NUM_EDGES = 7
NUM_MESSAGE_PASSING_STEPS = 10
EMBEDDING_SIZE = 32
HIDDEN_SIZE = 128

# Immutable class for storing nested node/edge features containing an embedding
# and a recurrent state..
StatefulField = collections.namedtuple("StatefulField", ["embedding", "state"])

def get_random_graph() -> jraph.GraphsTuple:
    NUM_NODES = np.random.randint(3, 8)
    NUM_EDGES = np.random.randint(5, 12)
    return jraph.GraphsTuple(n_node = np.asarray([NUM_NODES]),
                            n_edge = np.asarray([NUM_EDGES]),
                            nodes = np.random.normal(size = [NUM_NODES, EMBEDDING_SIZE]),
                            edges = np.random.normal(size = [NUM_EDGES, EMBEDDING_SIZE]),
                            globals = None,
                            senders=np.random.randint(0, NUM_NODES, [NUM_EDGES]),
                            receivers=np.random.randint(0, NUM_NODES, [NUM_EDGES]))

def network_definition(graph: jraph.GraphsTuple) -> jraph.ArrayTree:
    """
    `InteractionNetwork` with an LSTM in the edge update.
    """
    edge_fn_LSTM = hk.LSTM(hidden_size=HIDDEN_SIZE) # LSTM 

    edge_fn_mlp = hk.nets.MLP([HIDDEN_SIZE, EMBEDDING_SIZE])
    node_fn_mlp = hk.nets.MLP([HIDDEN_SIZE, EMBEDDING_SIZE])

    graph = graph._replace(edges = StatefulField(embedding = graph.edges, 
                                                state = edge_fn_LSTM.initial_state(graph.edges.shape[0])),
                            nodes = StatefulField(embedding = graph.nodes, state = None))

    def update_edge_fn(edges, sender_nodes, receiver_nodes):
        """
        We will run an LSTM memory on the inputs first, and then
        process the output of the LSTM with an MLP.
        """
        edge_inputs = jnp.concatenate([edges.embedding,
                                        sender_nodes.embedding,
                                        receiver_nodes.embedding], axis = -1)
        lstm_output, updated_state = edge_fn_LSTM(edge_inputs, edges.state)
        updated_embedding = edge_fn_mlp(lstm_output)
        updated_edges = StatefulField(embedding = updated_embedding, state = updated_state)
        return updated_edges

    def update_node_fn(nodes, received_edges):
        """
        Note `received_edges.state` will also contain the aggregated state for
        all received edges, which we may choose to use in the node update.
        """
        node_inputs = jnp.concatenate([nodes.embedding, received_edges.embedding], axis = -1)
        updated_embedding = node_fn_mlp(node_inputs)
        update_nodes = StatefulField(embedding = updated_embedding, state = None)
        return update_nodes

    
    reccurent_graph_network = jraph.InteractionNetwork(update_edge_fn = update_edge_fn,
                                                        update_node_fn = update_node_fn)

    for _ in range(NUM_MESSAGE_PASSING_STEPS):
        graph = reccurent_graph_network(graph)

    return graph


if __name__ == '__main__':
    network = hk.without_apply_rng(hk.transform(network_definition))

    jitted_apply = jax.jit(network.apply)

    input_graph = get_random_graph()
    params = network.init(jax.random.PRNGKey(42), input_graph)

    import time
    N = 100
    graphs = [get_random_graph() for _ in range(N)]
    batched_graph = jraph.batch(graphs)

    print(jax.tree_util.tree_map(lambda x: x.shape, batched_graph))

    graphs_2 = [get_random_graph() for _ in range(N)]
    batched_graph_2 = jraph.batch(graphs_2)

    print(jax.tree_util.tree_map(lambda x: x.shape, batched_graph_2))

    start = time.time()
    output_graph = network.apply(params, batched_graph)
    end = time.time()
    print("Apply: {}".format(end-start))

    start = time.time()
    output_graph = jitted_apply(params, batched_graph)
    end = time.time()
    print("Jitted apply: {}".format(end-start))

    print(jax.tree_util.tree_map(lambda x: x.shape, output_graph))

    print('Done...')
