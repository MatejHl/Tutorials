"""
From https://github.com/deepmind/jraph/blob/master/jraph/examples/basic.py
"""

import jax
import jax.numpy as jnp
import jraph
import numpy as np

def main():
    # Creates a GraphsTuple from scratch containing a single graph.
    single_graph = jraph.GraphsTuple(
      n_node=np.asarray([3]), 
      n_edge=np.asarray([2]),
      nodes=np.ones((3, 4)), 
      edges=np.ones((2, 5)),
      globals=np.ones((1, 6)),
      senders=np.array([0, 1]),    # Adjacency matrix row idx (from)
      receivers=np.array([2, 2]))  # Adjacency matrix col idx (to)
    # print(single_graph)

    # Creates a GraphsTuple from scatch containing a single graph with nested feature vectors.
    # The graph has 3 nodes and 2 edges.
    nested_graph = jraph.GraphsTuple(n_node=np.asarray([3]), 
                                    n_edge=np.asarray([2]),
                                    nodes={"a": np.ones((3, 4))},   # n_nodes x F_nodes  
                                    edges={"b": np.ones((2, 5))},   # n_edges x F_edges  
                                    globals={"c": np.ones((1, 6))}, # n_graphs x F_global
                                    senders=np.array([0, 1]), 
                                    receivers=np.array([2, 2]))
    # print(nested_graph)

    # Creates a GraphsTuple from scratch containing a 2 graphs using an implicit batch dimension.
    # The first graph has 3 nodes and 2 edges.
    # The second graph has 2 nodes and 1 edges.
    # This is similar to DisjointMode in spektral
    implicitly_batched_graph = jraph.GraphsTuple(n_node=np.asarray([3, 2]), 
                                                n_edge=np.asarray([2, 1]),
                                                nodes=np.concatenate([np.ones((3, 4)), 10*np.ones((2, 4))], axis = 0), # n_nodes x F_nodes   
                                                edges=np.ones((4, 5)),    # n_edges x F_edges  
                                                globals=np.ones((2, 6)),  # n_graphs x F_global   
                                                senders=np.array([0, 1, 3, 3]), 
                                                receivers=np.array([2, 2, 3, 4]))
    # print(implicitly_batched_graph)

    # Creates a GraphsTuple from two existing GraphsTuple using an implicit batch dimension.
    combined_batched_graph = jraph.batch([single_graph, implicitly_batched_graph])

    # print(combined_batched_graph)

    # Creates multiple GraphsTuples from an existing GraphsTuple with an implicit batch dimension.
    graph_1, graph_2, graph_3 = jraph.unbatch(combined_batched_graph)
    print(graph_1)
    print('----------')
    print(graph_2)
    print('----------')
    print(graph_3)
    print('----------')

    # Creates a padded GraphsTuple from an existing GraphsTuple.
    padded_graph = jraph.pad_with_graphs(single_graph, n_node=10, n_edge=5, n_graph=5)
    print(padded_graph)
  
    # Creates a GraphsTuple from an existing padded GraphsTuple.
    # padded is removed
    single_graph = jraph.unpad_with_graphs(padded_graph)
    print(single_graph)

    # Explicitly batched graphs
    # This is similar to BatchedMode in spektral
    explicitly_batched_graph = jraph.GraphsTuple(n_node=np.asarray([[3], [1]]), 
                                                n_edge=np.asarray([[2], [1]]),
                                                nodes=np.ones((2, 3, 4)), 
                                                edges=np.ones((2, 2, 5)),
                                                globals=np.ones((2, 1, 6)),
                                                senders=np.array([[0, 1], [0, -1]]),
                                                receivers=np.array([[2, 2], [0, -1]]))
    print(explicitly_batched_graph)

    def update_edge_fn(
          edge_features,
          sender_node_features,
          receiver_node_features,
          globals_):
        """Returns the update edge features."""
        del sender_node_features
        del receiver_node_features
        del globals_
        return 2.0*edge_features

    def update_node_fn(
          node_features,
          aggregated_sender_edge_features,
          aggregated_receiver_edge_features,
          globals_):
        """Returns the update node features."""
        del aggregated_sender_edge_features
        del aggregated_receiver_edge_features
        del globals_
        return -2.0*node_features

    def update_globals_fn(
          aggregated_node_features,
          aggregated_edge_features,
          globals_):
        del aggregated_node_features
        del aggregated_edge_features
        return globals_
    
    # Optionally define custom aggregation functions.
    # In this example we use the defaults (so no need to define them explicitly).
    aggregate_edges_for_nodes_fn = jax.ops.segment_sum
    aggregate_nodes_for_globals_fn = jax.ops.segment_sum
    aggregate_edges_for_globals_fn = jax.ops.segment_sum

    attention_logit_fn = None
    attention_reduce_fn = None

    # General graph neural network:
    network = jraph.GraphNetwork(update_edge_fn = update_edge_fn,
                                update_node_fn = update_node_fn,
                                update_global_fn = update_globals_fn,
                                attention_logit_fn = attention_logit_fn,
                                aggregate_edges_for_nodes_fn = aggregate_edges_for_nodes_fn,
                                aggregate_nodes_for_globals_fn = aggregate_nodes_for_globals_fn,
                                aggregate_edges_for_globals_fn = aggregate_edges_for_globals_fn,
                                attention_reduce_fn = attention_reduce_fn)

    updated_graph = network(single_graph)
    print('-------')
    print(updated_graph)
    print('-------')
    updated_graph = network(implicitly_batched_graph)
    print('-------')
    print(updated_graph)
    print('-------')

    # JIT-compile graph propagation.
    # Use padded graphs to avoid re-compilation at every step! ! ! <--------------
    jitted_network = jax.jit(network)
    # padded_graph = jraph.pad_with_graphs(single_graph, n_node=10, n_edge=5, n_graph=5)
    updated_graph = jitted_network(padded_graph)

if __name__ == '__main__':
    main()