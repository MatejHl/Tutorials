import jax
import jax.numpy as jnp
import jraph

import haiku as hk
import optax

def get_zacharys_karate_club() -> jraph.GraphsTuple:
    """
    """
    social_graph = [
      (1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2),
      (4, 0), (5, 0), (6, 0), (6, 4), (6, 5), (7, 0), (7, 1),
      (7, 2), (7, 3), (8, 0), (8, 2), (9, 2), (10, 0), (10, 4),
      (10, 5), (11, 0), (12, 0), (12, 3), (13, 0), (13, 1), (13, 2),
      (13, 3), (16, 5), (16, 6), (17, 0), (17, 1), (19, 0), (19, 1),
      (21, 0), (21, 1), (25, 23), (25, 24), (27, 2), (27, 23),
      (27, 24), (28, 2), (29, 23), (29, 26), (30, 1), (30, 8),
      (31, 0), (31, 24), (31, 25), (31, 28), (32, 2), (32, 8),
      (32, 14), (32, 15), (32, 18), (32, 20), (32, 22), (32, 23),
      (32, 29), (32, 30), (32, 31), (33, 8), (33, 9), (33, 13),
      (33, 14), (33, 15), (33, 18), (33, 19), (33, 20), (33, 22),
      (33, 23), (33, 26), (33, 27), (33, 28), (33, 29), (33, 30),
      (33, 31), (33, 32)]
    # Add reverse edges.
    social_graph += [(edge[1], edge[0]) for edge in social_graph]
    n_club_members = 34

    return jraph.GraphsTuple(n_node = jnp.asarray([n_club_members]),
                            n_edge = jnp.asarray([len(social_graph)]),
                            nodes = jnp.eye(n_club_members), # One-hot encoding for nodes
                            edges = None, # No edge features
                            globals = None,
                            senders = jnp.asarray([edge[0] for edge in social_graph]),
                            receivers = jnp.asarray([edge[1] for edge in social_graph])
                            )

def get_ground_truth_assignments() -> jnp.ndarray:
    """
    """
    return jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1,
                    0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

def network_definition(graph: jraph.GraphsTuple) -> jraph.ArrayTree:
    """
    """
    gn = jraph.GraphConvolution(update_node_fn = lambda n: jax.nn.relu(hk.Linear(5)(n)),
                                add_self_edges = True)
    graph = gn(graph)

    gn = jraph.GraphConvolution(update_node_fn = hk.Linear(2),
                                add_self_edges = False)
    graph = gn(graph)
    return graph.nodes

def optimize_club(num_steps: int):
    network = hk.without_apply_rng(hk.transform(network_definition))
    data = get_zacharys_karate_club()
    labels = get_ground_truth_assignments()
    params = network.init(jax.random.PRNGKey(42), data)

    @jax.jit
    def prediction_loss(params):
        decoded_nodes = network.apply(params, data)
        log_prob = jax.nn.log_softmax(decoded_nodes)
        # The only two assignments we know a-priori are those of Mr. Hi (Node 0)
        # and John A (Node 33).
        return -(log_prob[0, 0] + log_prob[33, 1])

    opt_init, opt_update = optax.adam(1e-2)
    opt_state = opt_init(params)

    @jax.jit
    def update(params, opt_state):
        g = jax.grad(prediction_loss)(params)
        updates, opt_state = opt_update(g, opt_state)
        return optax.apply_updates(params, updates), opt_state

    @jax.jit
    def accuracy(params):
        decoded_nodes = network.apply(params, data)
        return jnp.mean(jnp.argmax(decoded_nodes, axis = 1) == labels)

    for step in range(num_steps):
        print("Acc: {}".format(accuracy(params).item()))
        params, opt_state = update(params, opt_state)

if __name__ == '__main__':
    optimize_club(num_steps = 30)