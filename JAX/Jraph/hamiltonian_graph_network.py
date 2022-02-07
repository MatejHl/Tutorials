import functools

import jax
import jax.numpy as jnp
import jraph

import matplotlib.pyplot as plt
import numpy as np

from typing import Tuple, Callable

from frozendict import frozendict


jax.tree_util.register_pytree_node(frozendict,
                                flatten_func = lambda s: (tuple(s.values()), tuple(s.keys())),
                                unflatten_func = lambda k, xs: frozendict(zip(k, xs)) )


def hookes_hamiltonian_from_graph_fn(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    """
    Computes Hamiltonian of a Hooke's potential system represented in a graph.

    While this function hardcodes the Hamiltonian for a Hooke's potential, a
    learned Hamiltonian Graph Network (https://arxiv.org/abs/1909.12790) could
    be implemented by replacing the hardcoded formulas by learnable MLPs that
    take as inputs all of the concatenated features to the edge_fn, node_fn,
    and global_fn, and outputs a single scalar value in the global_fn.

    Args:
      graph: `GraphsTuple` where the nodes contain:
          - "mass": [num_particles]
          - "position": [num_particles, num_dims]
          - "momentum": [num_particles, num_dims]
          and the edges contain:
          - "spring_constant": [num_interations]
    Returns:
      `GraphsTuple` with features:
          - edge features: "hookes_potential" [num_interactions]
          - node features: "kinetic_energy" [num_particles]
          - global features: "hamiltonian" [batch_size]

    """
    def update_edge_fn(edges, senders, receivers, globals_):
        del globals_
        distance = jnp.linalg.norm(senders['position'] - receivers['position'])
        hookes_potential_per_edge = 0.5 * edges["spring_constant"] * distance**2
        return frozendict({'hookes_potential': hookes_potential_per_edge})

    def update_node_fn(nodes, sent_edges, received_edges, globals_):
        del sent_edges, received_edges, globals_
        momentum_norm = jnp.linalg.norm(nodes["momentum"])
        kinetic_energy_per_node = momentum_norm**2 / (2 * nodes["mass"])
        return frozendict({"kinetic_energy" : kinetic_energy_per_node})

    def update_global_fn(nodes, edges, globals_):
        del globals_
        hamiltonian_per_graph = nodes["kinetic_energy"] + edges["hookes_potential"]
        return frozendict({"hamiltonian" : hamiltonian_per_graph})

    gn = jraph.GraphNetwork(update_edge_fn = update_edge_fn,
                            update_node_fn = update_node_fn,
                            update_global_fn = update_global_fn)

    return gn(graph)


def get_random_uniform_norm2d_vectors(min_norm: float, max_norm: float, num_particles: int) -> np.ndarray:
    """
    Returns 2-d vectors with random norms.
    """
    norm = np.random.uniform(min_norm, max_norm, [num_particles, 1])
    angle = np.random.uniform(0, 2*np.pi, [num_particles])
    return norm * np.stack([np.cos(angle), np.sin(angle)], axis = -1)

# Generating data:
def build_hookes_particle_state_graph(num_particles: int) -> jraph.GraphsTuple:
    """
    """
    mass = np.random.uniform(0, 5, [num_particles])
    velocity = get_random_uniform_norm2d_vectors(0, 0.1, num_particles)
    position = get_random_uniform_norm2d_vectors(0, 1, num_particles)
    momentum = velocity * np.expand_dims(mass, axis = -1)
    # Remove average momentum, so center of mass does not move.
    momentum = momentum - momentum.mean(0, keepdims=True)

    # Connect all particles to all others.
    particle_indices = np.arange(num_particles)
    senders, receivers = np.meshgrid(particle_indices, particle_indices)
    senders, receivers = senders.flatten(), receivers.flatten()

    # Generate a symmetric random matrix of spring constants.
    # Generate random elements stringly in the lower triangular part.
    spring_constants = np.random.uniform(1e-2, 1e-1, [num_particles, num_particles])
    spring_constants = np.tril(spring_constants) + np.tril(spring_constants, -1).T
    spring_constants = spring_constants.flatten()

    # Remove interations of particles to themselves
    mask = senders != receivers
    senders, receivers = senders[mask], receivers[mask]
    spring_constants = spring_constants[mask]
    num_interactions = receivers.shape[0]

    return jraph.GraphsTuple(n_node = np.asarray([num_particles]),
                            n_edge = np.asarray([num_interactions]),
                            nodes = {"mass" : mass,
                                    "position" : position,
                                    "momentum" : momentum},
                            edges = {"spring_constant" : spring_constants},
                            globals = {},
                            senders = senders,
                            receivers = receivers)


def set_system_state(static_graph: jraph.GraphsTuple, position: np.ndarray, momentum: np.ndarray) -> jraph.GraphsTuple:
    """
    """
    nodes = static_graph.nodes.copy(position=position, momentum=momentum)
    return static_graph._replace(nodes = nodes)

def get_system_state(graph: jraph.GraphsTuple) -> Tuple[np.ndarray, np.ndarray]:
    return graph.nodes["position"], graph.nodes["momentum"]

def get_static_graph(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    """
    Returns the graph with the static parts of a system only.
    """
    nodes = dict(graph.nodes)
    del nodes["position"], nodes["momentum"]
    return graph._replace(nodes = frozendict(nodes))

# Utility methods to operate with Hamiltonian functions.
def get_hamiltonian_from_state_fn(static_graph: jraph.GraphsTuple, 
                                hamiltonian_from_graph_fn: Callable[[jraph.GraphsTuple], jraph.GraphsTuple]
                                ) -> Callable[[np.ndarray, np.ndarray], float]:
    """
    Returns fn such that fn(position, momentum) -> scalar Hamiltonian.

    Args:
        static_graph: `GraphsTuple` containing per-particle static parameters and
           connectivity, such as a full graph of the state can be build by calling
           `set_system_state(static_graph, position, momentum)`.
        hamiltonian_from_graph_fn: callable that given an input `GraphsTuple`
           returns a `GraphsTuple` with a "hamiltonian" field in the globals.
    Returns:
        Function that given a state (position, momentum) returns the scalar
        Hamiltonian.
    """
    def hamiltonian_from_state_fn(position, momentum):
        # Note we sum along the batch dimension to get the total energy in the batch
        # so get can easily get the gradient.
        graph = set_system_state(static_graph, position, momentum)
        output_graph = hamiltonian_from_graph_fn(graph)
        return output_graph.globals["hamiltonian"].sum()

    return hamiltonian_from_state_fn


def get_state_derivatives_from_hamiltonian_fn(hamiltonian_from_state_fn: Callable[[np.ndarray, np.ndarray], float],
        ) -> Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Returns fn(position, momentum, ...) -> (dposition_dt, dmomentum_dt).

    Args:
        hamiltonian_from_state_fn: Function that given a state
            (position, momentum)  returns the scalar Hamiltonian.
    Returns:
        Function that given a state (position, momentum) returns the time
        derivatives of the state (dposition_dt, dmomentum_dt) by applying
        Hamilton equations.
    """
    hamiltonian_gradients_fn = jax.grad(hamiltonian_from_state_fn, argnums=[0,1])

    def state_derivatives_from_hamiltonian_fn(
          position: np.ndarray, momentum: np.ndarray
          ) -> Tuple[np.ndarray, np.ndarray]:
        # Take the derivatives against position and momentum.
        dh_dposition, dh_dmomentum = hamiltonian_gradients_fn(position, momentum)

        # Hamilton equations.
        dposition_dt = dh_dmomentum
        dmomentum_dt = - dh_dposition
        return dposition_dt, dmomentum_dt

    return state_derivatives_from_hamiltonian_fn


# -----------------------------------------------------------------------------------------
# Implementations of some general purpose integrators for Hamiltonian states.
StateDerivativesFnType = Callable[
    [np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]


def abstract_integrator(
    position: np.ndarray, momentum: np.ndarray, time_step: float,
    state_derivatives_fn: StateDerivativesFnType,
    ) -> Tuple[np.ndarray, np.ndarray]:
  """Signature of an abstract integrator.
  An integrator is a function, that given the the current state, a time step,
  and a `state_derivatives_fn` returns the next state.
  Args:
      position: array with the position at time t.
      momentum: array with the momentum at time t.
      time_step: integration step size.
      state_derivatives_fn: a function fn, that returns time derivatives of a
          state such fn(position, momentum) -> (dposition_dt, dmomentum_dt)
          where dposition_dt, dmomentum_dt, have the same shapes as
          position, momentum.
  Returns:
      Tuple with position and momentum at time `t + time_step`.
  """
  raise NotImplementedError("Abstract integrator")


def euler_integrator(
    position: np.ndarray, momentum: np.ndarray, time_step: float,
    state_derivatives_fn: StateDerivativesFnType,
    ) -> Tuple[np.ndarray, np.ndarray]:
  """Implementation of an Euler integrator (see `abstract_integrator`)."""
  dposition_dt, dmomentum_dt = state_derivatives_fn(position, momentum)
  next_position = position + dposition_dt * time_step
  next_momentum = momentum + dmomentum_dt * time_step
  return next_position, next_momentum


def verlet_integrator(
    position: np.ndarray, momentum: np.ndarray, time_step: float,
    state_derivatives_fn: StateDerivativesFnType,
    ) -> Tuple[np.ndarray, np.ndarray]:
  """Implementation of Verlet integrator (see `abstract_integrator`)."""

  _, dmomentum_dt = state_derivatives_fn(position, momentum)
  aux_momentum = momentum + dmomentum_dt * time_step / 2

  dposition_dt, _ = state_derivatives_fn(position, aux_momentum)
  next_position = position + dposition_dt * time_step

  _, dmomentum_dt = state_derivatives_fn(next_position, aux_momentum)
  next_momentum = aux_momentum + dmomentum_dt * time_step / 2

  return next_position, next_momentum


# Single graph -> graph integration step.
IntegratorType = Callable[
    [np.ndarray, np.ndarray, float, StateDerivativesFnType],
    Tuple[np.ndarray, np.ndarray]
]


def single_integration_step(
    graph: jraph.GraphsTuple, time_step: float,
    integrator_fn: IntegratorType,
    hamiltonian_from_graph_fn: Callable[[jraph.GraphsTuple], jraph.GraphsTuple],
    ) -> Tuple[float, jraph.GraphsTuple]:
  """Updates a graph state integrating by a single step.
  Args:
    graph: `GraphsTuple` representing a system state at time t.
    time_step: size of the timestep to integrate for.
    integrator_fn: Integrator to use. A function fn such that
       fn(position_t, momentum_t, time_step, state_derivatives_fn) ->
           (position_tp1, momentum_tp1)
    hamiltonian_from_graph_fn: Function that given a `GraphsTuple`, returns
        another one with a "hamiltonian" global field.
  Returns:
    `GraphsTuple` representing a system state at time `t + time_step`.
  """

  # Template graph with particle/interactions parameters and connectiviity
  # but without the state (position/momentum).
  static_graph = get_static_graph(graph)

  # Get the Hamiltonian function, and the function that returns the state
  # derivatives.
  hamiltonian_fn = get_hamiltonian_from_state_fn(
      static_graph=static_graph,
      hamiltonian_from_graph_fn=hamiltonian_from_graph_fn)
  state_derivatives_fn = get_state_derivatives_from_hamiltonian_fn(
      hamiltonian_fn)

  # Get the current state.
  position, momentum = get_system_state(graph)

  # Calling the integrator to get the next state.
  next_position, next_momentum = integrator_fn(
      position, momentum, time_step, state_derivatives_fn)
  next_graph = set_system_state(static_graph, next_position, next_momentum)

  # Return the energy of the next state too for plotting.
  energy = hamiltonian_fn(next_position, next_momentum)

  return energy, next_graph


if __name__ == '__main__':
    # Get a state function and jit it.
    # We could switch to any other Hamiltonian and any other integrator here.
    # e.g. the non-symplectic `euler_integrator`.
    step_fn = functools.partial(
        single_integration_step,
        hamiltonian_from_graph_fn=hookes_hamiltonian_from_graph_fn,
        integrator_fn=verlet_integrator)
    step_fn = jax.jit(step_fn)

    # Get a graph with the initial state.
    num_particles = 10
    graph = build_hookes_particle_state_graph(num_particles)

    # Iterate for multiple timesteps.
    num_steps = 200
    time_step = 0.002
    positions_sequence = []
    total_energies = []
    steps = []
    for step_i in range(num_steps):
      energy, graph = step_fn(graph, time_step)
      total_energies.append(energy)
      positions_sequence.append(graph.nodes["position"])
      steps.append(step_i + 1)

    # Plot results (positions and energy as a function of time).
    unused_fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    positions_sequence_array = np.stack(positions_sequence, axis=0)
    axes[0].plot(positions_sequence_array[..., 0],
                 positions_sequence_array[..., -1])
    axes[0].set_xlabel("Particle position x")
    axes[0].set_ylabel("Particle position y")

    axes[1].plot(steps, total_energies)
    axes[1].set_ylim(0, max(total_energies)*1.2)
    axes[1].set_xlabel("Simulation step")
    axes[1].set_ylabel("Total energy")
    plt.show()