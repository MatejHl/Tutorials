import jax
from jax import lax, numpy as jnp
from typing import Any, Callable, Sequence, Optional
import flax
from flax.core import freeze, unfreeze
import optax

# linear regression:
model = flax.linen.Dense(features = 5)

key = jax.random.PRNGKey(0)
key1, key2 = jax.random.split(key)

x = jax.random.normal(key1, (10,))
output, params = model.init_with_output(key2, x)
print(output)
print(jax.tree_map(lambda x: x.shape, params))

output = model.apply(params, x)
print(output)

# Params are FrozenDict which is immutable.
try:
    params['new_key'] = jnp.ones((2,2))
except ValueError as e:
    print('Error: \n\t{}'.format(e))

# --------------------------
# --- Linear regression: ---
# --------------------------
nsamples = 20
xdim = 10
ydim = 5

# True process:
key = jax.random.PRNGKey(0)
k1, k2 = jax.random.split(key)
W = jax.random.normal(k1, (xdim, ydim))
b = jax.random.normal(k2, (ydim, ))
true_params = freeze({'params': {'bias': b, 'kernel': W}})

# Samples:
ksample, knoise = jax.random.split(k1)
x_samples = jax.random.normal(ksample, (nsamples, xdim))
y_samples = jnp.dot(x_samples, W) + b
y_samples += 0.1*jax.random.normal(knoise, (nsamples, ydim)) # Adding noise
print('x shape: ', x_samples.shape, '; y shape: ', y_samples.shape)


# Loss function:
def make_mse_func(x_batched, y_batched):
    def mse(params):
        # Define the squared loss for a single pair (x, y)
        def squared_error(x, y):
            pred = model.apply(params, x)
            return jnp.inner(y-pred, y-pred)/2.0
        # Vectorize:
        batch_squared_error = jax.vmap(squared_error)
        return jnp.mean(batch_squared_error(x_batched, y_batched), axis = 0)
    return jax.jit(mse)

loss = make_mse_func(x_samples, y_samples)


alpha = 0.3
print('Loss for "true" W, b: ', loss(true_params))
grad_fn = jax.value_and_grad(loss)

for i in range(101):
    loss_val, grads = grad_fn(params)
    # Apply gradient decent on all parameters (=> multimap on params which 
    # is set of all parameters)
    params = jax.tree_multimap(lambda p, g: p - alpha * g, params, grads)
    if i % 10 == 0:
        print('Loss step {} '.format(i), loss_val)


# Using OPTAX:
print('Optax ----------')
params = model.init(key2, x)

optimizer = optax.sgd(learning_rate=alpha)
opt_state = optimizer.init(params)
loss_grad_fn = jax.value_and_grad(loss)

for i in range(101):
    loss_val, grads = loss_grad_fn(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    if i % 10 == 0:
        print('Loss step {}'.format(i), loss_val)


# Serialization:
print('Serialization ----------')
bytes_output = flax.serialization.to_bytes(params)
dict_output = flax.serialization.to_state_dict(params)
print('Dict output')
print(dict_output)
print('Bytes output')
print(bytes_output)

_params = model.init(key2, x)
_params = flax.serialization.from_bytes(_params, bytes_output)

print(jax.tree_multimap(lambda x, y: x - y, params, _params))

# --------------------------
# --- Defining own model ---
# --------------------------
class ExplicitMLP(flax.linen.Module):
    features: Sequence[int]

    def setup(self):
        """
        A setup() method that is being called at the end of the 
        __postinit__ where you can register submodules, variables, 
        parameters you will need in your model.
        """
        self.layers = [flax.linen.Dense(feat) for feat in self.features]
        self.out_layer = flax.linen.Dense(5)

    def __call__(self, inputs):
        x = inputs
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = flax.linen.relu(x)
        x = self.out_layer(x)
        return x

key1, key2 = jax.random.split(jax.random.PRNGKey(0), 2)
x = jax.random.uniform(key1, (4,4))

model = ExplicitMLP(features = [3,4,5])
params = model.init(key2, x)
y = model.apply(params, x)

print('initialized paramter shapes:\n', jax.tree_map(jnp.shape, unfreeze(params)))
print('initialized paramter shapes:\n', jax.tree_map(jnp.shape, params))
print('output:\n', y)

# binded_model = model.bind(params)
# print(binded_model.layers[0](x))

# ------------------------------------
# --- simple "Batch" normalization ---
# ------------------------------------
class BiasAdderWithRunningMean(flax.linen.Module):
    decay: float = 0.99

    def setup(self):
        self.ra_mean = self.variable('batch_stats', 
                                'mean', # variable(col, name, init_fn, *init_args) 
                                lambda s: jnp.zeros(s),
                                x.shape[1:])
        self.bias = self.param('bias', lambda rng, shape: jnp.zeros(shape), x.shape[1:])
    
    def __call__(self, x):
        # Easy pattern to detect if we're initializating via empty variable tree
        is_initialized = self.has_variable('batch_stats', 'mean')
        if is_initialized:
            self.ra_mean.value = self.decay * self.ra_mean.value + (1.0 - self.decay) * jnp.mean(x, axis=0, keepdims=True)

        return x - self.ra_mean.value + self.bias

key1, key2 = jax.random.split(jax.random.PRNGKey(0), 2)
x = jnp.ones((10, 5))
model = BiasAdderWithRunningMean()
y, variables = model.init_with_output(key1, x)
print('initialized variables:\n', variables)
y, updated_state = model.apply(variables, x, mutable = ['batch_stats'])
print('updated state:\n', updated_state)

for val in [1.0, 2.0, 3.0]:
    x = val * jnp.ones((10,5))
    y, updated_state = model.apply(variables, x, mutable=['batch_stats'])
    old_state, params = variables.pop('params')
    variables = freeze({'params': params, **updated_state})
    print('---------- variables:\n', variables) # Shows only the mutable part