import functools
import jax
import jax.numpy as jnp
import haiku as hk
import optax
import numpy as np

class SelfAttention(hk.MultiHeadAttention):
    """
    Self attention with causal mask applied.
    """
    def __call__(self, query : jnp.ndarray,
                    key : Optional[jnp.ndarray] = None,
                    value : Optional[jnp.ndarray] = None,
                    mask : Optional[jnp.ndarray] = None) -> jnp.ndarray:
        if key is None:
            key = query
        if value is None:
            value = query

        seq_len = query.shape[1]
        causal_mask = np.tril(np.ones((seq_len, seq_len)))
        if mask is None:
            mask = causal_mask
        else:
            mask = mask * causal_mask

        return super.__call__(query, key, value, mask)


class DenseBlock(hk.Module):
    """
    A 2-layer MLP
    """
    def __init__(self, init_scale: float,
                    widening_factor: int = 4,
                    name: Optional[str] = None):
        super().__init__(name = name)
        self._init_scale = init_scale
        self._widening_factor = widening_factor

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        hiddens = x.shape[-1]
        initializer = hk.initializers.VarianceScaling(self._init_scale)
        x = hk.Linear(self._widening_factor * hiddens, w_init = initializer)(x)
        x = jax.nn.gelu(x)
        return hk.Linear(hiddens, w_init=initializer)(x)

def layer_norm(x: jnp.ndarray, 
                name : Optional[str] = None) -> jnp.ndarray:
    """
    Apply a unique LayerNorm to x with default settings
    """
    return hk.LayerNorm(axis = -1,
                        create_scale = True, 
                        create_offset = True,
                        name = name)(x)

class Transformer(hk.Module):
    """
    A transformer stack.
    """
    def __init__(self, 
                num_heads: int,
                num_layers: int,
                dropout_rate: float,
                name: Optional[str] = None):
        super(Transformer, self).__init__(name = name)
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._droput_rate = dropout_rate

    def __call__(self, 
                h: jnp.ndarray,
                mask: Optional[jnp.ndarray],
                is_training: bool) -> jnp.ndarray:
        """
        Args:
            h: Inputs, [B, T, H].
            mask: Padding mask, [B, T].
            is_training: Whether we're training or not
        Returns:
            Array of shape [B, T, H].
        """

        init_scale = 2. / self._num_layers
        if is_training:
            dropout_rate = self._droput_rate
        else:
            dropout_rate = 0.0

        if mask is not None:
            mask = mask[:, None, None, :]

        for i in range(self._num_layers):
            h_norm = layer_norm(h, name = 'h{}_ln_1'.format(i))
            h_attn = SelfAttention(num_heads=self._num_heads, # Defining layers in loop
                                    key_size = 64,
                                    w_init_scale = init_scale,
                                    name = 'h{}_attn'.format(i))(h_norm, mask = mask)
            h_attn = hk.dropout(hk.next_rng_key(), dropout_rate, h_attn)
            h = h + h_attn
            h_norm = layer_norm(h, name = 'h{}_ln_2'.format(i))
            h_dense = DenseBlock(init_scale, name = 'h{}_mlp'.format(i))(h_norm)
            h_dense = hk.dropout(hk.next_rng_key(), dropout_rate, h_dense)
            h = h + h_dense

        h = layer_norm(h, name='ln_f')

        return h

def embeddings(data: Mapping[str, jnp.ndarray],
                vocab_size: int,
                d_model: int):
    tokens = data['observations']
    input_mask = jnp.greater(tokens, 0)
    seq_lenght = tokens.shape[1]

    # Embed the input tokens and positions
    embed_init = hk.initializers.TruncatedNormal(stddev = 0.02)
    token_embedding_map = hk.Embed(vocab_size, d_model, w_init=embed_init)
    token_embs = token_embedding_map(tokens)
    positional_embeddings = hk.get_parameter(
        'pos_embs', [seq_lenght, d_model], init = embed_init)
    input_embeddings = token_embs + positional_embeddings
    return input_embeddings, input_mask


### ---------------------------------- ###
### ---------- Forward pass ---------- ###
### ---------------------------------- ###

def build_forward_fn(vocab_size: int, d_model: int, num_heads: int,
                    num_layers: int, dropout_rate: float):
    """
    create the model's forward pass.
    """
    def forward_fn(data: Mapping[str, jnp.ndarray],
                is_training: bool = True) -> jnp.ndarray:
        """
        Forward pass.
        """
        input_embeddings, input_mask = embeddings(data, vocab_size, d_model)

        transformer = Transformer(num_heads = num_heads,
                                num_layers = num_layers,
                                dropout_rate = dropout_rate)
                                
        output_embeddings = transformer(input_embeddings, 
                                        input_mask, 
                                        is_training)

        return hk.Linear(vocab_size)(output_embeddings)

    return forward_fn

# forward_fn = build_forward_fn(vocab_size, d_model, num_heads, num_layers, dropout_rate)
# forward_fn = hk.transform(forward_fn)


### ----------------------------- ###
### ---------- Trainig ---------- ###
### ----------------------------- ###

def lm_loss_fn(forward_fn, # transformed forward pass.
            vocab_size: int,
            params,
            rng,
            data: Mapping[str, jnp.ndarray],
            is_training: bool = True) -> jnp.ndarray:
    """
    Compute the loss on data wrt params.
    """
    logits = forward_fn(params, rng, data, is_training)
    targets = jax.nn.one_hot(data['target'], vocab_size)
    assert logits.shape == targets.shape

    mask = jnp.greater(data['observations'], 0)
    loss = -jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1)
    loss = jnp.sum(loss * mask) / jnp.sum(mask) # loss corrected by mask

    return loss

# Optax:
# state = init(params)
# grads, state = update(grads, state, params=None)

class GradientUpdater:
    """
    A stateless abstraction around an init_fn/update_fn pair.
    This extracts some common boilerplate from the training loop.

    net_init is forward_fn.init
    """
    def __init__(self, net_init, loss_fn,
                optimizer: optax.GradientTransformation):
        self._net_init = net_init
        self._loss_fn = loss_fn
        self._opt = optimizer

    # Decorator is equivalent to:
    # partial_jit = functools.partial(jax.jit, static_argnums = 0)
    # @partial_jit
    @functools.partial(jax.jit, static_argnums = 0)
    def init(self, master_rng, data):
        """
        Initializes state of the updater.
        """
        out_rng, init_rng = jax.random.split(master_rng)
        params = self._net_init(init_rng, data)
        opt_state = self._opt.init(params)
        state = {'step' : np.array(0),
                'rng' : out_rng,
                'opt_state' : opt_state,
                'params' : params}
        return state

    @functools.partial(jax.jit, static_argnums = 0)
    def update(self, state: Mapping[str, Any],
                data : Mapping[str, jnp.ndarray]):
        """
        Updates the state using some data and returns metrics.
        """
        rng, new_rng = jax.random.split(state['rng'])
        params = state['params']
        loss, g = jax.value_and_grad(self._loss_fn)(params, rng, data)

        updates, opt_state = self._opt.update(g, state['opt_state'])
        params = optax.apply_updates(params, updates)

        new_state = {'step' : state['step'] + 1,
                     'rng' : new_rng,
                     'opt_state' : opt_state,
                     'params' : params}
        
        metrics = {'step': state['step'],
                    'loss': loss}

        return new_state, metrics

def mian():
    # Create the dataset
    train_dataset, vocab_size = load(batch_size, sequence_length)

    # Set up the model, loss, and updater
    forward_fn = build_forward_fn(vocab_size, d_model, num_heads, num_layers, dropout_rate)
    forward_fn = hk.transform(forward_fn)

    loss_fn = functools.partial(lm_loss_fn, forward_fn.apply, vocab_size)

    optimizer = optax.chain(optax.clip_by_global_norm(grad_clip_value),
                            optax.adam(learning_rate, b1=0.9, b2=0.99))

    updater = GradientUpdater(forward_fn.init, loss_fn, optimizer)

    # Initialize paramters:
    rng = jax.random.PRNGKey(428)
    data = next(train_dataset)
    state = updater.init(rng, data)

    # Training loop:
    for step in range(N_STEPS):
        data = next(train_dataset)
        state, metrics = updater.update(state, data)