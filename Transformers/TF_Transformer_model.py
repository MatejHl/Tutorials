import tensorflow as tf

class PositionalEncoding:
    """
    PE_(pos, 2i) = sin(\frac{pos, 10000^{2i/d_model}})
    PE_(pos, 2i+1) = cos(\frac{pos, 10000^{2i/d_model}})
    """
    def __init__(self, d_model):
        self.d_model = d_model
    
    def get_angles(self, pos, _2i):
        d_model = tf.cast(self.d_model, tf.float32)
        powers = tf.math.pow(10000, _2i / d_model)
        angle_rates = tf.math.reciprocal(powers)
        return tf.matmul(pos, angle_rates)

    def __call__(self, num_positions):
        angle_rads = self.get_angles(tf.expand_dims(tf.range(num_positions, dtype=tf.float32), -1),
                                    tf.expand_dims(tf.range(self.d_model, delta = 2, dtype=tf.float32), 0))

        # apply sin to 2i:
        _sin = tf.math.sin(angle_rads)
        _cos = tf.math.cos(angle_rads)

        PE = tf.reshape(tf.concat([tf.expand_dims(_sin, -1), tf.expand_dims(_cos, -1)], axis=-1), 
                        shape = (num_positions,-1)) # Interleave

        PE = tf.expand_dims(PE, 0) # Input is rank 3 tensor
        return PE

def scaled_dot_product_attention(Q, K, V, mask = None):
    """
    Claculate the attention weights

    Paramters:
    ----------
    Q : tf.Tensor, dtype = tf.float32
        queries with shape = (..., seq_len_q, depth) 

    K : tf.Tensor, dtype = tf.float32
        keys with shape = (..., seq_len_k, depth)

    V : tf.Tensor, dtype = tf.float32
        values with shape = (..., seq_len_k, depth_v)

    mask : tf.Tensor, dtype = tf.bool, optional (default = None)
        mask with possibly different shapes, depending on its type 
        (padding or look ahead) but it must be broadcastable to
        (..., seq_len_q, seq_len_k) for addition.

    Returns:
    --------
    output, attention_weights

    Notes:
    ------
    Q, K, V must have matching leadning dimensions.
    K, V must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v
    The mask have different 
    """
    matmul_qk = tf.matmul(Q, K, transpose_b = True)

    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # normalized on the last axis (seq_len_k)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, V)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    For more sophisticated usecase see tf.keras.layers.MultiHeadAttention
    https://www.tensorflow.org/api_docs/python/tf/keras/layers/MultiHeadAttention
    https://github.com/tensorflow/tensorflow/blob/v2.4.0/tensorflow/python/keras/layers/multi_head_attention.py#L126-L479
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % num_heads == 0

        self.depth = d_model // num_heads

        self.W_Q = tf.keras.layers.Dense(d_model)
        self.W_K = tf.keras.layers.Dense(d_model)
        self.W_V = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def _split_heads(self, x, batch_size):
        """
        Split the last dimesnion into (num_heads, depth).
        Transpose the result such that the shape is 
        (batch_size, num_heads, seq_len, depth).
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        batch_size = tf.shape(Q)[0]

        Q = self.W_Q(Q) # (batch_size, seq_len, d_model)
        K = self.W_K(K) # (batch_size, seq_len, d_model)
        V = self.W_V(V) # (batch_size, seq_len, d_model)

        Q = self._split_heads(Q, batch_size) # (batch_size, num_heads, seq_len_q, depth)
        K = self._split_heads(K, batch_size) # (batch_size, num_heads, seq_len_k, depth)
        V = self._split_heads(V, batch_size) # (batch_size, num_heads, seq_len_v, depth)

        scaled_attention, attention_weights = scaled_dot_product_attention(Q, K, V, mask)

        scaled_attention = tf.transpose(scaled_attention, perm = [0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention) # (batch_size, seq_len_q, d_model)

        return output, attention_weights

# Ponit-wise feed forward network
def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([tf.keras.layers.Dense(dff, activation='relu'), # (batch_size, seq_len, dff)
                            tf.keras.layers.Dense(d_model, activation=None)]) # (batch_size, seq_len, d_model)


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, drate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.dropout_mha = tf.keras.layers.Dropout(drate)
        self.layernorm_mha = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.dropout_ffn = tf.keras.layers.Dropout(drate)
        self.layernorm_ffn = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training, mask):

        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout_mha(attn_output, training = training)
        out1 = self.layernorm_mha(x + attn_output) # Residual

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout_ffn(ffn_output, training = training)
        out2 = self.layernorm_ffn(out1 + ffn_output) # Residual

        return out2

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, drate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drate)
        self.dropout2 = tf.keras.layers.Dropout(drate)
        self.dropout3 = tf.keras.layers.Dropout(drate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training = training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(out1, enc_output, enc_output, padding_mask)
        attn2 = self.dropout2(attn2, training = training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        fnn_output = self.dropout3(ffn_output, training = training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2


def create_padding_mask(seq):
    # Ones in position where there is padding. (=> ones in places NOT to use).
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :] # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    # Indicates what should NOT be used.
    return 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, max_position_encoding, drate = 0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # Text embedding:
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)(max_position_encoding)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, drate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(drate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32)) # sqrt(d) instead of 1
        x += self.pos_encoding[:, :seq_len, :]

        # Random input? 
        x = self.dropout(x, training = training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training = training, mask = mask)
        
        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, max_position_encoding, drate = 0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # Text embedding:
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)(max_position_encoding)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, drate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(drate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training = training)

        for i in range(self.num_layers):
            x, attn_weights_block1, attn_weights_block2 = self.dec_layers[i](x, enc_output, training, 
                                                                        look_ahead_mask, padding_mask)
            attention_weights['decoder_layer{}_block1'.format(i+1)] = attn_weights_block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = attn_weights_block2

        return x, attention_weights

        
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size,
                max_pe_input, max_pe_target, drate = 0.1):
        super(Transformer, self).__init__(name = "Transformer")

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, max_pe_input, drate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, max_pe_target, drate = 0.1)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size) # Activation = softmax

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)

        return final_output, attention_weights



if __name__ == '__main__':
    sample_transformer = Transformer(
        num_layers=2, d_model=512, num_heads=8, dff=2048, 
        input_vocab_size=8500, target_vocab_size=8000, 
        max_pe_input=10000, max_pe_target=6000)

    temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
    temp_target = tf.random.uniform((64, 26), dtype=tf.int64, minval=0, maxval=200)

    fn_out, _ = sample_transformer(temp_input, temp_target, training=False, 
                                   enc_padding_mask=None, 
                                   look_ahead_mask=None,
                                   dec_padding_mask=None)

    print(fn_out[0, 0, :])  # (batch_size, tar_seq_len, target_vocab_size)

