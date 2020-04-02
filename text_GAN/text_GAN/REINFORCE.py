import tensorflow as tf
import numpy as np

from text_GAN_model import tf_MC_rollouts


def _tf_discout_rewards_while_body(t, r_TA,   gamma):
        r_TA = r_TA.write(t, tf.math.add(r_TA.read(t),  tf.math.scalar_mul(gamma, r_TA.read(t+1))))
        return (t - 1, r_TA)


def _REINFORCE_while_body(index, a_t, s_t, r_TA, REINFORCE_loss_TA,   num_trajectories, Generator_model, Discriminator_model, end_char_id, padding_shape):
    """
    a_t is needed for the condition
    """
    predictions = Generator_model(tf.expand_dims(s_t, 0))
    a_t = tf.random.categorical(predictions, num_samples=1, dtype = tf.int32)
    REINFORCE_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true = a_t, y_pred = predictions, from_logits=True)
    REINFORCE_loss_TA = REINFORCE_loss_TA.write(index, REINFORCE_loss)

    a_t = tf.reshape(a_t, shape = (1,))
    s_t = tf.concat([s_t, a_t], axis = 0, name = None)
    
    trajectories = tf_MC_rollouts(s_t, num_trajectories, Generator_model, end_char_id, maximum_iterations = padding_shape[1])
    trjs = tf.ragged.stack(trajectories)
    padded_trajectories = trjs.to_tensor(default_value=0, name=None, shape = padding_shape)
    r_t = tf.math.reduce_mean(Discriminator_model(padded_trajectories), axis=None, keepdims=False, name=None)
    r_TA = r_TA.write(index, r_t)

    return (index + 1, a_t, s_t, r_TA, REINFORCE_loss_TA)

@tf.function
def tf_REINFORCE_step(Generator_model, Discriminator_model, gamma, optimizer, num_trajectories, begin_char_id , end_char_id, padding_shape):
    """
    """
    index = 0
    a_t = tf.constant([-1], name = 'a_t')
    s_t = begin_char_id
    r_TA = tf.TensorArray(tf.float32, size=0, dynamic_size=True, name = 'r_TA')
    REINFORCE_loss_TA = tf.TensorArray(tf.float32, size=0, dynamic_size=True, name = 'REINFORCE_loss_TA')

    Generator_model.reset_states()
    Discriminator_model.reset_states()
    index, a_t, s_t, r_TA, REINFORCE_loss_TA = tf.while_loop(cond = lambda index, a_t, s_t, r_TA, REINFORCE_loss_TA: tf.math.not_equal(a_t,  
                                                                                                                                       end_char_id), 
                  body =  lambda index, a_t, s_t, r_TA, REINFORCE_loss_TA: _REINFORCE_while_body(index, a_t, s_t, r_TA, REINFORCE_loss_TA,   
                                                                                                 num_trajectories, Generator_model, Discriminator_model, end_char_id, padding_shape), 
                  loop_vars = [index, a_t, s_t, r_TA, REINFORCE_loss_TA],
                  shape_invariants=[tf.TensorShape([]), a_t.get_shape(), tf.TensorShape([None]), None, None],
                  name = 'while_tf_REINFORCE_step',
                  maximum_iterations = padding_shape[1])
    
    t = r_TA.size() - 1
    t, r_TA = tf.while_loop(cond = lambda t, r_TA: tf.math.less(t, 0),
                            body =  lambda t, r_TA: _tf_discout_rewards_while_body(t, r_TA, gamma),
              loop_vars = [t, r_TA],
              shape_invariants=[tf.TensorShape([]), None],
              name = 'while_tf_discout_rewards')

    G_mul_REINFORCE_loss = tf.math.multiply(r_TA.stack(), REINFORCE_loss_TA.stack())
    grads = tf.gradients(G_mul_REINFORCE_loss, Generator_model.trainable_variables)

    optimizer.apply_gradients(zip(grads, Generator_model.trainable_variables))

    return None