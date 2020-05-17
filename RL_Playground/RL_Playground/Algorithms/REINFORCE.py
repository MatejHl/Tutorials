
import tensorflow as tf
import numpy as np
"""
sparse_categorical_crossentropy is used, because it's derivation is 
1*\frac{\partial log(\pi(s, a, \theta)}{\partial \theta}) for A_t = a chosen and 0 for other actions.
"""

def discount_rewards(r, gamma):
    G = np.zeros_like(r)
    G_t = 0.0
    for t in reversed(range(0, r.size)):
        G_t = r[t] + gamma*G_t
        G[t] = G_t
    return G

def REINFORCE(model, env, gamma, optimizer, n_epochs, update_mod = 100):
    avg_score = 0.0
    actions = np.arange(env.action_space.n)
    gradBuffer = {}
    scores = []
    for epoch in range(n_epochs):
        done = False
        s_t = env.reset()
        memory = []
        score = 0
        while not done:
        
            with tf.GradientTape() as tape:
                pi_logits = model(np.expand_dims(s_t, axis=0))
                a_pdf = pi_logits.numpy()
                # Choose random action with p = action dist
                a_t = np.random.choice(actions,p=a_pdf[0])
                loss = tf.keras.losses.sparse_categorical_crossentropy(y_true = a_t, 
                                                                   y_pred = pi_logits, 
                                                                   from_logits=False, axis=-1)
            grads_t = tape.gradient(loss, model.trainable_variables)
            s_t, r_t, done, _ = env.step(a_t)
            memory.append([grads_t, r_t])  # Maybe find better way for numerical reasons.
            score += r_t

        
        avg_score += score/update_mod
        scores.append(score)
        memory = np.array(memory)
        memory[:,1] = discount_rewards(memory[:,1], gamma)
        if not gradBuffer:
            for grads_t, G_t in memory:
                for ix, grad in enumerate(grads_t):
                    gradBuffer[ix] = tf.math.scalar_mul(G_t, grad)
        else:
            for grads_t, G_t in memory:
                for ix, grad in enumerate(grads_t):
                    gradBuffer[ix] += tf.math.scalar_mul(G_t, grad)
         
        if epoch % update_mod == 0:
            optimizer.apply_gradients(zip(gradBuffer.values(), model.trainable_variables))

            print('epoch: {}    avg_score: {}'.format(epoch, avg_score))
            avg_score = 0.0
            for ix in gradBuffer.keys():
                gradBuffer[ix] = tf.math.scalar_mul(tf.constant(0.0), gradBuffer.get(ix))
