# Ref: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data

import tensorflow as tf

import os
import tempfile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ------------------ Plots --------------------
mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def plot_metrics(history):
  metrics =  ['loss', 'auc', 'precision', 'recall']
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,2,n+1)
    plt.plot(history.epoch,  history.history[metric], color=colors[0], label='Train')
    plt.plot(history.epoch, history.history['val_'+metric],
             color=colors[0], linestyle="--", label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    elif metric == 'auc':
      plt.ylim([0.8,1])
    else:
      plt.ylim([0,1])

    plt.legend()


def plot_loss(history, label, n):
  # Use a log scale to show the wide range of values.
  plt.semilogy(history.epoch,  history.history['loss'],
               color=colors[n], label='Train '+label)
  plt.semilogy(history.epoch,  history.history['val_loss'],
          color=colors[n], label='Val '+label,
          linestyle="--")
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  
  plt.legend()
# ------------------ ----- --------------------


file = tf.keras.utils
raw_df = pd.read_csv('https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv')
# print(raw_df.head())
print(raw_df[['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V26', 'V27', 'V28', 'Amount', 'Class']].describe())

neg, pos = np.bincount(raw_df['Class'])
total = neg + pos
print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    total, pos, 100 * pos / total))

# Preprocessing:
cleaned_df = raw_df.copy()
cleaned_df.pop('Time')
# The 'Amount' column covers a huge range. Convert to log-space.
eps=0.001 # 0 => 0.1
cleaned_df['Log Ammount'] = np.log(cleaned_df.pop('Amount')+eps)


# Use a utility from sklearn to split and shuffle our dataset.
train_df, test_df = train_test_split(cleaned_df, test_size=0.2)
train_df, val_df = train_test_split(train_df, test_size=0.2)

# Form np arrays of labels and features.
train_labels = np.array(train_df.pop('Class'))
bool_train_labels = train_labels != 0
val_labels = np.array(val_df.pop('Class'))
test_labels = np.array(test_df.pop('Class'))

train_features = np.array(train_df)
val_features = np.array(val_df)
test_features = np.array(test_df)


scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)

val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)

train_features = np.clip(train_features, -5, 5)
val_features = np.clip(val_features, -5, 5)
test_features = np.clip(test_features, -5, 5)


print('Training labels shape:', train_labels.shape)
print('Validation labels shape:', val_labels.shape)
print('Test labels shape:', test_labels.shape)

print('Training features shape:', train_features.shape)
print('Validation features shape:', val_features.shape)
print('Test features shape:', test_features.shape)


pos_df = pd.DataFrame(train_features[ bool_train_labels], columns = train_df.columns)
neg_df = pd.DataFrame(train_features[~bool_train_labels], columns = train_df.columns)

# sns.jointplot(pos_df['V5'], pos_df['V6'],
#               kind='hex', xlim = (-5,5), ylim = (-5,5))
# plt.suptitle("Positive distribution")
# 
# sns.jointplot(neg_df['V5'], neg_df['V6'],
#               kind='hex', xlim = (-5,5), ylim = (-5,5))
# _ = plt.suptitle("Negative distribution")
# 
# plt.show()


METRICS = [tf.keras.metrics.TruePositives(name='tp'),
           tf.keras.metrics.FalsePositives(name='fp'),
           tf.keras.metrics.TrueNegatives(name='tn'),
           tf.keras.metrics.FalseNegatives(name='fn'), 
           tf.keras.metrics.BinaryAccuracy(name='accuracy'),
           tf.keras.metrics.Precision(name='precision'),
           tf.keras.metrics.Recall(name='recall'),
           tf.keras.metrics.AUC(name='auc')]

def make_model(metrics = METRICS, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, 
                              activation='relu',
                              input_shape=(train_features.shape[-1],)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, 
                              activation='sigmoid',
                              bias_initializer=output_bias)])
    
    model.compile(optimizer = tf.keras.optimizers.Adam(lr=1e-3),
                  loss = tf.keras.losses.BinaryCrossentropy(),
                  metrics = metrics)
    return model


EPOCHS = 100
BATCH_SIZE = 2048

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_auc', 
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)


# -------------------- Init bias --------------
# Init loss without bias init
model = make_model()
results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)
print("Loss without init: {:0.4f}".format(results[0]))

# Set initial bias
initial_bias = np.log([pos/neg])
model = make_model(output_bias = initial_bias)
print(model.predict(train_features[:10]))
# Init loss with bias init
results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)
print("Loss with init: {:0.4f}".format(results[0]))
# Save initial weights:
initial_weights = os.path.join(tempfile.mkdtemp(),'initial_weights')
model.save_weights(initial_weights)


if False: # See the difference:
    model = make_model()
    model.load_weights(initial_weights)
    model.layers[-1].bias.assign([0.0]) # init as 0.0
    zero_bias_history = model.fit(
        train_features,
        train_labels,
        batch_size=BATCH_SIZE,
        epochs=20,
        validation_data=(val_features, val_labels), 
        verbose=0)
    
    model = make_model()
    model.load_weights(initial_weights)
    careful_bias_history = model.fit(
        train_features,
        train_labels,
        batch_size=BATCH_SIZE,
        epochs=20,
        validation_data=(val_features, val_labels), 
        verbose=0)

    plot_loss(zero_bias_history, "Zero Bias", 0)
    plot_loss(careful_bias_history, "Careful Bias", 1)
    plt.show()


if True:
    model = make_model()
    model.summary()
    # test run the model:
    print(model.predict(train_features[:10]))
    
    model = make_model()
    model.load_weights(initial_weights)
    baseline_history = model.fit(
        train_features,
        train_labels,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks = [early_stopping],
        validation_data=(val_features, val_labels))

    # plot_metrics(baseline_history)
# -------------------- --------- --------------


# ------------------ Class weights ------------
if True:
    # Scaling by total/2 helps keep the loss to a similar magnitude.
    # The sum of the weights of all examples stays the same.
    weight_for_0 = (1 / neg)*(total)/2.0 
    weight_for_1 = (1 / pos)*(total)/2.0
    
    class_weight = {0: weight_for_0, 1: weight_for_1}
    
    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))
    
    
    weighted_model = make_model()
    weighted_model.load_weights(initial_weights)
    
    weighted_history = weighted_model.fit(
        train_features,
        train_labels,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks = [early_stopping],
        validation_data=(val_features, val_labels),
        # The class weights go here
        class_weight=class_weight)
    
    # plot_metrics(weighted_history)
# ------------------ ------------- ------------

plot_metrics(baseline_history)
plt.show()
plot_metrics(weighted_history)
plt.show()


# ------------------ Oversampling -------------
# Tried in work project ReadingPoz (probably private repo)
# based on: https://www.kaggle.com/yihdarshieh/tutorial-oversample

