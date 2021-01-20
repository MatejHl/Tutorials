"""
Tutorial:    https://www.tensorflow.org/tutorials/text/transformer
"""

import tensorflow as tf

import time
import numpy as np
import matplotlib.pyplot as plt


# load data: ----------
# ---------- ----------
# import tensorflow_datasets as tfds
# examples, metadata = tfds.load("ted_hrlr_translate/pt_to_en", 
#                                 with_info = True,
#                                 as_supervised = True)
# train_examples, val_examples = examples['train'], examples['validation']
# tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
#     (en.numpy() for pt, en in train_examples), 
#     target_vocab_size=2**13)
# tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
#     (pt.numpy() for pt, en in train_examples), 
#     target_vocab_size=2**13)
# ---------- ----------

# Look at data: -------------
# ------------- -------------
# sample_string = "Transformer is awesome"
# tokenized_string = tokenizer_en.encode(sample_string)
# print("Tokenized string is {}".format(tokenized_string))
# original_string = tokenizer_en.decode(tokenized_string)
# print("The original string: {}".format(original_string))
# assert original_string == tokenized_string
# 
# for ts in tokenized_string:
#     print("{} ------> {}".format(ts, tokenizer_en.decode([ts])))
# ------------- -------------

dummy_src = [1,2,3]
dummy_tgt = [3,2,1,0]

