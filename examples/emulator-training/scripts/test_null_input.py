import numpy as np
from math import isclose


import tensorflow as tf
from tensorflow.keras import losses, optimizers, layers, Input, Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.layers import Dense,TimeDistributed

batch_size = 10
units = 4

null_input = Input(shape=(0), batch_size=batch_size, name="null_input") 

probs = Dense(units=units,activation='softmax',bias_initializer='random_normal')(null_input)

#mu = Dense(units=1,activation='sigmoid',bias_initializer='random_normal')(null_input)
mu = Dense(units=1,activation='sigmoid',bias_initializer=tf.constant_initializer(0.5))(null_input)

model = Model(inputs=[null_input,], outputs=[probs,mu])

a = tf.zeros([batch_size, 0])

output = model(a)

print(f"Output = {output}")
