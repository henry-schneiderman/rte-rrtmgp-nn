import numpy as np
from math import isclose


import tensorflow as tf
from tensorflow.keras import losses, optimizers, layers, Input, Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.layers import Dense,TimeDistributed

class Cell(layers.Layer):
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)
        self.state_size = (1,)
        self.output_size = (1,)

    def call(self, input_at_t, states_at_t):
        output_at_t = input_at_t
        state_at_t_plus_1 = states_at_t + input_at_t
        return output_at_t, state_at_t_plus_1
    
def train():
    n_layers = 4
    batch_size = 5
    input = Input(shape=(n_layers, 1), batch_size=batch_size, name="input")
    upward_output, upward_state = tf.keras.layers.RNN(Cell, return_sequences=True, return_state=True, go_backwards=False)(input)
    model = Model(inputs=[input,], outputs=[upward_output, upward_state])

    tmp = tf.range(n_layers)
    print ("input = " + tmp)
    print ("output = " + model(tmp))

if __name__ == "__main__":
    train()
