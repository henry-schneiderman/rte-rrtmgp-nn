import numpy as np
from math import isclose


import tensorflow as tf
from tensorflow.keras import losses, optimizers, layers, Input, Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.layers import Dense,TimeDistributed


class Cell(layers.Layer):
    def __init__(self, batch_size, **kwargs):
        super().__init__(**kwargs)
        #self.state_size = (batch_size, 2,4,)
        self.state_size = (2,)
        #self.state_size = (1)
        self.output_size = (1,)

    def call(self, input_at_t, states_at_t):
        #output_at_t = input_at_t
        #s1, s2 = states_at_t[0]
        s1 = states_at_t[0][0] + states_at_t[0][0]
        s2 = states_at_t[0][1] + input_at_t
        output_at_t = input_at_t + states_at_t[0][1]
        state_at_t_plus_1 = states_at_t[0]  + [1.0, 5.0]
        #state_at_t_plus_1 = states_at_t[0] + input_at_t
        print(" ")
        print (f' input = {input_at_t}')
        print(f'States = {states_at_t}')
        return output_at_t, state_at_t_plus_1


    """ def get_initial_state(self, value, inputs=None, batch_size=None, dtype=None):
        if inputs.shape[0] == None:
            bs = batch_size
        else:
            bs = inputs.size[0]

        if inputs.dtype == None:
            dt = dtype
        else:
            dt = inputs.dtype

        output = tf.fill((bs,self.state_size), value, dtype=dt)

        return output """


        
def train():
    n_layers = 10
    batch_size = 2

    is_dynamic_input = True

    tmp = tf.range(n_layers)
    tmp = tf.expand_dims(tmp, axis=1)
    tmp = tf.expand_dims(tmp, axis=0)
    offset = 1000
    tmp2 = tf.range(start=offset, limit=offset + n_layers)
    tmp2 = tf.expand_dims(tmp2, axis=1)
    tmp2 = tf.expand_dims(tmp2, axis=0)
    tmp = tf.concat([tmp,tmp2], axis=0)
    print (f"input = {tmp}   shape = {tmp.shape}")

    input = Input(shape=(n_layers, 1,), batch_size=batch_size, name="input")


    layer = tf.keras.layers.RNN(Cell(batch_size), return_sequences=True, return_state=True, go_backwards=False, time_major=False)

    if is_dynamic_input:
        initial_state = Input(shape=(2,4,),batch_size=batch_size, name="initial_state")
        initial_state = Input(shape=(2,),batch_size=batch_size, name="initial_state")
        initial_state_2 = Input(shape=(1,),batch_size=batch_size, name="initial_state")
    else:
        initial_state = tf.fill([batch_size, 2, 4], 100.0) # this doesn't work if batch is different from input size

    upward_output, upward_state = layer(inputs=input, initial_state=[initial_state])

    if is_dynamic_input:
        model = Model(inputs=[input, initial_state], outputs=[upward_output,upward_state])
        state = tf.constant([[[10, 14, 17, 20],[110, 120, 130, 140]],
                             [[20, 24, 27, 30],[410, 420, 530, 640]]])
        
        state = tf.constant([[10.0,14.0],
                             [640.0,9.0]])

        print (f"output = {model(inputs=[tmp, state])}")
    else:
        model = Model(inputs=[input], outputs=[upward_output,upward_state])
        print (f"output = {model(inputs=[tmp])}")


if __name__ == "__main__":
    train()
