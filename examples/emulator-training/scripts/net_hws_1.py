import os

import numpy as np


import tensorflow as tf
from tensorflow.keras import losses, optimizers, layers, Input, Model, Layer, Sequential
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.layers import Dense,TimeDistributed


concat_input_state = tf.concat([input, state], axis=-1)

class AtmTransmission(Layer):
    def __init__(self, *, n_state_variables, n_input_layers, n_internal_layers, n_output_layers, n_nodes_input_layer, n_nodes_internal_layer, n_nodes_output_layer):
        super().__init__()

        # Need to add initialization
        # Things to add: 
        # - Constraint: sum of output and flux = 1 
        # - Multiply processed input and flux

        self.input_layer = [tf.keras.layers.Dense(n_nodes_input_layer, activation='elu', name='input_layer_' + str(i)) for i in range(n_input_layers)]

        self.internal_layers = [tf.keras.layers.Dense(n_nodes_internal_layer,    activation='elu', name='internal_layer_' + str(i)) for i in range(n_internal_layers)]

        self.output_layers = [tf.keras.layers.Dense(n_nodes_output_layer, activation='elu', name='output_layer_' + str(i)) for i in range(n_output_layers)]

        self.final_output_layer = tf.keras.layers.Dense(3, activation='softmax', name='final_output_layer')

    def _evaluate_layer(self, input, state):
        x = input
        for layer in self.input_layers:
            x = layer(x)

        state = tf.concat([x, state], axis=-1)
        for layer in self.internal_layers:
            state = layer(state)

        output = state

        for layer in self.output_layers:
            output = layer(output)

        output = self.final_output_layer(output)

        return output, state

    def forward(self, inputs, state):
        outputs = []
        for input in inputs:  # Shape of inputs: (num_steps, batch_size, num_inputs)
            output, state = self._evaluate_layer(input, state)
            outputs.append(output)
        return outputs, state

class AtmTransmissionOld(Layer):
    def __init__(self, n_inputs, n_flux, n_internal_variables, n_internal_layers, units=32):
        super(AtmTransmission, self).__init__()

        #Input to Internal
        self.W_xi = tf.Variable(tf.random.normal(
            (n_inputs, n_internal_variables)) * sigma)

        #Flux to Internal
        self.W_fi = tf.Variable(tf.random.normal(
            (n_flux, n_internal_variables)) * sigma)
        self.b_x = tf.Variable(tf.zeros(n_internal_variables))

        #Internal to Internal
        self.W_ii = tf.Variable(tf.random.normal(
            (n_internal_layers, n_internal_variables, n_internal_variables)) * sigma)

        self.b_ii = tf.Variable(tf.zeros(n_internal_layers, n_internal_variables))

        #Internal to Flux
        self.W_if = tf.Variable(tf.random.normal(
            (n_internal_variables, n_flux)) * sigma)

        self.b_if = tf.Variable(tf.zeros(n_flux))

        #Internal to Output
        self.W_io = tf.Variable(tf.random.normal(
            (n_internal_variables, 1)) * sigma)

        self.b_io = tf.Variable(tf.zeros(1))


        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, units), dtype="float32"),
            trainable=True,
        )
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
        )

    def call(self, inputs, flux):
        ii = tf.matmul(inputs, self.w_xi) + tf.matmul(flux, self.w_fi) + self.b_x
        for j in self.W_ii.shape[0]:
            ii = tf.matmul(ii, self.w_ii) + self.b_ii



inputs = Input(shape=(n_layers,n_input_variables), name='inputs')

layer_properties = TimeDistributed(layers.Dense(n_layer_properties, activation='elu'),name='layer_properties')(inputs)

# absorbed flux + downward flux = 1 at each level
downward_flux, absorbed_radiation_down = TimeDistributed(layers.Dense(n_flux, activation='elu'),name='downward_flux')(layer_properties,initial_state=TOA_flux)

surface_flux = downward_flux[-1] * albedo

upward_flux, absorbed_radiation_up = TimeDistributed(layers.Dense(n_flux, activation='elu'),name='upward_flux')(layer_properties, initial_state=surface_flux)

outputs = downward_flux + tf.reverse(upward_flux, [1])



