
"""
    Copyright (C) 2022 Francesca Meneghello
    contact: meneghello@dei.unipd.it
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import tensorflow as tf


class ConvNormalization(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='same', activation='selu',
                 kernel_initializer='lecun_normal', bias_initializer='zeros', bn=False, name_layer=None, **kwargs):
        super(ConvNormalization, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.name_layer = name_layer
        self.conv_l = tf.keras.layers.Conv2D(self.filters, self.kernel_size, strides=self.strides, padding=self.padding,
                                             name=self.name_layer, kernel_initializer=self.kernel_initializer,
                                             bias_initializer=self.bias_initializer)
        self.bn = bn
        self.activation = activation
        if bn:
            bn_name = None if self.name_layer is None else self.name_layer + '_bn'
            self.bn_l = tf.keras.layers.BatchNormalization(axis=3, name=bn_name)
        if activation is not None:
            self.act_l = tf.keras.layers.Activation(self.activation)

    def call(self, x_in):
        x = self.conv_l(x_in)
        if self.bn:
            x = self.bn_l(x)
        if self.activation is not None:
            x = self.act_l(x)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
             'filters': self.filters,
             'kernel_size': self.kernel_size,
             'strides': self.strides,
             'padding': self.padding,
             'activation': self.activation,
             'kernel_initializer': self.kernel_initializer,
             'bias_initializer': self.bias_initializer,
             'bn': self.bn,
             'name_layer': self.name_layer
        })
        return config


def conv_network(input_sh, num_classes, base_name):

    x_in = tf.keras.Input(input_sh)

    x = ConvNormalization(64, (1, 7), name_layer=base_name + '_conv1')(x_in)
    x = tf.keras.layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2), name=base_name + '_maxpool1')(x)
    x = ConvNormalization(64, (1, 7), name_layer=base_name + '_conv2')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2), name=base_name + '_maxpool2')(x)
    x = ConvNormalization(64, (1, 7), name_layer=base_name + '_conv3')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2), name=base_name + '_maxpool3')(x)
    x = ConvNormalization(64, (1, 5), name_layer=base_name + '_conv4')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2), name=base_name + '_maxpool4')(x)
    x = ConvNormalization(64, (1, 3), name_layer=base_name + '_conv5')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2), name=base_name + '_maxpool5')(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='selu', kernel_initializer='lecun_normal',
                              bias_initializer='zeros')(x)
    x = tf.keras.layers.AlphaDropout(0.5)(x)

    x = tf.keras.layers.Dense(64, activation='selu', kernel_initializer='lecun_normal',
                              bias_initializer='zeros')(x)
    x = tf.keras.layers.AlphaDropout(0.2)(x)

    x = tf.keras.layers.Dense(num_classes, activation=None)(x)

    model = tf.keras.Model(inputs=x_in, outputs=x, name='conv_net')

    return model


def conv_network_hyper_selection(input_sh, num_classes, filters_dimension, kernels_dimension, base_name):

    x_in = tf.keras.Input(input_sh)

    num_filters = len(filters_dimension)
    if len(kernels_dimension) != num_filters:
        print('the number of filters and kernels must coincide')

    x = ConvNormalization(filters_dimension[0], (1, kernels_dimension[0]), name_layer=base_name + '_conv1')(x_in)
    x = tf.keras.layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2), name=base_name + '_maxpool1')(x)

    for idx_filter in range(1, num_filters):
        x = ConvNormalization(filters_dimension[idx_filter], (1, kernels_dimension[idx_filter]),
                              name_layer=base_name + '_conv' + str(idx_filter + 1))(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2),
                                      name=base_name + '_maxpool1' + str(idx_filter + 1))(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='selu', kernel_initializer='lecun_normal',
                              bias_initializer='zeros')(x)
    x = tf.keras.layers.AlphaDropout(0.5)(x)

    x = tf.keras.layers.Dense(64, activation='selu', kernel_initializer='lecun_normal',
                              bias_initializer='zeros')(x)
    x = tf.keras.layers.AlphaDropout(0.2)(x)

    x = tf.keras.layers.Dense(num_classes, activation=None)(x)

    model = tf.keras.Model(inputs=x_in, outputs=x, name='conv_net')

    return model


def att_network(input_sh, num_classes):
    x_input = tf.keras.Input(input_sh)
    base_name = 'new'
    x = ConvNormalization(128, (1, 7), name_layer=base_name + '_conv1')(x_input)
    x = tf.keras.layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2), name=base_name + '_maxpool1')(x)
    x = ConvNormalization(128, (1, 7), name_layer=base_name + '_conv2')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2), name=base_name + '_maxpool2')(x)
    x = ConvNormalization(128, (1, 7), name_layer=base_name + '_conv3')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2), name=base_name + '_maxpool3')(x)
    x = ConvNormalization(128, (1, 5), name_layer=base_name + '_conv4')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2), name=base_name + '_maxpool4')(x)
    x = ConvNormalization(128, (1, 3), name_layer=base_name + '_conv5')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2), name=base_name + '_maxpool5')(x)

    x2_avg = tf.reduce_mean(x, axis=3, keepdims=True)
    x2_max = tf.reduce_max(x, axis=3, keepdims=True)
    x2_concat = tf.concat([x2_avg, x2_max], axis=3)
    att2 = tf.keras.layers.Conv2D(1, (1, 5), activation='sigmoid', padding='same')(x2_concat)
    x_att = tf.multiply(x, att2)

    x = tf.add(x, x_att)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='selu', kernel_initializer='lecun_normal',
                              bias_initializer='zeros')(x)
    x = tf.keras.layers.AlphaDropout(0.5)(x)

    x = tf.keras.layers.Dense(64, activation='selu', kernel_initializer='lecun_normal',
                              bias_initializer='zeros')(x)
    x = tf.keras.layers.AlphaDropout(0.2)(x)

    logits = tf.keras.layers.Dense(num_classes, activation=None)(x)

    model = tf.keras.Model(inputs=x_input, outputs=logits, name='self_att_net')

    return model


def att_network_hyper_selection(input_sh, num_classes, filters_dimension, kernels_dimension, base_name):

    x_in = tf.keras.Input(input_sh)

    num_filters = len(filters_dimension)
    if len(kernels_dimension) != num_filters:
        print('the number of filters and kernels must coincide')

    x = ConvNormalization(filters_dimension[0], (1, kernels_dimension[0]), name_layer=base_name + '_conv1')(x_in)
    x = tf.keras.layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2), name=base_name + '_maxpool1')(x)

    for idx_filter in range(1, num_filters):
        x = ConvNormalization(filters_dimension[idx_filter], (1, kernels_dimension[idx_filter]),
                              name_layer=base_name + '_conv' + str(idx_filter + 1))(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2),
                                      name=base_name + '_maxpool1' + str(idx_filter + 1))(x)

    x2_avg = tf.reduce_mean(x, axis=3, keepdims=True)
    x2_max = tf.reduce_max(x, axis=3, keepdims=True)
    x2_concat = tf.concat([x2_avg, x2_max], axis=3)
    att2 = tf.keras.layers.Conv2D(1, (1, 5), activation='sigmoid', padding='same')(x2_concat)
    x_att = tf.multiply(x, att2)

    x = tf.add(x, x_att)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='selu', kernel_initializer='lecun_normal',
                              bias_initializer='zeros')(x)
    x = tf.keras.layers.AlphaDropout(0.5)(x)

    x = tf.keras.layers.Dense(64, activation='selu', kernel_initializer='lecun_normal',
                              bias_initializer='zeros')(x)
    x = tf.keras.layers.AlphaDropout(0.2)(x)

    logits = tf.keras.layers.Dense(num_classes, activation=None)(x)

    model = tf.keras.Model(inputs=x_in, outputs=logits, name='self_att_net')

    return model
