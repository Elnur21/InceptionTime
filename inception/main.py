from tensorflow import keras
from .InceptionModule import *

def shortcut_layer(inputs, Z_inception):
    Z_shortcut = keras.layers.Conv1D(filters=int(Z_inception.shape[-1]), kernel_size=1, padding='same', use_bias=False) (inputs)
    Z_shortcut = keras.layers. BatchNormalization()(Z_shortcut)
    Z = keras.layers.Add()([Z_shortcut, Z_inception])
    return keras.layers.Activation('relu')(Z)

def build_model(input_shape, num_classes, num_modules=6):

    input_layer = keras.layers.Input(shape=input_shape)

    Z = input_layer

    Z_residual = input_layer

    for i in range(num_modules):

        Z = InceptionModule().call(Z)
        if i % 3 == 2: 
            Z = shortcut_layer(Z_residual, Z) 
            Z_residual = Z

    gap_layer = keras.layers.GlobalAveragePooling1D()(Z) 
    output_layer = keras.layers.Dense(num_classes, activation='softmax')(gap_layer)

    model = keras.models.Model(inputs = input_layer, outputs = output_layer) 
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.01), metrics=['accuracy'])

    return model