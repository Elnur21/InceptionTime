from tensorflow import keras

class InceptionModule(keras.layers.Layer):

    def __init__(self, num_filters=32, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        self.activation = keras.activations.get(activation)

    def _default_Conv1D(self, filters, kernel_size):
        return keras.layers.Conv1D(filters=filters,
                                   kernel_size=kernel_size,
                                   strides=1,
                                   activation="relu", 
                                   padding='same', 
                                   use_bias=False)

    def call(self, inputs):
        # Step 1
        Z_bottleneck = self._default_Conv1D(filters=self.num_filters, kernel_size=1)(inputs)
        Z_maxpool = keras.layers.MaxPool1D(pool_size=3, strides=1, padding='same')(inputs)

        # Step 2
        Z1 = self._default_Conv1D(filters=self.num_filters, kernel_size=10)(Z_bottleneck)
        Z2 = self._default_Conv1D(filters=self.num_filters, kernel_size=20)(Z_bottleneck)
        Z3 = self._default_Conv1D(filters=self.num_filters, kernel_size=40)(Z_bottleneck)
        Z4 = self._default_Conv1D(filters=self.num_filters, kernel_size=1)(Z_maxpool)

        # Step 3
        Z = keras.layers.Concatenate(axis=2)([Z1, Z2, Z3, Z4])
        Z = keras.layers.BatchNormalization()(Z)
        return self.activation(Z)
