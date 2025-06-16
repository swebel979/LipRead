import os
import tf_keras

# Set environment variable to use Keras 2
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Define the load_model function
def load_model() -> tf_keras.Sequential:
    model = tf_keras.Sequential()

    # Add the layers as before
    model.add(tf_keras.layers.Conv3D(128, 3, input_shape=(75,46,140,1), padding='same'))
    model.add(tf_keras.layers.Activation('relu'))
    model.add(tf_keras.layers.MaxPool3D((1,2,2)))

    model.add(tf_keras.layers.Conv3D(256, 3, padding='same'))
    model.add(tf_keras.layers.Activation('relu'))
    model.add(tf_keras.layers.MaxPool3D((1,2,2)))

    model.add(tf_keras.layers.Conv3D(75, 3, padding='same'))
    model.add(tf_keras.layers.Activation('relu'))
    model.add(tf_keras.layers.MaxPool3D((1,2,2)))

    model.add(tf_keras.layers.TimeDistributed(tf_keras.layers.Flatten()))

    model.add(tf_keras.layers.Bidirectional(tf_keras.layers.LSTM(128, kernel_initializer='orthogonal', return_sequences=True)))
    model.add(tf_keras.layers.Dropout(.5))

    model.add(tf_keras.layers.Bidirectional(tf_keras.layers.LSTM(128, kernel_initializer='orthogonal', return_sequences=True)))
    model.add(tf_keras.layers.Dropout(.5))

    model.add(tf_keras.layers.Dense(41, kernel_initializer='he_normal', activation='softmax'))
    # Load the weights from the file
    model.load_weights(os.path.join('models','checkpoint'))

    return model