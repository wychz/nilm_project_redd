import tensorflow as tf
import os

tf.random.set_seed(120)


def model_select(input_window_length, model_type, appliance_count, predict_mode):
    if model_type == 'cnn':
        input_layer = tf.keras.layers.Input(shape=(input_window_length, 1))
        reshape_layer = tf.keras.layers.Reshape((1, input_window_length, 1))(input_layer)
        conv_layer_1 = tf.keras.layers.Convolution2D(filters=30, kernel_size=(10, 1), strides=(1, 1), padding="same",
                                                     activation="relu")(reshape_layer)
        conv_layer_2 = tf.keras.layers.Convolution2D(filters=30, kernel_size=(8, 1), strides=(1, 1), padding="same",
                                                     activation="relu")(conv_layer_1)
        conv_layer_3 = tf.keras.layers.Convolution2D(filters=40, kernel_size=(6, 1), strides=(1, 1), padding="same",
                                                     activation="relu")(conv_layer_2)
        conv_layer_4 = tf.keras.layers.Convolution2D(filters=50, kernel_size=(5, 1), strides=(1, 1), padding="same",
                                                     activation="relu")(conv_layer_3)
        conv_layer_5 = tf.keras.layers.Convolution2D(filters=50, kernel_size=(5, 1), strides=(1, 1), padding="same",
                                                     activation="relu")(conv_layer_4)
        flatten_layer = tf.keras.layers.Flatten()(conv_layer_5)
        label_layer = tf.keras.layers.Dense(1024, activation="relu")(flatten_layer)
        output_layer = tf.keras.layers.Dense(appliance_count, activation="linear")(label_layer)
        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        return model


    if model_type == 'lstm':
        input_layer = tf.keras.layers.Input(shape=(input_window_length, 1))
        conv1d_layer = tf.keras.layers.Conv1D(filters=32, kernel_size=5, activation='relu')(input_layer)
        maxpool_layer = tf.keras.layers.MaxPooling1D(3)(conv1d_layer)
        conv1d_layer = tf.keras.layers.Conv1D(filters=32, kernel_size=5, activation='relu')(maxpool_layer)
        lstm_layer = tf.keras.layers.LSTM(32, dropout=0.1)(conv1d_layer)
        if predict_mode == 'multi_label':
            dense_layer = tf.keras.layers.Dense(appliance_count, activation="sigmoid")(lstm_layer)
        else:
            dense_layer = tf.keras.layers.Dense(appliance_count, activation='linear')(lstm_layer)
        model = tf.keras.Model(inputs=input_layer, outputs=dense_layer)
        return model


def save_model(model, save_model_dir):
    model_path = save_model_dir
    if not os.path.exists(model_path):
        open((model_path), 'a').close()
    model.save(model_path)


def load_model(saved_model_dir):
    print("PATH NAME: ", saved_model_dir)
    model = tf.keras.models.load_model(saved_model_dir)
    num_of_weights = model.count_params()
    print("Loaded model with ", str(num_of_weights), " weights")
    return model
