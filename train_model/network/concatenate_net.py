import tensorflow as tf


def create_lstm(input_window_length):
    input_layer = tf.keras.layers.Input(shape=(input_window_length, 1))
    conv1d_layer = tf.keras.layers.Conv1D(filters=32, kernel_size=5, activation='relu')(input_layer)
    maxpool_layer = tf.keras.layers.MaxPooling1D(3)(conv1d_layer)
    conv1d_layer = tf.keras.layers.Conv1D(filters=32, kernel_size=5, activation='relu')(maxpool_layer)
    lstm_layer = tf.keras.layers.LSTM(32, dropout=0.1)(conv1d_layer)
    dense_layer = tf.keras.layers.Dense(16, activation='relu')(lstm_layer)
    model = tf.keras.Model(inputs=input_layer, outputs=dense_layer)
    return model


def create_mlp(input_window_length):
    input_layer = tf.keras.layers.Input(shape=(input_window_length, 3))
    flatten_layer = tf.keras.layers.Flatten()(input_layer)
    second_layer = tf.keras.layers.Dense(32, activation='relu')(flatten_layer)
    third_layer = tf.keras.layers.Dense(16, activation='relu')(second_layer)
    model = tf.keras.Model(inputs=input_layer, outputs=third_layer)
    return model


def create_concatenate(dim, appliance_count, predict_mode):
    lstm = create_lstm(dim)
    mlp = create_mlp(dim)
    combined_input = tf.keras.layers.concatenate([lstm.output, mlp.output])
    final_layer = tf.keras.layers.Dense(16, activation='relu')(combined_input)
    if predict_mode == 'multi_label':
        output_layer = tf.keras.layers.Dense(appliance_count, activation="sigmoid")(final_layer)
    else:
        output_layer = tf.keras.layers.Dense(appliance_count, activation='linear')(final_layer)
    model = tf.keras.Model(inputs=[lstm.input, mlp.input], outputs=output_layer)
    return model
