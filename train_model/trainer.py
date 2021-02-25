import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from train_model.model import model_select, save_model
from train_model.train_generator import TrainSlidingWindowGenerator
from train_model.train_generator_common import TrainSlidingWindowGeneratorCommon


class Trainer:

    def __init__(self, appliance, batch_size, model_type,
                 training_directory, validation_directory, save_model_dir, predict_mode, appliance_count,
                 epochs=100, input_window_length=50, validation_frequency=1,
                 patience=3, min_delta=1e-6, verbose=1, learning_rate=0.001):
        self.__appliance = appliance
        self.__model_type = model_type
        self.__batch_size = batch_size
        self.__epochs = epochs
        self.__patience = patience
        self.__min_delta = min_delta
        self.__verbose = verbose
        self.__loss = "mse"
        self.__metrics = ["mse", "msle", "mae"]
        self.__learning_rate = learning_rate
        self.__beta_1 = 0.9
        self.__beta_2 = 0.999
        self.__save_model_dir = save_model_dir
        self.__predict_mode = predict_mode
        if self.__predict_mode == 'single':
            self.__appliance_count = 1
        else:
            self.__appliance_count = appliance_count
        self.__input_window_length = input_window_length
        self.__window_offset = int((0.5 * (self.__input_window_length + 2)) - 1)
        self.__validation_frequency = validation_frequency
        self.__validation_steps = 100
        self.__training_directory = training_directory
        self.__validation_directory = validation_directory
        np.random.seed(120)
        tf.random.set_seed(120)
        if model_type == 'concat':
            self.__training_generator = TrainSlidingWindowGenerator(file_name=self.__training_directory,
                                                                    batch_size=self.__batch_size,
                                                                    shuffle=True,
                                                                    offset=self.__window_offset,
                                                                    predict_mode=self.__predict_mode,
                                                                    appliance_count=self.__appliance_count)
            self.__validation_generator = TrainSlidingWindowGenerator(file_name=self.__validation_directory,
                                                                      batch_size=self.__batch_size,
                                                                      shuffle=True,
                                                                      offset=self.__window_offset,
                                                                      predict_mode=self.__predict_mode,
                                                                      appliance_count=self.__appliance_count)
        else:
            self.__training_generator = TrainSlidingWindowGeneratorCommon(file_name=self.__training_directory,
                                                                          batch_size=self.__batch_size,
                                                                          shuffle=True,
                                                                          offset=self.__window_offset,
                                                                          predict_mode=self.__predict_mode,
                                                                          appliance_count=self.__appliance_count)
            self.__validation_generator = TrainSlidingWindowGeneratorCommon(file_name=self.__validation_directory,
                                                                            batch_size=self.__batch_size,
                                                                            shuffle=True,
                                                                            offset=self.__window_offset,
                                                                            predict_mode=self.__predict_mode,
                                                                            appliance_count=self.__appliance_count)

    def train_model(self):
        steps_per_training_epoch = np.round(int(self.__training_generator.maximum_batch_size / self.__batch_size),
                                            decimals=0)
        model = model_select(self.__input_window_length, self.__model_type, self.__appliance_count, self.__predict_mode)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.__learning_rate, beta_1=self.__beta_1,
                                                         beta_2=self.__beta_2),
                      loss=self.__loss, metrics=self.__metrics)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=self.__min_delta,
                                                          patience=self.__patience, verbose=self.__verbose, mode="auto")
        callbacks = [early_stopping]
        training_history = self.default_train(model, callbacks, steps_per_training_epoch)
        training_history.history["val_loss"] = np.repeat(training_history.history["val_loss"],
                                                         self.__validation_frequency)
        model.summary()
        save_model(model, self.__save_model_dir)
        self.plot_training_results(training_history)

    def default_train(self, model, callbacks, steps_per_training_epoch):
        training_history = model.fit(self.__training_generator.load_dataset(),
                                     steps_per_epoch=steps_per_training_epoch,
                                     epochs=self.__epochs,
                                     verbose=self.__verbose,
                                     callbacks=callbacks,
                                     validation_data=self.__validation_generator.load_dataset(),
                                     validation_freq=self.__validation_frequency,
                                     validation_steps=self.__validation_steps)
        return training_history

    def plot_training_results(self, training_history):
        plt.plot(training_history.history["loss"], label="MSE (Training Loss)")
        plt.plot(training_history.history["val_loss"], label="MSE (Validation Loss)")
        plt.title('Training History')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
