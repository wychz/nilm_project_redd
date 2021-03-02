import numpy as np
import pandas as pd


class TrainSlidingWindowGeneratorCommon:

    def __init__(self, predict_mode, appliance_count, file_name, shuffle, offset, batch_size):
        self.__predict_mode = predict_mode
        self.__appliance_count = appliance_count
        self.__file_name = file_name
        self.__batch_size = batch_size
        self.__chunk_size = 10 ** 8
        self.__shuffle = shuffle
        self.__offset = offset

        self.data_array = np.array(pd.read_csv(self.__file_name, header=0))
        self.total_size = len(self.data_array)
        self.maximum_batch_size = len(self.data_array) - 2 * self.__offset
        self.window_size = 2 * offset + 1
        np.random.seed(120)

    def load_dataset(self):
        print("The dataset contains ", self.total_size, " rows")
        inputs, outputs = self.generate_train_data(self.data_array)
        indices = np.arange(self.maximum_batch_size)
        if self.__shuffle:
            np.random.shuffle(indices)
        while True:
            for start_index in range(0, self.maximum_batch_size, self.__batch_size):
                splice = indices[start_index: start_index + self.__batch_size]
                input_data = np.array([inputs[index: index + 2 * self.__offset + 1] for index in splice])
                output_data = outputs[splice + self.__offset].reshape(-1, self.__appliance_count)
                yield input_data, output_data

    def generate_train_data(self, data_array):
        inputs = data_array[:, 0]
        inputs = np.reshape(inputs, (-1, 1))
        outputs = data_array[:, -self.__appliance_count:]
        outputs = np.reshape(outputs, (-1, self.__appliance_count))
        return inputs, outputs
