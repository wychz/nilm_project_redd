import numpy as np


class TestSlidingWindowGenerator(object):
    def __init__(self, number_of_windows, inputs, targets, offset, predict_mode, appliance_name_list):
        self.__number_of_windows = number_of_windows
        self.__offset = offset
        self.__inputs = inputs
        self.__targets = targets
        self.total_size = len(inputs)
        self.__predict_mode = predict_mode
        self.__appliance_name_list = appliance_name_list

    def load_dataset(self):
        self.__inputs, self.__targets = self.generate_test_data()
        max_number_of_windows = self.__inputs.size - 2 * self.__offset
        if self.__number_of_windows < 0:
            self.__number_of_windows = max_number_of_windows

        indices = np.arange(max_number_of_windows, dtype=int)

        if self.__predict_mode == 'single':
            for start_index in range(0, max_number_of_windows, self.__number_of_windows):
                splice = indices[start_index: start_index + self.__number_of_windows]
                input_data = np.array([self.__inputs[index: index + 2 * self.__offset + 1] for index in splice])
                target_data = self.__targets[splice + self.__offset].reshape(-1, 1)
                yield input_data, target_data
        elif self.__predict_mode == 'multiple' or self.__predict_mode == 'multi_label':
            for start_index in range(0, max_number_of_windows, self.__number_of_windows):
                splice = indices[start_index: start_index + self.__number_of_windows]
                input_data = np.array([self.__inputs[index: index + 2 * self.__offset + 1] for index in splice])
                target_data = self.__targets[splice + self.__offset].reshape(-1, len(self.__appliance_name_list))
                yield input_data, target_data

    def generate_test_data(self):
        if self.__predict_mode == 'single':
            self.__inputs = self.__inputs.flatten()
            self.__inputs = np.reshape(self.__inputs, (-1, 1))
            self.__targets = np.reshape(self.__targets, (-1, 1))
            return self.__inputs, self.__targets
        elif self.__predict_mode == 'multiple' or self.__predict_mode == 'multi_label':
            self.__inputs = self.__inputs.flatten()
            self.__inputs = np.reshape(self.__inputs, (-1, 1))
            self.__targets = np.reshape(self.__targets, (-1, len(self.__appliance_name_list)))
            return self.__inputs, self.__targets