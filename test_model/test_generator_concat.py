import numpy as np
import pandas as pd


class TestSlidingWindowGenerator(object):
    def __init__(self, number_of_windows, offset, predict_mode, appliance_name_list, test_directory, appliance_count):
        self.__number_of_windows = number_of_windows
        self.__offset = offset
        self.__test_directory = test_directory
        self.__predict_mode = predict_mode
        self.__appliance_name_list = appliance_name_list
        self.data_array = np.array(pd.read_csv(test_directory, skiprows=0, header=0))
        self.total_size = len(self.data_array)
        self.window_size = 2 * offset + 1
        self.max_number_of_windows = self.total_size - 2 * self.__offset
        self.__appliance_count = appliance_count

    def load_dataset(self):
        inputs, outputs = self.generate_dataset_concat()
        indices = np.arange(self.max_number_of_windows, dtype=int)
        for start_index in range(0, self.max_number_of_windows, self.__number_of_windows):
            splice = indices[start_index: start_index + self.__number_of_windows]
            # 生成input
            input_data_list = []
            for index in splice:
                input_data_temp = inputs[index: index + self.window_size]
                input_data_list.append(input_data_temp)
            input_data = np.array(input_data_list)
            input1 = input_data[:, :, [0]]
            input2 = input_data[:, :, [1, 2, 3]]
            input_all = [input1, input2]
            # 生成output
            output_data_list = []
            for index in splice:
                output_data_temp = outputs[index + self.__offset]
                output_data_list.append(output_data_temp)
            target_data = np.array(output_data_list)
            target_data = target_data.reshape(-1, self.__appliance_count)
            yield input_all, target_data

    def generate_dataset_concat(self):
        data_array = self.data_array
        inputs = data_array[:, 0: 4]
        inputs = np.reshape(inputs, (-1, 4))
        outputs = data_array[self.__offset: -self.__offset, -self.__appliance_count:]
        outputs = np.reshape(outputs, (-1, self.__appliance_count))
        return inputs, outputs
