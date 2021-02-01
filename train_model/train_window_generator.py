import numpy as np
import pandas as pd

class TrainSlidingWindowGenerator:

    def __init__(self, predict_mode, appliance_count, file_name, chunk_size, shuffle, offset, batch_size=1000, crop=100000, skip_rows=0, ram_threshold=5 * 10 ** 5):
        self.__predict_mode = predict_mode
        self.__appliance_count = appliance_count
        self.__file_name = file_name
        self.__batch_size = batch_size
        self.__chunk_size = 10 ** 8
        self.__shuffle = shuffle
        self.__offset = offset
        self.__crop = crop
        self.__skip_rows = skip_rows
        self.__ram_threshold = ram_threshold
        self.total_size = 0
        self.__total_num_samples = crop
        np.random.seed(120)

    @property
    def total_num_samples(self):
        return self.__total_num_samples

    @total_num_samples.setter
    def total_num_samples(self, value):
        self.__total_num_samples = value

    def check_if_chunking(self):
        print("Importing training file...")
        chunks = pd.read_csv(self.__file_name, header=0, nrows=self.__crop, skiprows=self.__skip_rows)
        print("Counting number of rows...")
        self.total_size = len(chunks)
        del chunks
        print("Done.")
        print("The dataset contains ", self.total_size, " rows")
        if self.total_size > self.__ram_threshold:
            print("There is too much data to load into memory, so it will be loaded in chunks. Please note that this "
                  "may result in decreased training times.")

    def load_dataset(self):
        if self.total_size == 0:
            self.check_if_chunking()

        if self.total_size < self.__ram_threshold:
            data_array = np.array(pd.read_csv(self.__file_name, nrows=self.__crop, skiprows=self.__skip_rows, header=0))
            inputs, outputs = self.generate_train_data(data_array)
            maximum_batch_size = inputs.size - 2 * self.__offset
            self.total_num_samples = maximum_batch_size
            if self.__batch_size < 0:
                self.__batch_size = maximum_batch_size
            indices = np.arange(maximum_batch_size)
            if self.__shuffle:
                np.random.shuffle(indices)
            while True:
                for start_index in range(0, maximum_batch_size, self.__batch_size):
                    splice = indices[start_index: start_index + self.__batch_size]
                    input_data = np.array([inputs[index: index + 2 * self.__offset + 1] for index in splice])
                    if self.__predict_mode == 'single':
                        output_data = outputs[splice + self.__offset].reshape(-1, 1)
                        yield input_data, output_data
                    elif self.__predict_mode == 'multiple' or self.__predict_mode == 'multi_label':
                        output_data = outputs[splice + self.__offset].reshape(-1, self.__appliance_count)
                        yield input_data, output_data

        if self.total_size >= self.__ram_threshold:
            number_of_chunks = np.arange(self.total_size / self.__chunk_size)
            if self.__shuffle:
                np.random.shuffle(number_of_chunks)
            for index in number_of_chunks:
                data_array = np.array(pd.read_csv(self.__file_name, skiprows=int(index) * self.__chunk_size, header=0,nrows=self.__crop))
                inputs, outputs = self.generate_train_data(data_array)
                maximum_batch_size = inputs.size - 2 * self.__offset
                self.total_num_samples = maximum_batch_size
                if self.__batch_size < 0:
                    self.__batch_size = maximum_batch_size
                indices = np.arange(maximum_batch_size)
                if self.__shuffle:
                    np.random.shuffle(indices)
                while True:
                    for start_index in range(0, maximum_batch_size, self.__batch_size):
                        splice = indices[start_index: start_index + self.__batch_size]
                        input_data = np.array([inputs[index: index + 2 * self.__offset + 1] for index in splice])
                        if self.__predict_mode == 'single':
                            output_data = outputs[splice + self.__offset].reshape(-1, 1)
                            yield input_data, output_data
                        elif self.__predict_mode == 'multiple' or self.__predict_mode == 'multi_label':
                            output_data = outputs[splice + self.__offset].reshape(-1, self.__appliance_count)
                            yield input_data, output_data

    def generate_train_data(self, data_array):
        if self.__predict_mode == 'single':
            inputs = data_array[:, 0]
            inputs = np.reshape(inputs, (-1, 1))
            outputs = data_array[:, 1]
            outputs = np.reshape(outputs, (-1, 1))
            return inputs, outputs
        elif self.__predict_mode == 'multiple' or self.__predict_mode == 'multi_label':
            inputs = data_array[:, 0]
            inputs = np.reshape(inputs, (-1, 1))
            outputs = data_array[:, 1:]
            outputs = np.reshape(outputs, (-1, self.__appliance_count))
            return inputs, outputs