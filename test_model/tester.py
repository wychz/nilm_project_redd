import logging
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from appliance_param import appliance_param, mains_data
from test_model.metrics import recall_precision_accuracy_f1, relative_error_total_energy, mean_absolute_error
from test_model.test_data import TestSlidingWindowGenerator
from train_model.model import load_model


class Tester:
    def __init__(self, appliance, crop, batch_size, model_type, predict_mode, appliance_name_list,
                 test_directory, saved_model_dir, log_file_dir,
                 input_window_length):
        self.__appliance = appliance
        self.__model_type = model_type
        self.__predict_mode = predict_mode
        self.__appliance_name_list = appliance_name_list
        self.__crop = crop
        self.__batch_size = batch_size
        self._input_window_length = input_window_length
        self.__window_size = self._input_window_length + 2
        self.__window_offset = int(0.5 * self.__window_size - 1)
        self.__number_of_windows = 100
        self.__test_directory = test_directory
        self.__saved_model_dir = saved_model_dir
        self.__log_file = log_file_dir
        # logging.basicConfig(filename=self.__log_file, level=logging.INFO)

    def test_model(self):
        test_input, test_target = self.load_dataset(self.__test_directory)
        model = load_model(self.__saved_model_dir)
        test_generator = TestSlidingWindowGenerator(number_of_windows=self.__number_of_windows, inputs=test_input,
                                                    appliance_name_list=self.__appliance_name_list,
                                                    targets=test_target, offset=self.__window_offset,
                                                    predict_mode=self.__predict_mode)
        steps_per_test_epoch = np.round(int(test_generator.total_size / self.__batch_size), decimals=0)
        start_time = time.time()
        testing_history = model.predict(x=test_generator.load_dataset(), steps=steps_per_test_epoch, verbose=2)
        end_time = time.time()
        test_time = end_time - start_time
        evaluation_metrics = model.evaluate(x=test_generator.load_dataset(), steps=steps_per_test_epoch)
        self.plot_results(model, test_time, evaluation_metrics, testing_history, test_input, test_target)

    def load_dataset(self, directory):
        data_frame = pd.read_csv(directory, nrows=self.__crop, skiprows=0, header=0)
        test_input = np.round(np.array(data_frame.iloc[:, 0], float), 6)
        if self.__predict_mode == 'single':
            test_target = np.round(np.array(data_frame.iloc[self.__window_offset: -self.__window_offset, 1], float), 6)
            del data_frame
            return test_input, test_target
        elif self.__predict_mode == 'multiple' or self.__predict_mode == 'multi_label':
            test_target = np.round(np.array(data_frame.iloc[self.__window_offset: -self.__window_offset, 1:], float), 6)
            del data_frame
            return test_input, test_target

    def log_results(self, model, test_time, evaluation_metrics):
        inference_log = "Inference Time: " + str(test_time)
        logging.info(inference_log)
        metric_string = "MSE: ", str(evaluation_metrics[0]), " MAE: ", str(evaluation_metrics[3])
        logging.info(metric_string)
        self.count_pruned_weights(model)

    def count_pruned_weights(self, model):
        num_total_zeros = 0
        num_dense_zeros = 0
        num_dense_weights = 0
        num_conv_zeros = 0
        num_conv_weights = 0
        for layer in model.layers:
            if np.shape(layer.get_weights())[0] != 0:
                layer_weights = layer.get_weights()[0].flatten()
                if "conv" in layer.name:
                    num_conv_weights += np.size(layer_weights)
                    num_conv_zeros += np.count_nonzero(layer_weights == 0)
                    num_total_zeros += np.size(layer_weights)
                else:
                    num_dense_weights += np.size(layer_weights)
                    num_dense_zeros += np.count_nonzero(layer_weights == 0)
        conv_zeros_string = "CONV. ZEROS: " + str(num_conv_zeros)
        conv_weights_string = "CONV. WEIGHTS: " + str(num_conv_weights)
        conv_sparsity_ratio = "CONV. RATIO: " + str(num_conv_zeros / num_conv_weights)
        dense_weights_string = "DENSE WEIGHTS: " + str(num_dense_weights)
        dense_zeros_string = "DENSE ZEROS: " + str(num_dense_zeros)
        dense_sparsity_ratio = "DENSE RATIO: " + str(num_dense_zeros / num_dense_weights)
        total_zeros_string = "TOTAL ZEROS: " + str(num_total_zeros)
        total_weights_string = "TOTAL WEIGHTS: " + str(model.count_params())
        total_sparsity_ratio = "TOTAL RATIO: " + str(num_total_zeros / model.count_params())
        print("LOGGING PATH: ", self.__log_file)
        logging.info(conv_zeros_string)
        logging.info(conv_weights_string)
        logging.info(conv_sparsity_ratio)
        logging.info("")
        logging.info(dense_zeros_string)
        logging.info(dense_weights_string)
        logging.info(dense_sparsity_ratio)
        logging.info("")
        logging.info(total_zeros_string)
        logging.info(total_weights_string)
        logging.info(total_sparsity_ratio)

    def plot_results(self, model, test_time, evaluation_metrics, testing_history, test_input, test_target):
        # self.log_results(model, test_time, evaluation_metrics)

        if self.__predict_mode == 'single':
            mean, std = generate_mean_std(self.__appliance)
            self.plot_results_single(testing_history, test_input, test_target, mean, std)

        elif self.__predict_mode == 'multiple':
            count = 0
            for appliance_name in self.__appliance_name_list:
                mean, std = generate_mean_std(appliance_name)
                self.plot_results_multiple(testing_history[:, count:count + 1], test_input,
                                           test_target[:, count:count + 1], appliance_name, count + 1, mean, std)
                count = count + 1
        elif self.__predict_mode == 'multi_label':
            count = 0
            for appliance_name in self.__appliance_name_list:
                self.plot_results_multiple_label(testing_history[:, count:count + 1], test_input,
                                                 test_target[:, count:count + 1], appliance_name)
                count = count + 1

    def plot_results_single(self, testing_history, test_input, test_target, mean, std):
        testing_history, test_target, test_agg = self.testing_data_process(testing_history, test_target, test_input, mean, std)
        threshold = generate_threshold(self.__appliance)
        rpaf, rete, mae = self.calculate_metrics(testing_history, test_agg, test_target, threshold)
        print_metrics(self.__appliance, rpaf, rete, mae)
        appliance_name = " "
        self.print_plots(test_agg, test_target, testing_history, 1, appliance_name)

    def plot_results_multiple(self, testing_history, test_input, test_target, appliance_name, count, mean, std):
        testing_history, test_target, test_agg = self.testing_data_process(testing_history, test_target, test_input, mean, std)
        threshold = generate_threshold(appliance_name)
        rpaf, rete, mae = self.calculate_metrics(testing_history, test_agg, test_target, threshold)
        print_metrics(appliance_name, rpaf, rete, mae)
        self.print_plots(test_agg, test_target, testing_history, count, appliance_name)

    def plot_results_multiple_label(self, testing_history, test_input, test_target, appliance_name):
        test_agg = test_input.flatten()
        test_agg = test_agg[:testing_history.size]
        rpaf, rete, mae = self.calculate_metrics(testing_history, test_agg, test_target, 0.5)
        print_metrics(appliance_name, rpaf, rete, mae)

    def testing_data_process(self, testing_history, test_target, test_input, mean, std):
        testing_history = ((testing_history * std) + mean)
        test_target = ((test_target * std) + mean)
        test_agg = (test_input.flatten() * mains_data["std"]) + mains_data["mean"]
        test_agg = test_agg[:testing_history.size]
        test_target[test_target < 0] = 0
        testing_history[testing_history < 0] = 0
        return testing_history, test_target, test_agg

    def calculate_metrics(self, testing_history, test_agg, test_target, threshold):
        rpaf = recall_precision_accuracy_f1(testing_history[:test_agg.size - (2 * self.__window_offset)].flatten(),
                                            test_target[:test_agg.size - (2 * self.__window_offset)].flatten(),
                                            threshold)
        rete = relative_error_total_energy(testing_history[:test_agg.size - (2 * self.__window_offset)].flatten(),
                                           test_target[:test_agg.size - (2 * self.__window_offset)].flatten())
        mae = mean_absolute_error(testing_history[:test_agg.size - (2 * self.__window_offset)].flatten(),
                                  test_target[:test_agg.size - (2 * self.__window_offset)].flatten())
        return rpaf, rete, mae

    def print_plots(self, test_agg, test_target, testing_history, count, appliance_name):
        # plt.figure(count, figsize=(300, 300))
        plt.figure(count)
        plt.plot(test_agg[self.__window_offset: -self.__window_offset], label="Aggregate")
        plt.plot(test_target[:test_agg.size - (2 * self.__window_offset)], label="Ground Truth")
        plt.plot(testing_history[:test_agg.size - (2 * self.__window_offset)], label="Predicted")
        plt.title(self.__appliance + " " + appliance_name + " " + self.__model_type)
        plt.ylabel("Power Value (Watts)")
        plt.xlabel("Testing Window")
        plt.legend()
        # plt.savefig("../plot_results/" + self.__predict_mode + "/" + self.__appliance + "_" + appliance_name + ".png")
        plt.show()


def print_metrics(appliance_name, rpaf, rete, mae):
    print("======================================Appliance: {}".format(appliance_name))
    print("============ Recall: {}".format(rpaf[0]))
    print("============ Precision: {}".format(rpaf[1]))
    print("============ Accuracy: {}".format(rpaf[2]))
    print("============ F1 Score: {}".format(rpaf[3]))
    print("============ Relative error in total energy: {}".format(rete))
    print("============ Mean absolute error(in Watts): {}".format(mae))
    print("                                                  ")


def generate_mean_std(appliance_name):
    if 'mean' in appliance_param[appliance_name]:
        mean = appliance_param[appliance_name]['mean']
    else:
        mean = appliance_param['default_param']['mean']
    if 'std' in appliance_param[appliance_name]:
        std = appliance_param[appliance_name]['std']
    else:
        std = appliance_param['default_param']['std']
    return mean, std


def generate_threshold(appliance_name):
    if 'on_power_threshold' in appliance_param[appliance_name]:
        threshold = appliance_param[appliance_name]['on_power_threshold']
    else:
        threshold = appliance_param['default_param']['on_power_threshold']
    return threshold
