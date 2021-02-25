import pandas as pd
import numpy as np
from appliance_param import appliance_param
import running_param


# 将电器功率数据转换为0,1开关状态
appliance_name_list = running_param.appliance_name_list
data_dir = 'data_process/redd/processed_dataset/1min_csv/multiple/'
save_path = 'data_process/redd/processed_dataset/1min_csv/multi_label/'
predict_mode = 'multi_label'


def data_process_redd_multi_label():
    if predict_mode == 'single_label':
        for appliance_name in appliance_name_list:
            df_test = pd.read_csv(data_dir + appliance_name + "_test_.csv", usecols=[0, 1], names=['time', 'test'], dtype={'time': str})
            df_training = pd.read_csv(data_dir + appliance_name + "_training_.csv", usecols=[0, 1], names=['time', 'training'], dtype={'time': str})
            df_validation = pd.read_csv(data_dir + appliance_name + "_validation_.csv", usecols=[0, 1], names=['time', 'validation'], dtype={'time': str})

            test_array = np.round(np.array(df_test.iloc[:, 1], float), 6)
            training_array = np.round(np.array(df_training.iloc[:, 1], float), 6)
            validation_array = np.round(np.array(df_validation.iloc[:, 1], float), 6)

            std = appliance_param[appliance_name]["std"]
            mean = appliance_param[appliance_name]["mean"]
            on_power_threshold = appliance_param[appliance_name]['on_power_threshold']
            test_array = ((test_array * std) + mean)
            training_array = ((training_array * std) + mean)
            validation_array = ((validation_array * std) + mean)

            test_array[test_array < on_power_threshold] = 0
            test_array[test_array > 0] = 1

            training_array[training_array < on_power_threshold] = 0
            training_array[training_array > 0] = 1

            validation_array[validation_array < on_power_threshold] = 0
            validation_array[validation_array > 0] = 1

            df_test.loc[:, 'test'] = test_array
            df_training.loc[:, 'training'] = training_array
            df_validation.loc[:, 'validation'] = validation_array

            df_test.to_csv(save_path + appliance_name + '_test_.csv', index=False, header=False)
            df_training.to_csv(save_path + appliance_name + '_training_.csv', index=False, header=False)
            df_validation.to_csv(save_path + appliance_name + '_validation_.csv', index=False, header=False)

    elif predict_mode == 'multi_label':

        names_list = ['mains']
        for appliance_name in appliance_name_list:
            names_list.append(appliance_name)
        names_array = np.array(names_list)

        df_test_all = pd.read_csv(data_dir + 'all' + "_test_.csv", names=names_array, dtype={'time': str})
        df_training_all = pd.read_csv(data_dir + 'all' + "_training_.csv", names=names_array, dtype={'time': str})
        df_validation_all = pd.read_csv(data_dir + 'all' + "_validation_.csv", names=names_array, dtype={'time': str})

        for i, appliance_name in enumerate(appliance_name_list):
            test_array = np.round(np.array(df_test_all.iloc[:, i + 1], float), 6)
            training_array = np.round(np.array(df_training_all.iloc[:, i + 1], float), 6)
            validation_array = np.round(np.array(df_validation_all.iloc[:, i + 1], float), 6)

            std = appliance_param[appliance_name]["std"]
            mean = appliance_param[appliance_name]["mean"]
            on_power_threshold = appliance_param[appliance_name]['on_power_threshold']
            test_array = ((test_array * std) + mean)
            training_array = ((training_array * std) + mean)
            validation_array = ((validation_array * std) + mean)

            test_array[test_array < on_power_threshold] = 0
            test_array[test_array > 0] = 1

            training_array[training_array < on_power_threshold] = 0
            training_array[training_array > 0] = 1

            validation_array[validation_array < on_power_threshold] = 0
            validation_array[validation_array > 0] = 1

            df_test_all.loc[:, appliance_name] = test_array
            df_training_all.loc[:, appliance_name] = training_array
            df_validation_all.loc[:, appliance_name] = validation_array

        df_test_all.to_csv(save_path + 'all' + '_test_.csv', index=False, header=False)
        df_training_all.to_csv(save_path + 'all' + '_training_.csv', index=False, header=False)
        df_validation_all.to_csv(save_path + 'all' + '_validation_.csv', index=False, header=False)