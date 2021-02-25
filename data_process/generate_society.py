import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer

data_types = ['training', 'test', 'validation']
appliance_name_list = ['dishwasher', 'fridge', 'microwave', 'washingmachine']
predict_types = ['single', 'multiple', 'multi_label']
centigrade = [16, 17, 18, 19, 20]
people = [1, 2, 3, 4, 5]
is_workday = [0, 1]
columns = ['aggregate', 'centigrade', 'people', 'is_workday', 'appliance']

for appliance_name in appliance_name_list:
    for data_type in data_types:
        file_path = "./redd/processed_dataset/1min_csv/single/" + appliance_name + "_" + data_type + "_.csv"
        df_data = pd.read_csv(file_path, names=['aggregate', 'appliance'])
        df_data['centigrade'] = np.random.choice(centigrade, len(df_data))
        df_data['people'] = np.random.choice(people, len(df_data))
        df_data['is_workday'] = np.random.choice(is_workday, len(df_data))
        df_data = df_data.reindex(columns=columns)
        df_data.to_csv(file_path, index=False, header=False)

# for appliance_name in appliance_name_list:
#     for data_type in data_types:
#         file_path = "./redd/processed_dataset/1min_csv/multiple/" + 'all' + "_" + data_type + "_.csv"
#         df_data = pd.read_csv(file_path, names=['aggregate', 'appliance'])
#         df_data['centigrade'] = np.random.choice(centigrade, len(df_data))
#         df_data['people'] = np.random.choice(people, len(df_data))
#         df_data['is_workday'] = np.random.choice(is_workday, len(df_data))
#         df_data = df_data.reindex(columns=columns)
#         df_data.to_csv(file_path, index=False, header=False)
#
# for appliance_name in appliance_name_list:
#     for data_type in data_types:
#         file_path = "./redd/processed_dataset/1min_csv/multi_label/" + 'all' + "_" + data_type + "_.csv"
#         df_data = pd.read_csv(file_path, names=['aggregate', 'appliance'])
#         df_data['centigrade'] = np.random.choice(centigrade, len(df_data))
#         df_data['people'] = np.random.choice(people, len(df_data))
#         df_data['is_workday'] = np.random.choice(is_workday, len(df_data))
#         df_data = df_data.reindex(columns=columns)
#         df_data.to_csv(file_path, index=False, header=False)