import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer

data_types = ['training', 'test', 'validation']
appliance_name_list = ['dishwasher', 'fridge', 'microwave', 'washingmachine']

for appliance_name in appliance_name_list:

    train_file_path = "./redd/processed_dataset/1min_csv/single/" + appliance_name + "_{}".format('training') + "_.csv"
    test_file_path = "./redd/processed_dataset/1min_csv/single/" + appliance_name + "_{}".format('test') + "_.csv"
    validation_file_path = "./redd/processed_dataset/1min_csv/single/" + appliance_name + "_{}".format('validation') + "_.csv"

    df_train = pd.read_csv(train_file_path, names=['aggregate', 'centigrade', 'people', 'is_workday', 'appliance'])
    df_test = pd.read_csv(test_file_path, names=['aggregate', 'centigrade', 'people', 'is_workday', 'appliance'])
    df_validation = pd.read_csv(validation_file_path, names=['aggregate', 'centigrade', 'people', 'is_workday', 'appliance'])

    continues = ['centigrade', 'people']
    cs = MinMaxScaler()
    df_train[continues] = cs.fit_transform(df_train[continues])
    df_test[continues] = cs.transform(df_test[continues])
    df_validation[continues] = cs.transform(df_validation[continues])

    lb = LabelBinarizer().fit(df_train['is_workday'])
    df_train['is_workday'] = lb.transform(df_train['is_workday'])
    df_test['is_workday'] = lb.transform(df_test['is_workday'])
    df_validation['is_workday'] = lb.transform(df_validation['is_workday'])

    df_train.to_csv(train_file_path, index=False, header=False)
    df_test.to_csv(test_file_path, index=False, header=False)
    df_validation.to_csv(validation_file_path, index=False, header=False)
