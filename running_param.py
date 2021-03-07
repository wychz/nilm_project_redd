import configparser

cf = configparser.ConfigParser()
cf.read('config.ini', encoding='utf-8')

username = cf.get('mysql', 'username')
password = cf.get('mysql', 'password')
host = cf.get('mysql', 'host')
port = int(cf.get('mysql', 'port'))
db = cf.get('mysql', 'db')

dataset = cf.get('data_process', 'dataset')

save_path = cf.get('data_process', 'save_path')
main_meter = cf.get('data_process', 'main_meter')

epochs = int(cf.get('train', 'epochs'))
validation_frequency = int(cf.get('train', 'validation_frequency'))
meter_name_list = eval(cf.get('train', 'meter_name_list'))
predict_mode = cf.get('train', 'predict_mode')
model_type = cf.get('train', 'model_type')
batch_size = int(cf.get('train', 'batch_size'))
input_window_length = int(cf.get('train', 'input_window_length'))
validation_percent = int(cf.get('train', 'validation_percent'))
test_percent = int(cf.get('train', 'test_percent'))
sample_seconds = int(cf.get('train', 'sample_seconds'))
learning_rate = float(cf.get('train', 'learning_rate'))
is_load_model = cf.getboolean('train', 'is_load_model')

on_power_threshold = cf.getint('data', 'on_power_threshold')

appliance_name_list = eval(cf.get('train', 'appliance_name_list'))
aggregate_mean = int(cf.get('redd', 'aggregate_mean'))
aggregate_std = int(cf.get('redd', 'aggregate_std'))
raw_data_dir = cf.get('redd', 'raw_data_dir')
