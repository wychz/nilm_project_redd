import configparser

cf = configparser.ConfigParser()
cf.read('config.ini', encoding='utf-8')

username = cf.get('mysql', 'username')
password = cf.get('mysql', 'password')
host = cf.get('mysql', 'host')
port = int(cf.get('mysql', 'port'))
db = cf.get('mysql', 'db')

dataset = cf.get('data_process', 'dataset')
raw_data_dir = cf.get('redd', 'raw_data_dir')
aggregate_mean = int(cf.get('redd', 'aggregate_mean'))
aggregate_std = int(cf.get('redd', 'aggregate_std'))

save_path = cf.get('data_process', 'save_path')
main_meter = cf.get('data_process', 'main_meter')

epochs = int(cf.get('train', 'epochs'))
validation_frequency = int(cf.get('train', 'validation_frequency'))
crop = int(cf.get('train', 'crop'))
appliance_name_list = eval(cf.get('train', 'appliance_name_list'))
meter_name_list = eval(cf.get('train', 'meter_name_list'))
predict_mode = cf.get('train', 'predict_mode')
model_type = cf.get('train', 'model_type')
batch_size = int(cf.get('train', 'batch_size'))
input_window_length = int(cf.get('train', 'input_window_length'))


running_param = {
    'database': {
        'username': username,
        'password': password,
        'host': host,
        'port': port,
        'db': db
    },
    'data_process': {
        'dataset': dataset,
        'raw_data_dir': raw_data_dir,             # 数据集位置
        'aggregate_mean': aggregate_mean,              # 总功率的平均值 （用来标准化）
        'aggregate_std': aggregate_std,               # 总功率的标准值 （用来标准化）
        'save_path': save_path,              # 处理后的数据保存位置
        'main_meter': main_meter
    },
    'train_process': {
        'epochs': epochs,               # 训练周期
        'validation_frequency': validation_frequency           # 验证频率
    },
    'crop': crop,             # 一次性加载的数据量
    'appliance_name_list': appliance_name_list,         # 电器列表
    'meter_name_list': meter_name_list,
    'predict_mode': predict_mode,          # 预测模式
    'model_type': model_type,       # 模型选择
    'batch_size': batch_size,          # 一次性训练的数据量
    'input_window_length': input_window_length,          # 训练窗口大小
}


