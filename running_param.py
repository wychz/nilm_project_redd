running_param = {
    'dataset': 'redd',
    'data_process': {
        'raw_data_dir': 'raw_dataset/low_freq',             # 数据集位置
        'aggregate_mean': 522,              # 总功率的平均值 （用来标准化）
        'aggregate_std': 814,               # 总功率的标准值 （用来标准化）
        'save_path': 'processed_dataset/1min_csv/',              # 处理后的数据保存位置
        'main_meter': '160327039'
    },
    'train_process': {
        'epochs': 10,               # 训练周期
        'validation_frequency': 1           # 验证频率
    },
    'crop': 500000,             # 一次性加载的数据量
    'appliance_name_list': ['microwave'],         # 电器列表
    'meter_name_list': ['70213', '489190910751'],
    'predict_mode': 'single',          # 预测模式
    'model_type': 'lstm',       # 模型选择
    'batch_size': 128,          # 一次性训练的数据量
    'input_window_length': 19,          # 训练窗口大小
}

# predict_mode支持三种：single, multiple, multi_label
# model_type支持三种： lstm, cnn, resnet（resnet可能有问题）
# input_window_length: 常用的是cnn的599，lstm的19
# appliance_name_list: microwave, fridge, dishwasher, washingmachine，其他的目前尚有问题

# 使用方法
# 1. 调整 running_param.py 的参数，即此文件的参数
# 2. 处理原始数据：
#     2.1 运行 data_main.py，会自动将处理过的数据保存到processed_dataset中。
#     2.2 运行 data_to_label，会自动将
