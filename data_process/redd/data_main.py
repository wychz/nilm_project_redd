from data_process.redd import single_data_generate
from data_process.redd import multiple_data_generate
import running_param


# 主程序： 指定参数，进行数据生成
appliance_name_list = running_param.appliance_name_list
predict_mode = running_param.predict_mode
raw_data_dir = running_param.raw_data_dir
aggregate_mean = running_param.aggregate_mean
aggregate_std = running_param.aggregate_std
save_path = running_param.save_path + predict_mode + "/"


def redd_data_process():
    if predict_mode == 'single':
        single_data_generate.generate(appliance_name_list, raw_data_dir, aggregate_mean, aggregate_std, save_path)
    elif predict_mode == 'multiple':
        multiple_data_generate.generate(appliance_name_list, raw_data_dir, aggregate_mean, aggregate_std, save_path)