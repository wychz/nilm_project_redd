from data_process.redd import single_data_generate
from data_process.redd import multiple_data_generate
from running_param import running_param


# 主程序： 指定参数，进行数据生成
appliance_name_list = running_param['appliance_name_list']
predict_mode = running_param['predict_mode']
data_dir = running_param['data_process']['raw_data_dir']
aggregate_mean = running_param['data_process']['aggregate_mean']
aggregate_std = running_param['data_process']['aggregate_std']
save_path = running_param['data_process']['save_path'] + predict_mode + "/"

if predict_mode == 'single':
    single_data_generate.generate(appliance_name_list, data_dir, aggregate_mean, aggregate_std, save_path)
elif predict_mode == 'multiple' or predict_mode == 'multi_label':
    multiple_data_generate.generate(appliance_name_list, data_dir, aggregate_mean, aggregate_std, save_path)