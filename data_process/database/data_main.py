from data_process.database import single_data_generate
from data_process.database import multiple_data_generate
from running_param import running_param
from sqlalchemy import create_engine

# 主程序： 指定参数，进行数据生成
appliance_name_list = running_param['meter_name_list']
predict_mode = running_param['predict_mode']
data_dir = running_param['data_process']['raw_data_dir']
aggregate_mean = running_param['data_process']['aggregate_mean']
aggregate_std = running_param['data_process']['aggregate_std']
save_path = running_param['data_process']['save_path'] + predict_mode + "/"
main_meter = running_param['data_process']['main_meter']

username = running_param['database']['username']
password = running_param['database']['password']
host = running_param['database']['host']
port = running_param['database']['port']
db = running_param['database']['db']

engine = create_engine("mysql+pymysql://{}:{}@{}:{}/{}".format(username, password, host, port, db))


def database_data_process():
    if predict_mode == 'single':
        single_data_generate.generate(appliance_name_list, main_meter, aggregate_mean, aggregate_std, save_path, engine)
    elif predict_mode == 'multiple' or predict_mode == 'multi_label':
        multiple_data_generate.generate(appliance_name_list, main_meter, aggregate_mean, aggregate_std, save_path, engine)