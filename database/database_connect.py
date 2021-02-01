import pandas as pd
from sqlalchemy import create_engine
import numpy as np


def convert(x):
    timestamp = x.timestamp() * 1000
    return timestamp


# 请修改1, 2两处的数据
# 1. 数据分别为用户名，密码，ip，port，数据库名
engine = create_engine("mysql+pymysql://{}:{}@{}:{}/{}".format('root', 'root', 'localhost', '3306', 'energydb'))
# 2. 要进行数据清洗的电表号码
meter_list = [1, 2, 3, 4, 5, 6, 7, 101, 102]
param_list = ['KW', 'V', 'A']

for meter in meter_list:
    source_table_name = 'meter_{}_source'.format(meter)
    target_table_name = 'meter_{}_deal'.format(meter)
    sql_query = 'select * from {}'.format(source_table_name)
    df_read = pd.read_sql_query(sql_query, engine)
    df_read['time'] = pd.to_datetime(df_read['timestamp'], unit='ms')
    df_read.set_index('time', inplace=True)
    df_read = df_read.resample('60S').fillna(method='backfill', limit=1)
    df_read.reset_index(inplace=True)
    df_read['timestamp'] = df_read['time'].apply(lambda x: convert(x))
    del df_read['time']
    for param in param_list:
        df_read[param] = df_read[param].astype(np.float64)
        df_read[param] = df_read[param].fillna((df_read[param].fillna(method='ffill', limit=1) + df_read[param].fillna(method='bfill', limit=1)) / 2)
    df_read = df_read.fillna(method="backfill", limit=1)
    df_read.dropna(subset=['meter'], inplace=True)
    df_read.to_sql(target_table_name, con=engine, if_exists='replace', index=False)

print('success')
