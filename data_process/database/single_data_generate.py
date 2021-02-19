import pandas as pd
import matplotlib.pyplot as plt
import time
from appliance_param import appliance_param


# 训练单个电器 ---- 数据生成
def generate(appliance_name_list, main_meter, aggregate_mean, aggregate_std, save_path, engine):
    start_time = time.time()
    sample_seconds = 60
    validation_percent = 10
    test_percent = 20
    mains = main_meter
    debug = True

    for appliance_name in appliance_name_list:
        print('\n' + appliance_name)
        mains_df = generate_mains(mains, sample_seconds, debug, engine)
        app_df = generate_appliance(appliance_name, sample_seconds, debug, engine)
        df_align = generate_mains_appliance(mains_df, app_df, appliance_name, sample_seconds, debug)
        normalization(df_align, appliance_name, aggregate_mean, aggregate_std)

        # test CSV
        test_len = int((len(df_align) / 100) * test_percent)
        test = df_align.tail(test_len)
        test.reset_index(drop=True, inplace=True)
        df_align.drop(df_align.index[-test_len:], inplace=True)
        test.to_csv(save_path + appliance_name + '_test_' + '.csv', mode='a', index=False, header=False)

        # Validation CSV
        val_len = int((len(df_align) / 100) * validation_percent)
        val = df_align.tail(val_len)
        val.reset_index(drop=True, inplace=True)
        df_align.drop(df_align.index[-val_len:], inplace=True)
        val.to_csv(save_path + appliance_name + '_validation_' + '.csv', mode='a', index=False, header=False)

        # Training CSV
        df_align.to_csv(save_path + appliance_name + '_training_.csv', mode='a', index=False, header=False)

        print("    Size of total training set is {:.4f} M rows.".format(len(df_align) / 10 ** 6))
        print("    Size of total validation set is {:.4f} M rows.".format(len(val) / 10 ** 6))
        print("    Size of total test set is {:.4f} M rows.".format(len(test) / 10 ** 6))
        print("\nPlease find files in: " + save_path)
        print("Total elapsed time: {:.2f} min.".format((time.time() - start_time) / 60))
        del df_align, val


# 生成总功率数据
def generate_mains(meter, sample_seconds, debug, engine):
    mains_df = data_read_database(meter, engine)
    mains_df.rename(columns={"KW": 'aggregate', "timestamp": "time"}, inplace=True)
    mains_df['time'] = mains_df['time'].astype('str')
    mains_df['aggregate'] = mains_df['aggregate'].astype('float64')
    mains_df['aggregate'] = mains_df['aggregate'] * 1000
    mains_df['time'] = pd.to_datetime(mains_df['time'], unit='ms')
    mains_df.set_index('time', inplace=True)

    mains_df = mains_df.resample(str(sample_seconds) + 'S').fillna(method='backfill', limit=1)
    mains_df.reset_index(inplace=True)

    if debug:
        print("    mains_df:")
        print(mains_df.head())
        plt.plot(mains_df['time'], mains_df['aggregate'])
        plt.show()

    return mains_df


# 生成单一电器数据
def generate_appliance(meter, sample_seconds, debug, engine):
    appliance_name = meter
    app_df = data_read_database(meter, engine)
    app_df.rename(columns={"KW": meter, "timestamp": "time"}, inplace=True)
    app_df['time'] = app_df['time'].astype('str')
    app_df[meter] = app_df[meter].astype('float64')
    app_df[meter] = app_df[meter] * 1000
    app_df['time'] = pd.to_datetime(app_df['time'], unit='ms')
    app_df.set_index('time', inplace=True)
    app_df = app_df.resample(str(sample_seconds) + 'S').fillna(method='backfill', limit=1)
    app_df.reset_index(inplace=True)
    if debug:
        print("app_df:")
        print(app_df.head())
        plt.plot(app_df['time'], app_df[appliance_name])
        plt.show()
    return app_df


# 读取数据库中的 timestamp 和 KW 数据
def data_read_database(meter, engine):
    meter_table_name = 'meter_{}_source'.format(meter)
    sql_query = "select timestamp, KW from {}".format(meter_table_name)
    df_read = pd.read_sql_query(sql_query, engine)
    return df_read


# 拼装总功率和单一电器的功率
def generate_mains_appliance(mains_df, app_df, appliance_name, sample_seconds, debug):
    mains_df.set_index('time', inplace=True)
    app_df.set_index('time', inplace=True)
    df_align = mains_df.join(app_df, how='outer').resample(str(sample_seconds) + 'S').fillna(method='backfill', limit=1)
    df_align = df_align.dropna()
    df_align.reset_index(inplace=True)
    df_align['time'] = df_align['time'].astype('str')
    df_align['aggregate'] = df_align['aggregate'].astype('float64')
    df_align[appliance_name] = df_align[appliance_name].astype('float64')
    if debug:
        print("df_align_time:")
        print(df_align.head())
    del mains_df, app_df, df_align['time']
    if debug:
        print("df_align:")
        print(df_align.head())
        plt.plot(df_align['aggregate'].values)
        plt.plot(df_align[appliance_name].values)
        plt.show()
    return df_align


# 归一化
def normalization(df_align, appliance_name, aggregate_mean, aggregate_std):
    # Normalization
    if appliance_name in appliance_param:
        if 'mean' in appliance_param[appliance_name]:
            mean = appliance_param[appliance_name]['mean']
        else:
            mean = appliance_param['default_param']['mean']

        if 'std' in appliance_param[appliance_name]:
            std = appliance_param[appliance_name]['std']
        else:
            std = appliance_param['default_param']['std']
    else:
        mean = appliance_param['default_param']['mean']
        std = appliance_param['default_param']['std']

    df_align['aggregate'] = (df_align['aggregate'] - aggregate_mean) / aggregate_std
    df_align[appliance_name] = (df_align[appliance_name] - mean) / std
