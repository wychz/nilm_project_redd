import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np
from appliance_param import appliance_param
from appliance_param import multiple_data


def generate(appliance_name_list, data_dir, save_path):
    start_time = time.time()
    sample_seconds = 60
    nrows = None
    debug = True

    columns = ['aggregate']
    for appliance_name in appliance_name_list:
        columns.append(appliance_name)
    columns_array = np.array(columns)
    train = pd.DataFrame(columns=columns_array)

    for h in multiple_data['houses']:
        mains_df = generate_mains(data_dir, nrows, sample_seconds, debug, h)
        app_df = generate_appliance(appliance_name_list, data_dir, nrows, sample_seconds, debug, h)
        df_align = generate_mains_appliance(mains_df, app_df, appliance_name_list, sample_seconds, debug)

        train = train.append(df_align, ignore_index=True)
        del df_align

    cols = list(train)
    cols.insert(0, cols.pop(cols.index('time')))
    train = train.ix[:, cols]
    train = train.round(0)


    train.to_csv(save_path + 'all' + '_training_.csv', mode='a', index=False, header=True)

    print("    Size of total training set is {:.4f} M rows.".format(len(train) / 10 ** 6))
    del train
    print("\nPlease find files in: " + save_path)
    print("Total elapsed time: {:.2f} min.".format((time.time() - start_time) / 60))


def generate_mains(data_dir, nrows, sample_seconds, debug, h):
    mains1_df = pd.read_table(data_dir + '/' + 'house_' + str(h) + '/' + 'channel_' + str(1) + '.dat',
                              sep="\s+", nrows=nrows, usecols=[0, 1], names=['time', 'mains1'], dtype={'time': str}, )
    mains2_df = pd.read_table(data_dir + '/' + 'house_' + str(h) + '/' + 'channel_' + str(2) + '.dat',
                              sep="\s+", nrows=nrows, usecols=[0, 1], names=['time', 'mains2'], dtype={'time': str}, )

    mains1_df['time'] = pd.to_datetime(mains1_df['time'], unit='s')
    mains2_df['time'] = pd.to_datetime(mains2_df['time'], unit='s')
    mains1_df.set_index('time', inplace=True)
    mains2_df.set_index('time', inplace=True)
    mains_df = mains1_df.join(mains2_df, how='outer')
    mains_df['aggregate'] = mains_df.iloc[:].sum(axis=1)
    mains_df.reset_index(inplace=True)

    # 此处重采样
    mains_df.set_index('time', inplace=True)
    mains_df = mains_df.resample(str(sample_seconds) + 'S').fillna(method='backfill', limit=1)
    mains_df.reset_index(inplace=True)
    del mains_df['mains1'], mains_df['mains2']

    if debug:
        print("    mains_df:")
        print(mains_df.head())
        plt.plot(mains_df['time'], mains_df['aggregate'])
        plt.show()

    del mains1_df, mains2_df
    return mains_df


def generate_appliance(appliance_name_list, data_dir, nrows, sample_seconds, debug, h):
    app_df_list = []
    for i in range(len(appliance_name_list)):
        appliance_name = appliance_name_list[i]
        app_df = pd.read_table(data_dir + '/' + 'house_' + str(h) + '/' + 'channel_' + str(
            appliance_param[appliance_name]['channels'][appliance_param[appliance_name]['houses'].index(h)]) + '.dat',
                               sep="\s+", nrows=nrows, usecols=[0, 1], names=['time', appliance_name],
                               dtype={'time': str}, )
        app_df['time'] = pd.to_datetime(app_df['time'], unit='s')
        app_df.set_index('time', inplace=True)
        app_df = app_df.resample(str(sample_seconds) + 'S').fillna(method='backfill', limit=1)
        app_df.reset_index(inplace=True)
        app_df_list.append(app_df)
        app_df_list[i].set_index('time', inplace=True)

    app_df = app_df_list[0]
    for i in range(1, len(app_df_list)):
        app_df = app_df.join(app_df_list[i])
    del app_df_list
    app_df.reset_index(inplace=True)
    app_df.set_index('time', inplace=True)
    app_df = app_df.resample(str(sample_seconds) + 'S').fillna(method='backfill', limit=1)
    app_df.reset_index(inplace=True)
    if debug:
        print("app_df:")
        print(app_df.head())
        for appliance_name in appliance_name_list:
            plt.plot(app_df['time'], app_df[appliance_name])
        plt.show()
    return app_df


def generate_mains_appliance(mains_df, app_df, appliance_name_list, sample_seconds, debug):
    mains_df.set_index('time', inplace=True)
    app_df.set_index('time', inplace=True)
    df_align = mains_df.join(app_df, how='outer').resample(str(sample_seconds) + 'S').fillna(method='backfill', limit=1)
    df_align.reset_index(inplace=True)
    if debug:
        print("df_align_time:")
        print(df_align.head())
    del mains_df, app_df
    if debug:
        print("df_align:")
        print(df_align.head())
        for appliance_name in appliance_name_list:
            plt.plot(df_align[appliance_name].values)
        plt.show()
    return df_align


appliance_name_list = ['microwave', 'fridge', 'dishwasher', 'washingmachine']
data_dir = 'raw_dataset/low_freq'
save_path = 'processed_dataset/1min_csv/' + 'data_output/'
generate(appliance_name_list, data_dir, save_path)
